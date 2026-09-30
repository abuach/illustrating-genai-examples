"""Verification: from "passes the tests" to "provably correct."

A test samples a handful of inputs and checks the answer on each. A verifier
does the opposite: it searches *every* input for one that breaks a stated rule,
and reports a concrete counterexample when it finds one. The rule is a contract,
a precondition on a function's inputs and a postcondition on its result, and the
checker here is CrossHair, which explores a function symbolically (the same kind
of SMT search the Reasoning chapter handed to Z3, now pointed at code instead of
a puzzle).

The split of labour mirrors the rest of Part II. The model does the fluent part,
*writing* the contract in words turned into ``pre:``/``post:`` lines; CrossHair
does the exact part, deciding whether the code actually obeys it. That's also
where the model fails in a new way: it writes confident, plausible specifications
that are subtly false, and the verifier is what tells a true spec from a fluent
one. The model is gpt-oss:20b; contracts are nondeterministic, so each is
captured once by ``scripts/_verify_probe.py`` and baked below. CrossHair's
verdict is deterministic and baked alongside it for byte-stability.
"""
import importlib.util
import re
import sys
import tempfile
import uuid
from pathlib import Path

import ollama

from genai.arch import EIS_MODEL

# The functions we ask the model to specify. We own the implementation and its
# type signature (CrossHair needs the hints to know what symbolic inputs to
# generate); the model writes only the contract. Each is a few lines a reader can
# check by hand, so the verifier's verdict can be confirmed without trusting it.
IMPLS = {
    "midpoint": "def midpoint(a: int, b: int) -> int:\n    return (a + b) // 2",
    "clamp": ("def clamp(x: int, lo: int, hi: int) -> int:\n"
              "    return max(lo, min(x, hi))"),
    "average": ("def average(nums: list) -> float:\n"
                "    return sum(nums) / len(nums)"),
}


def _normalize(contract: str) -> str:
    """Re-join a contract folded across lines for the page into one line per rule.

    A long ``post:`` is wrapped at ``or``/``and`` boundaries so it fits a book
    column, but CrossHair's PEP 316 reader wants each ``pre:``/``post:`` on a
    single line. Any line that doesn't begin a new rule is appended to the rule
    above it, so the display form and the checked form are the same expression.
    """
    out = []
    for line in contract.strip().split("\n"):
        if re.match(r"\s*(pre|post)\s*:", line, re.I) or not out:
            out.append(line.strip())
        else:
            out[-1] += " " + line.strip()
    return "\n".join(out)


def _assemble(impl: str, contract: str) -> str:
    """Insert a contract as the function's docstring so CrossHair can read it.

    CrossHair's PEP 316 mode looks for ``pre:``/``post:`` lines in the docstring
    of a function defined in a real source file, so we splice the model's
    contract under the ``def`` line and write the result to disk in ``check``.
    """
    head, *body = impl.split("\n")
    pad = "    "
    doc = "\n".join(pad + line for line in _normalize(contract).split("\n"))
    return f'{head}\n{pad}"""\n{doc}\n{pad}"""\n' + "\n".join(body)


def _parse(messages) -> dict:
    """Turn CrossHair's analysis messages into a plain verdict.

    A postcondition failure or an unhandled exception means the contract is
    *refuted*: there is an input satisfying the precondition that breaks the
    postcondition, and the message names it. A clean "confirmed over all paths"
    means *confirmed*. Anything else (a timeout, an unusable spec) is unknown.
    """
    from crosshair.core_and_libs import MessageType
    fails = [m for m in messages
             if m.state in (MessageType.POST_FAIL, MessageType.EXEC_ERR)]
    if fails:
        text = fails[0].message
        call = re.search(r"calling (\w+\([^)]*\))", text)
        ret = re.search(r"returns (.+?)\)?$", text)
        return {"status": "refuted",
                "witness": call.group(1) if call else text,
                "returns": ret.group(1).rstrip(")") if ret else None,
                "message": text}
    if any(m.state == MessageType.CONFIRMED for m in messages):
        return {"status": "confirmed", "witness": None, "returns": None,
                "message": "Confirmed over all paths."}
    return {"status": "unknown", "witness": None, "returns": None,
            "message": "No verdict in the time given."}


def check_src(impl: str, contract: str, timeout: float = 10.0) -> dict:
    """Run CrossHair on an implementation passed directly, carrying ``contract``.

    Materialises the implementation plus the contract as a throwaway module,
    hands the function to CrossHair, and reduces its messages to one of
    ``refuted`` (with a witness), ``confirmed``, or ``unknown``. ``check`` is the
    thin wrapper that looks the implementation up in ``IMPLS``; this form takes
    the source directly, so the same contract can be aimed at a *different* body
    (a deliberately broken one) to show what a weak spec fails to catch.
    """
    from crosshair.core_and_libs import (AnalysisKind, analyze_function,
                                         run_checkables)
    from crosshair.options import AnalysisOptionSet

    name = impl.split("(", 1)[0].split()[-1]                 # def NAME(...)
    source = _assemble(impl, contract)
    modname = f"_verify_{name}_{uuid.uuid4().hex[:8]}"
    path = Path(tempfile.gettempdir()) / f"{modname}.py"
    path.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        opts = AnalysisOptionSet(per_condition_timeout=timeout,
                                 analysis_kind=[AnalysisKind.PEP316])
        messages = run_checkables(analyze_function(getattr(module, name), opts))
        return _parse(messages)
    finally:
        sys.modules.pop(modname, None)
        path.unlink(missing_ok=True)


def check(name: str, contract: str, timeout: float = 10.0) -> dict:
    """Run CrossHair on ``IMPLS[name]`` carrying ``contract`` and return a verdict.

    Reduces CrossHair's messages to one of ``refuted`` (with a witness),
    ``confirmed``, or ``unknown``. Deterministic for the tiny functions here; the
    demo bakes the verdict anyway.
    """
    return check_src(IMPLS[name], contract, timeout)


def specify(name: str, model: str = EIS_MODEL) -> str:
    """Ask the model to write a contract for ``IMPLS[name]``; return the lines.

    We push for the *strongest* postcondition on purpose: that's the prompt that
    tempts a fluent model into over-claiming, which is the failure the verifier
    exists to catch. gpt-oss often empties ``content`` and answers in its
    thinking channel, so we scan both and keep the final ``pre:``/``post:`` block.
    """
    prompt = (
        "Here is a Python function:\n\n" + IMPLS[name] + "\n\n"
        "Write a contract for it in PEP 316 style: an optional `pre:` line for any "
        "precondition on the inputs, and a `post:` line for the postcondition on "
        "the result, where `__return__` names the returned value. Give the "
        "strongest postcondition you can. Output only the contract lines.")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    lines = re.findall(r"^\s*((?:pre|post)\s*:.*)$", text, re.M | re.I)
    return "\n".join(line.strip() for line in lines[-2:]) if lines else text.strip()


# Both captured by scripts/_verify_probe.py (gpt-oss:20b + CrossHair): the model's
# REAL contracts, kept verbatim, and CrossHair's REAL verdicts. Same model, same
# "give me the strongest contract" prompt, opposite outcomes, and the difference
# isn't visible from reading the specs.
#
# The win. Asked to specify the integer midpoint, the model writes an
# order-independent bound (min/max, not the naive a <= r <= b) and even pins down
# the rounding. CrossHair confirms it over every path: a real guarantee.
CONFIRM_DEMO = {
    "name": "midpoint",
    "impl": IMPLS["midpoint"],
    "contract": ("post: min(a,b) <= __return__ <= max(a,b)\n"
                 "post: 2*__return__ <= a + b < (2*(__return__ + 1))"),
    "verdict": {"status": "confirmed", "witness": None, "returns": None},
}

# The catch. Asked to specify clamp, the model writes an elaborate, rigorous-
# looking case analysis, but it won't commit to a precondition (its `pre:` line is
# the model's own hedge), so when lo > hi the cases don't hold. CrossHair returns
# the witness clamp(0, -1, -2). The post is one logical line, folded at `or` here
# to fit the page; _normalize rejoins it for the checker. This is the chapter's
# centerpiece: a fluent spec that reads as careful and is false.
SPEC_DEMO = {
    "name": "clamp",
    "impl": IMPLS["clamp"],
    "contract": ("pre:  # optional; maybe include lo <= hi or not.\n"
                 "post: ((x <= lo) and (__return__ == lo))\n"
                 "  or  ((hi <= x) and (__return__ == hi))\n"
                 "  or  (lo < x < hi and __return__ == x)"),
    "verdict": {"status": "refuted", "witness": "clamp(0, -1, -2)", "returns": "-1"},
}


def _verdict_line(verdict: dict) -> str:
    """One line of plain CrossHair output: a counterexample, or a confirmation."""
    if verdict["status"] == "refuted":
        if verdict.get("returns") is None and "Error" in (verdict.get("message") or ""):
            exc = verdict["message"].split(":", 1)[0]
            return (f"refuted: {verdict['witness']} raises {exc}, an input the "
                    "contract never ruled out")
        return (f"refuted: {verdict['witness']} returns {verdict['returns']}, "
                "which breaks the post-condition")
    if verdict["status"] == "confirmed":
        return "confirmed over all paths: no input can break the contract"
    return "no verdict in the time given"


def show_spec(demo: dict = SPEC_DEMO) -> None:
    """The verifier catching a fluent-but-false spec: the function, the model's
    contract, and the counterexample CrossHair returns."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["impl"])
    show_code("gpt-oss", demo["contract"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict"]))


def show_confirm(demo: dict = CONFIRM_DEMO) -> None:
    """The other outcome: a true contract on correct code, proven over all inputs."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["impl"])
    show_code("gpt-oss", demo["contract"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict"]))


# ── Deductive verification: prove a loop with an invariant ─────────────────────
# CrossHair confirmed midpoint and clamp because they have no loops: a few paths,
# fully explored. A loop is where exhaustive checking stops, because the trip
# count isn't bounded. Deductive verification handles it by reasoning about the
# loop once, through an *invariant*: a property true before the loop and preserved
# by every pass, so it still holds however many times the loop runs. The verifier
# discharges three obligations (the invariant holds on entry, is preserved by the
# body, and is strong enough at exit to force the postcondition) and a solver
# decides each. This is what Dafny and Frama-C do under their surface languages;
# we do the three obligations by hand and let Z3 discharge them, so they're visible.

# times(a, b): a*b by repeated addition, for b >= 0. Shown verbatim; the loop is
# what makes this a deductive problem rather than a CrossHair one.
TIMES_SRC = ("def times(a, b):              # b >= 0\n"
             "    result = 0\n"
             "    k = 0\n"
             "    while k < b:\n"
             "        result += a\n"
             "        k += 1\n"
             "    return result")


def check_invariant(inv_src: str) -> dict:
    """Discharge the three verification conditions for ``times`` against ``inv_src``.

    ``inv_src`` is a Z3 boolean expression over a, b, result, k (the model writes
    it). We encode the loop's one-step effect (result += a, k += 1) and ask Z3
    whether the invariant survives each obligation. A failing obligation comes back
    with the concrete state that breaks it.
    """
    import z3
    a, b, result, k = z3.Ints("a b result k")
    ns = dict(z3.__dict__); ns.update(a=a, b=b, result=result, k=k)
    inv = eval(inv_src, ns)                                   # noqa: S307 - demo
    inv_next = eval(inv_src, {**ns, "result": result + a, "k": k + 1})
    pre, post = b >= 0, result == a * b

    def discharge(assumptions, goal):
        solver = z3.Solver(); solver.add(*assumptions, z3.Not(goal))
        if solver.check() == z3.unsat:
            return {"ok": True, "cex": None}
        m = solver.model()
        return {"ok": False,
                "cex": ", ".join(f"{v}={m.eval(v)}" for v in (a, b, result, k))}

    obligations = {
        "init": discharge([pre, result == 0, k == 0], inv),
        "preserved": discharge([inv, k < b], inv_next),
        "post": discharge([inv, z3.Not(k < b)], post),
    }
    ok = all(o["ok"] for o in obligations.values())
    return {"status": "verified" if ok else "refuted", "obligations": obligations}


def propose_invariant(model: str = EIS_MODEL) -> str:
    """Ask the model for a loop invariant for ``times``, as a Z3 expression.

    Same shape as the Reasoning chapter's ``formalize``: the model writes Z3, we
    run it. The invariant has to relate the accumulator to the counter AND bound
    the counter, and a fluent guess often gets the relation and forgets the bound,
    which is the failure the post-condition obligation catches.
    """
    prompt = (
        "Here is a Python function that multiplies by repeated addition:\n\n"
        + TIMES_SRC + "\n\n"
        "Give a loop invariant strong enough to prove the function returns a*b. "
        "Write it as a single Z3 boolean expression over the variables a, b, "
        "result, k, using And(...) for conjunction (e.g. And(result == a*k, ...)). "
        "Output only the expression.")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    hits = re.findall(r"And\(.*?\)\s*\)", text) or re.findall(r"And\(.*\)", text)
    return hits[-1].strip() if hits else text.strip().splitlines()[-1]


# Captured by scripts/_verify_probe.py (gpt-oss:20b + Z3). The model's REAL loop
# invariant, kept verbatim, and Z3's REAL verdict on the three obligations. The
# win at the top of the ladder: the invariant relates the running total to the
# counter and bounds the counter, so all three obligations discharge and the loop
# is proved correct for every a and every b >= 0, something no finite test or
# path-by-path search could establish.
INVARIANT_DEMO = {
    "src": TIMES_SRC,
    "invariant": "And(result == a*k, k >= 0, k <= b)",
    "verdict": check_invariant("And(result == a*k, k >= 0, k <= b)"),
}


_OBLIGATION_TEXT = {
    "init": ("holds: true before the loop runs",
             "fails: {cex} breaks it before the loop"),
    "preserved": ("holds: one pass of the loop keeps it true",
                  "fails: {cex} satisfies it but one pass breaks it"),
    "post": ("holds: with the loop done, it forces result == a*b",
             "fails: {cex} satisfies it but not result == a*b"),
}
_OBLIGATION_LABEL = {"init": "ENTRY", "preserved": "STEP", "post": "EXIT"}


def show_invariant(demo: dict = INVARIANT_DEMO) -> None:
    """The model writes a loop invariant; Z3 discharges the three obligations and
    proves the loop correct for every input, or names the state that breaks one."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["src"])
    show_code("gpt-oss", demo["invariant"])
    for key in ("init", "preserved", "post"):
        result = demo["verdict"]["obligations"][key]
        held, failed = _OBLIGATION_TEXT[key]
        line = held if result["ok"] else failed.format(cex=result["cex"])
        show_turn(_OBLIGATION_LABEL[key], line)
    if demo["verdict"]["status"] == "verified":
        show_turn("Z3", "proved: result == a*b for every a and every b >= 0")
    else:
        show_turn("Z3", "refuted: the invariant is too weak to prove the result")


# ── Termination: proving the loop actually stops ───────────────────────────────
# The invariant proved *partial* correctness: IF times halts, it returns a*b. It
# said nothing about whether times halts at all. Termination needs a second
# artifact, a *variant* (ranking function): an integer measure that stays >= 0
# while the loop keeps running and strictly falls on every pass. A non-negative
# integer can't fall forever, so a loop with a valid variant must stop. Two
# obligations, discharged by the same Z3 pattern as the invariant.

# A loop that stops only for non-negative input. When n < 0 the guard n != 0
# never goes false and the loop runs forever; the missing precondition is the
# bug, the same shape as the one that sank the clamp contract.
COUNTDOWN_SRC = ("def countdown(n):        # intended for n >= 0\n"
                 "    while n != 0:\n"
                 "        n -= 1\n"
                 "    return n")

# Each loop as the checker sees it: its precondition, its guard, the one-step
# effect on each variable, and which variables to name in a counterexample.
LOOPS = {
    "times": {"src": TIMES_SRC, "pre": "b >= 0", "guard": "k < b",
              "step": {"result": "result + a", "k": "k + 1"},
              "vars": ("a", "b", "result", "k")},
    "countdown": {"src": COUNTDOWN_SRC, "pre": "True", "guard": "n != 0",
                  "step": {"n": "n - 1"}, "vars": ("n",)},
}


def check_variant(loop: str, variant_src: str) -> dict:
    """Discharge the two termination obligations for ``loop`` against a variant.

    ``variant_src`` is a Z3 integer expression the model writes. The measure must
    be non-negative whenever the loop takes another pass (BOUND) and must strictly
    decrease across a pass (FALL). Both holding forces termination; a failing
    obligation names a state where the measure goes negative or won't drop, which
    is a state the loop could spin in forever.
    """
    import z3
    a, b, result, k, n = z3.Ints("a b result k n")
    ns = dict(z3.__dict__); ns.update(a=a, b=b, result=result, k=k, n=n)
    spec = LOOPS[loop]
    variant = eval(variant_src, ns)                          # noqa: S307 - demo
    step_ns = {**ns, **{v: eval(e, ns) for v, e in spec["step"].items()}}
    variant_next = eval(variant_src, step_ns)
    pre, guard = eval(spec["pre"], ns), eval(spec["guard"], ns)
    track = [ns[v] for v in spec["vars"]]

    def discharge(goal):
        solver = z3.Solver(); solver.add(pre, guard, z3.Not(goal))
        if solver.check() == z3.unsat:
            return {"ok": True, "cex": None}
        m = solver.model()
        return {"ok": False,
                "cex": ", ".join(f"{v}={m.eval(v)}" for v in track)}

    obligations = {"bound": discharge(variant >= 0),
                   "fall": discharge(variant_next < variant)}
    ok = all(o["ok"] for o in obligations.values())
    return {"status": "terminates" if ok else "unknown", "obligations": obligations}


def propose_variant(loop: str, model: str = EIS_MODEL) -> str:
    """Ask the model for a variant (ranking function) for ``loop`` as a Z3 expr.

    Same shape as ``propose_invariant``: the model writes the fluent artifact, Z3
    checks it. A good variant both bounds the measure below and makes it fall; a
    fluent guess often nails the falling and forgets the bound, which is the
    obligation that then refutes it.
    """
    spec = LOOPS[loop]
    prompt = (
        "Here is a Python loop:\n\n" + spec["src"] + "\n\n"
        "Give a variant (ranking function) proving it terminates: an integer "
        "measure over the loop variables that is >= 0 while the loop runs and "
        "strictly decreases on every pass. Write it as a single Z3 integer "
        "expression over the variables (for example, b - k). Output only the "
        "expression, nothing else.")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    # A real variant is an expression over the loop's variables and nothing else,
    # which is how we tell it from a line of English (whose words aren't loop
    # variables). A bare variable like `n` is a valid variant, so we don't require
    # an operator.
    allowed = set(LOOPS[loop]["vars"])
    exprs = [line.strip() for line in re.findall(
        r"^\s*([A-Za-z0-9_ .()*/+\-]+?)\s*$", text, re.M)
        if line.strip() and set(re.findall(r"[A-Za-z_]+", line)) <= allowed]
    return exprs[-1] if exprs else text.strip().splitlines()[-1]


# Both variants captured by scripts/_verify_probe.py (gpt-oss:20b); Z3's verdict
# on the two obligations is deterministic and recomputed at import.
#
# The win. Asked for a variant for times, the model writes b - k: the number of
# passes still to come. It's non-negative while k < b and drops by one each pass,
# so Z3 proves the loop halts for every b >= 0. Partial correctness (the invariant)
# plus termination (the variant) is full correctness.
VARIANT_DEMO = {
    "src": TIMES_SRC,
    "variant": "b - k",
    "verdict": check_variant("times", "b - k"),
}

# The catch. For countdown the model writes the obvious measure, n, and it does
# fall by one each pass. But countdown has no precondition, and n isn't bounded
# below: at n = -1 the guard n != 0 still holds while the measure is already
# negative. Z3 refuses to certify termination and hands back that state, which is
# exactly the input the loop runs forever on. The fix is a precondition, n >= 0,
# the same missing promise that sank the clamp contract.
NONTERM_DEMO = {
    "src": COUNTDOWN_SRC,
    "variant": "n",
    "verdict": check_variant("countdown", "n"),
}


_VARIANT_TEXT = {
    "bound": ("holds: the measure stays >= 0 while the loop runs",
              "fails: at {cex} the measure is already negative"),
    "fall": ("holds: every pass drives the measure strictly down",
             "fails: at {cex} a pass doesn't lower the measure"),
}


def show_variant(demo: dict) -> None:
    """The model writes a variant; Z3 checks the two termination obligations and
    either proves the loop stops for every input or names a state it could hang in."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["src"])
    show_code("gpt-oss", demo["variant"])
    for key in ("bound", "fall"):
        result = demo["verdict"]["obligations"][key]
        held, failed = _VARIANT_TEXT[key]
        line = held if result["ok"] else failed.format(cex=result["cex"])
        show_turn(key.upper(), line)
    if demo["verdict"]["status"] == "terminates":
        show_turn("Z3", "proved: the loop halts for every input the pre allows")
    else:
        show_turn("Z3", "unknown: nothing rules out the loop running forever")


# ── Abstract interpretation: proving safety with no spec at all ────────────────
# The third family on the landscape, run for real this time. It asks the model
# for nothing. Instead it runs the function in a coarser world where a variable
# isn't a value but a *range* of values, and asks whether a bad thing (here, a
# division by zero) can happen in that world. Because a range over-approximates,
# the analysis is sound in one direction: if it says the division is safe, it is.
# The price is imprecision: it can raise a false alarm about a division the real
# program would never actually reach. Both outcomes appear below.

class _Interval:
    """A closed integer range [lo, hi]. Arithmetic follows interval rules: the
    result range is the widest the inputs' ranges allow, which is exactly where
    the analysis loses track of how variables relate to each other."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi

    def __add__(self, o): return _Interval(self.lo + o.lo, self.hi + o.hi)
    def __sub__(self, o): return _Interval(self.lo - o.hi, self.hi - o.lo)

    def __mul__(self, o):
        corners = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return _Interval(min(corners), max(corners))

    def spans_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def __repr__(self) -> str:
        return f"[{self.lo}, {self.hi}]"


def analyze_intervals(src: str, inputs: dict) -> dict:
    """Interpret a straight-line function over integer ranges, flagging any
    division whose divisor range includes zero.

    ``inputs`` maps each parameter to a ``(lo, hi)`` range (the only "spec" this
    family needs, and it's about inputs, not intent). Returns the range computed
    for each variable and, if a divisor range straddles zero, the range that
    tripped the alarm. Sound (it never misses a real division by zero) but not
    exact (the range it computes can be wider than the values that truly occur).
    """
    import ast
    fn = ast.parse(src).body[0]
    env = {name: _Interval(lo, hi) for name, (lo, hi) in inputs.items()}
    trace = [(name, repr(env[name])) for name in inputs]
    alarm = {"divisor": None}

    def ev(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            return _Interval(node.value, node.value)
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.FloorDiv, ast.Div)):
                if alarm["divisor"] is None and right.spans_zero():
                    alarm["divisor"] = repr(right)
                return _Interval(0, 0)
            return {ast.Add: left.__add__, ast.Sub: left.__sub__,
                    ast.Mult: left.__mul__}[type(node.op)](right)
        raise ValueError(ast.dump(node))

    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            value = ev(stmt.value)
            env[stmt.targets[0].id] = value
            trace.append((stmt.targets[0].id, repr(value)))
        elif isinstance(stmt, ast.Return):
            ev(stmt.value)
    return {"safe": alarm["divisor"] is None, "divisor": alarm["divisor"],
            "trace": trace}


# scale divides by x - 5. With x ranging over 0..10 the divisor ranges over
# -5..5, which includes 0, so the alarm is real: x = 5 divides by zero.
SCALE_SRC = ("def scale(x):        # 0 <= x <= 10\n"
             "    d = x - 5\n"
             "    return 100 // d")

# steady divides by x - x + 1, which is always 1, so the code never divides by
# zero. But interval arithmetic evaluates x - x as [-10, 10] (it has forgotten
# the two x's are the same number), so the divisor range comes out [-9, 11] and
# the analyzer cries wolf. This false alarm is the imprecision that soundness costs.
STEADY_SRC = ("def steady(x):       # 0 <= x <= 10\n"
              "    d = x - x + 1\n"
              "    return 100 // d")

INTERVAL_REAL = {"src": SCALE_SRC,
                 "result": analyze_intervals(SCALE_SRC, {"x": (0, 10)})}
INTERVAL_FALSE = {"src": STEADY_SRC,
                  "result": analyze_intervals(STEADY_SRC, {"x": (0, 10)})}


def show_intervals(demo: dict) -> None:
    """The analyzer runs the function over ranges and reports whether a division
    by zero is possible, with no contract from the model at all."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["src"])
    ranges = ", ".join(f"{name} in {rng}" for name, rng in demo["result"]["trace"])
    show_turn("RANGES", ranges)
    if demo["result"]["safe"]:
        show_turn("ANALYZER", "safe: no divisor range includes 0")
    else:
        show_turn("ANALYZER", f"alarm: divisor range {demo['result']['divisor']} "
                              "includes 0, so the division might be by zero")


# ── A weak spec confirms for free ──────────────────────────────────────────────
# The centerpiece caught a spec that was false. This catches one that's true and
# still worthless. Asked for *a* postcondition (not the strongest), a fluent model
# reaches for the obvious safe thing to say. For absolute value that's "the result
# is non-negative" — true, and confirmed by CrossHair over every input. But it's a
# promise so weak a function that SQUARES its input keeps it too, so the identical
# confirmed verdict lands on code that isn't absolute value at all. A verdict is
# only ever as strong as the contract it checked.
ABS_SRC = ("def absval(x: int) -> int:\n"
           "    return x if x >= 0 else -x")
# Not absolute value; still non-negative, so a `>= 0` post waves it through.
ABS_BROKEN = ("def absval(x: int) -> int:\n"
              "    return x * x")

# Captured by scripts/_verify_probe.py (gpt-oss:20b + CrossHair). The model's real
# postcondition, and CrossHair's real verdict on the true function and on the
# broken one. Both confirmed: the spec is true, and it is useless.
HOLLOW_DEMO = {
    "src": ABS_SRC,
    "broken": ABS_BROKEN,
    "post": "post: __return__ >= 0",
    "verdict_ok": {"status": "confirmed", "witness": None, "returns": None},
    "verdict_broken": {"status": "confirmed", "witness": None, "returns": None},
}


def propose_postcondition(impl: str, model: str = EIS_MODEL) -> str:
    """Ask the model for *a* postcondition, not the strongest one.

    The mirror image of ``specify``'s prompt. Where that one pushes for the
    strongest claim (and tempts an over-reach the verifier refutes), this one asks
    for one simple thing the model is confident holds, and tempts an under-reach:
    a claim so weak the verifier confirms it and it still guarantees nothing.
    """
    prompt = (
        "Here is a Python function:\n\n" + impl + "\n\n"
        "Write a postcondition for it in PEP 316 style: a `post:` line stating "
        "something true about the result, where `__return__` names the returned "
        "value. Give one simple property you are confident holds. Output only the "
        "post: line.")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    lines = re.findall(r"^\s*(post\s*:.*)$", text, re.M | re.I)
    return lines[-1].strip() if lines else text.strip()


def show_hollow(demo: dict = HOLLOW_DEMO) -> None:
    """A true-but-weak postcondition: CrossHair confirms it on the real function,
    then confirms the same promise on a broken one the spec was too weak to catch."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["src"])
    show_code("gpt-oss", demo["post"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict_ok"]))
    show_code("broken", demo["broken"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict_broken"]))


# ── The verifier in the loop: does the witness actually help? ───────────────────
# The refuted clamp contract, handed back to the model along with CrossHair's
# counterexample, with one instruction: fix it. Whether the witness is enough to
# close the distance is an empirical question (the Distributed chapter found a
# counterexample that didn't help its model at all), so this is captured, not
# asserted: the model's real revised contract and CrossHair's real re-verdict.
def repair(name: str, contract: str, witness: str, returns: str,
           model: str = EIS_MODEL) -> str:
    """Hand the model its refuted contract plus the counterexample; return its fix.

    Same model, same function, now with a concrete input it got wrong. A single
    counterexample is the most actionable feedback a verifier can give: not "try
    harder" but "here is the case you missed."
    """
    prompt = (
        "Here is a Python function:\n\n" + IMPLS[name] + "\n\n"
        "You wrote this contract for it:\n\n" + contract + "\n\n"
        "A verifier refuted it: " + witness + " returns " + returns + ", an input "
        "that meets your precondition but breaks your postcondition. Fix the "
        "contract so no input can break it, changing as little as possible. Output "
        "only the corrected pre:/post: lines.")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    # gpt-oss rehearses the contract several times in its thinking channel, so we
    # dedupe and keep the last precondition it settled on and its fullest
    # postcondition (the complete case analysis, not a half-written fragment).
    seen = []
    for line in re.findall(r"^\s*((?:pre|post)\s*:.*?)\s*\\?$", text, re.M | re.I):
        line = line.strip().rstrip("\\").strip()
        if line and line not in seen:
            seen.append(line)
    pres = [l for l in seen if l.lower().startswith("pre")]
    posts = [l for l in seen if l.lower().startswith("post")]
    fixed = ([pres[-1]] if pres else []) + ([max(posts, key=len)] if posts else [])
    return "\n".join(fixed) if fixed else text.strip()


# Captured by scripts/_verify_probe.py (gpt-oss:20b + CrossHair). Handed the
# refuted clamp contract and the witness clamp(0, -1, -2), the model commits to
# the precondition it hedged on the first time, lo <= hi, and CrossHair confirms
# the revised contract over every input. Here the single counterexample was
# enough to close the distance; it isn't always (see the Distributed chapter, where an
# analogous counterexample left its model no better off). Post folded for the page.
REPAIR_DEMO = {
    "name": "clamp",
    "before": SPEC_DEMO["contract"],
    "witness": "clamp(0, -1, -2)",
    "returns": "-1",
    "after": ("pre: lo <= hi\n"
              "post: ((x <= lo) and (__return__ == lo))\n"
              "  or  ((hi <= x) and (__return__ == hi))\n"
              "  or  (lo < x < hi and __return__ == x)"),
    "verdict": {"status": "confirmed", "witness": None, "returns": None},
}


def show_repair(demo: dict = REPAIR_DEMO) -> None:
    """The loop closing: the refuted contract, the witness, the model's revised
    contract, and the verifier's verdict on the second try."""
    from genai.agent import show_code, show_turn
    show_code("function", IMPLS[demo["name"]])
    show_code("gpt-oss", demo["before"])
    show_turn("CROSSHAIR", f"refuted: {demo['witness']} returns {demo['returns']}, "
                           "which breaks the post-condition")
    show_code("gpt-oss", demo["after"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict"]))


# Captured by scripts/_verify_probe.py (gpt-oss:20b + CrossHair). The model states
# the right relationship between the result and the input but never guards the
# domain, so it says nothing about the empty list. CrossHair tries [] and the
# function divides by zero before any postcondition is even evaluated.
AVERAGE_DEMO = {
    "impl": IMPLS["average"],
    "contract": "post: __return__ * len(nums) == sum(nums)",
    "verdict": {"status": "refuted", "witness": "average([])", "returns": None,
                "message": "ZeroDivisionError:  when calling average([])"},
}


def show_average(demo: dict = AVERAGE_DEMO) -> None:
    """A contract that reads fine until the verifier tries the one list a test
    never would: the empty one."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["impl"])
    show_code("gpt-oss", demo["contract"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict"]))


# ── The contract that says too little ─────────────────────────────────────────
# "A Proof of Nothing" left absval with a postcondition CrossHair confirmed on the
# real function and on an `x*x` impostor alike: `__return__ >= 0` is true but too
# weak to separate them. Confirmed and useful are different axes. Strengthen the
# spec to say what absval actually promises, the result is x's magnitude, non-
# negative and equal to x or -x, and CrossHair still confirms it on the real
# function but now refutes the impostor with a witness. Same verifier, same code;
# only the spec's strength changed. CrossHair's verdicts are deterministic;
# captured once by scripts/_verify_strength_probe.py.

STRENGTH_WEAK = "__return__ >= 0"
STRENGTH_STRONG = "__return__ >= 0 and (__return__ == x or __return__ == -x)"

STRENGTH_DEMO = {
    "weak": STRENGTH_WEAK,
    "strong": STRENGTH_STRONG,
    "grid": {"real absval": {"weak": "confirmed", "strong": "confirmed"},
             "x*x impostor": {"weak": "confirmed", "strong": "REFUTED"}},
    "witness": "absval(2) returns 4, not 2",
}


def show_contract_strength(demo: dict = None) -> None:
    """The strength axis as a grid: a weak spec confirms the real function and the
    impostor both; the strong spec confirms the real one and refutes the impostor.
    Confirmed isn't the same as useful."""
    from genai.agent import show_turn
    demo = demo or STRENGTH_DEMO
    show_turn("weak spec", f"post: {demo['weak']}")
    show_turn("strong spec", f"post: {demo['strong']}")
    rows = demo["grid"]
    print()
    print(f"{'':15}{'weak':>11}{'strong':>11}")
    for name, verdicts in rows.items():
        print(f"{name:15}{verdicts['weak']:>11}{verdicts['strong']:>11}")
    show_turn("WITNESS", f"the strong spec's counterexample: {demo['witness']}; the "
              "weak spec waved it through")


# ── The verdict between yes and no: an over-tight precondition ─────────────────
# The precondition is a dial. Section 3 left clamp's off and the verifier refuted
# it (nothing ruled out lo > hi); section 4 set it to lo <= hi and the verifier
# confirmed. Crank it to a precondition no input can satisfy and CrossHair gives
# its third verdict, `unknown`: it goes looking for an input that meets the pre,
# finds none, and can neither confirm a path nor return a counterexample. The
# postcondition is the same true bound in both; only the precondition moves.
# CrossHair's verdicts are deterministic here (both return at once); baked for
# byte-stability by scripts/_verify_unknown_probe.py.

VACUOUS_DEMO = {
    "impl": IMPLS["clamp"],
    "tight": ("pre:  lo <= hi and hi < lo   # no lo, hi can be both\n"
              "post: lo <= __return__ <= hi"),
    "right": ("pre:  lo <= hi\n"
              "post: lo <= __return__ <= hi"),
    "verdict_tight": {"status": "unknown", "witness": None, "returns": None,
                      "message": "No input satisfies the precondition."},
    "verdict_right": {"status": "confirmed", "witness": None, "returns": None,
                      "message": "Confirmed over all paths."},
}


def show_vacuous(demo: dict = VACUOUS_DEMO) -> None:
    """The precondition cranked past what any input can meet. Same function, same
    true postcondition; the over-tight precondition draws `unknown` (nothing to
    check), the satisfiable one draws `confirmed`."""
    from genai.agent import show_code, show_turn
    show_code("function", demo["impl"])
    show_code("too tight", demo["tight"])
    if demo["verdict_tight"]["status"] == "unknown":
        show_turn("CROSSHAIR", "unknown: no input satisfies the precondition, so "
                               "nothing is left to confirm or refute")
    else:
        show_turn("CROSSHAIR", _verdict_line(demo["verdict_tight"]))
    show_code("just right", demo["right"])
    show_turn("CROSSHAIR", _verdict_line(demo["verdict_right"]))

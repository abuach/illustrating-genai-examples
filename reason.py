"""Automated reasoning as a tool the model reaches for.

A generative model is fluent but unreliable at exact logic: it will state a
confident, wrong answer to a constraint puzzle the way it states a true one.
The neuro-symbolic fix is to split the work. The model does what it is good at,
*translating* a natural-language puzzle into formal constraints, and a symbolic
solver (Z3, an SMT solver) does what it is good at, deducing the answer those
constraints force. The solver is a tool, reached for the way the model reached
for a search tool over MCP, and it is there to stop the model from hallucinating
the part it can't be trusted with.

``solve_with_z3`` runs the loop: the model writes a Z3 program, we run it, and if
it doesn't run, the error goes back to the model to fix. The model is gpt-oss:20b;
runs are nondeterministic, so capture REASONING_STUDY once and freeze the cells.
"""
import re

import ollama

from genai.arch import EIS_MODEL


def run_z3(program: str):
    """Run a model-written Z3 program and return its `answer` variable.

    The z3 API is pre-imported into the program's namespace, so the model can
    write Solver(), Int(), Bool() directly. The program is expected to assign the
    result to a variable named ``answer``.
    """
    import z3
    namespace = dict(z3.__dict__)
    exec(program, namespace)            # noqa: S102 - controlled local demo
    return namespace.get("answer")


def _emit_code(model: str, prompt: str) -> str:
    """Ask the model for a Z3 program; strip any markdown fence around it."""
    msg = ollama.chat(model=model, think=True, options={"num_predict": 1200},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = msg.get("content") or msg.get("thinking") or ""
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (fenced.group(1) if fenced else text).strip()


def answer_directly(problem: str, model: str = EIS_MODEL) -> str:
    """The model alone: reason in its head, then state a final answer line.

    gpt-oss often leaves `content` empty and puts everything in its thinking
    channel, so we ask for an explicit ``ANSWER:`` line and scan both channels
    for it, falling back to the last line of whatever it produced.
    """
    msg = ollama.chat(model=model, think=True, options={"num_predict": 2000},
                      messages=[{"role": "user", "content": problem +
                                 " Reason it through, then end with a line "
                                 "'ANSWER: <your final answer>'."}])["message"]
    text = f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"
    hits = re.findall(r"ANSWER:\s*(.+)", text)
    if hits:
        return hits[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def solve_with_z3(problem: str, model: str = EIS_MODEL, max_tries: int = 3) -> tuple:
    """Neuro-symbolic: the model formalizes, Z3 deduces, errors are fed back."""
    instruction = (
        "Translate this puzzle into a Z3 program. The z3 Python API is already "
        "imported (use Solver, Int, Bool, Bools, Not, Or, And, sat, etc.). Solve "
        "it, then assign the final answer to a variable named `answer` as a short "
        "string. Output only the Python code.\nPuzzle: " + problem)
    feedback, code = "", ""
    for attempt in range(1, max_tries + 1):
        code = _emit_code(model, instruction + feedback)
        try:
            result = run_z3(code)
            if result is not None:
                return str(result).strip(), code, attempt
            feedback = "\n\nYour program ran but never set `answer`. Fix it."
        except Exception as exc:
            feedback = f"\n\nYour program raised: {exc!r}. Fix it."
    return None, code, max_tries


# A single forced answer each; gold checked leniently (substring). Three cryptarithms
# (where even a reasoning model hallucinates) and two short logic puzzles (where it
# doesn't), so the study shows the solver earning its keep on the hard ones.
PROBLEMS = [
    ("In SEND + MORE = MONEY, each letter is a different digit 0-9 and no number "
     "starts with zero. What number does MONEY spell?", "10652"),
    ("In CROSS + ROADS = DANGER, each letter is a different digit 0-9 and no number "
     "starts with zero. What number does DANGER spell?", "158746"),
    ("In BASE + BALL = GAMES, each letter is a different digit 0-9 and no number "
     "starts with zero. What number does GAMES spell?", "14938"),
    ("Three boxes, one holds gold. The red box says 'the gold is here.' The blue box "
     "says 'the gold is not here.' The green box says 'the gold is not in the red box.' "
     "Exactly one of the three statements is true. Which box has the gold?", "blue"),
    ("Anna, Bill, and Cara each always tell the truth or always lie. Anna says 'Bill "
     "lies.' Bill says 'Cara lies.' Cara says 'Anna and Bill both lie.' Who tells the "
     "truth?", "bill"),
]


def _hit(answer, gold) -> bool:
    return answer is not None and gold.lower() in str(answer).lower()


def reasoning_study(model: str = EIS_MODEL, trials: int = 3) -> dict:
    """Accuracy of the model alone versus the model handing logic to Z3."""
    out = {"model_alone": 0, "with_solver": 0, "n": 0}
    for problem, gold in PROBLEMS:
        for _ in range(trials):
            out["n"] += 1
            out["model_alone"] += int(_hit(answer_directly(problem, model), gold))
            answer, _, _ = solve_with_z3(problem, model)
            out["with_solver"] += int(_hit(answer, gold))
    n = out["n"]
    return {"model_alone": round(out["model_alone"] / n, 3),
            "with_solver": round(out["with_solver"] / n, 3), "n": n}


# Captured by scripts/_reason_z3_probe.py (gpt-oss:20b + Z3). The model's REAL Z3
# program, kept verbatim. The model translates a logic puzzle into constraints and
# Z3 returns the one assignment that satisfies them all — a proven answer, not a guess.
SOLVER_DEMO = {
    "problem": "Anna, Bill, and Cara each always tell the truth or always lie. "
               "Anna says 'Bill lies.' Bill says 'Cara lies.' Cara says 'Anna and "
               "Bill both lie.' Who tells the truth?",
    "code": (
        "from z3 import Solver, Bool, Not, And\n"
        "s = Solver()\n"
        "A = Bool('Anna')\n"
        "B = Bool('Bill')\n"
        "C = Bool('Cara')\n"
        "s.add(A == Not(B))                 # Anna says \"Bill lies.\"\n"
        "s.add(B == Not(C))                 # Bill says \"Cara lies.\"\n"
        "s.add(C == And(Not(A), Not(B)))    # Cara says \"Anna and Bill both lie.\"\n"
        "s.check()\n"
        "m = s.model()\n"
        "answer = 'Bill' if m[B] else ('Anna' if m[A] else 'Cara')"),
    "answer": "Bill",
}

# What the per-puzzle runs show, qualitatively (the numbers are too run-to-run noisy
# to chart honestly): on short logic puzzles the model both reasons and translates
# fine; on cryptarithms it reasons wrong AND often can't write working Z3 either —
# the bottleneck moves from reasoning to translation. Discussed, not plotted.


def show_solver_demo(demo=SOLVER_DEMO):
    """The neuro-symbolic round trip: the model writes Z3, the solver proves it."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["problem"])
    show_code("gpt-oss", demo["code"])
    show_turn("Z3", f"answer = {demo['answer']!r}   (the one assignment that "
                    "satisfies all three statements)")


# Real knights-puzzle Z3, one shot per model, captured by _reason_bakeoff_probe.py.
# A spectrum of translation quality: correct, right-logic-wrong-code, invalid syntax.
TRANSLATIONS = [
    ("qwen2.5-coder", "runs, answers Bill",
     "solver.add(Implies(anna_truth, Not(bill_truth)))\n"
     "solver.add(Implies(Not(anna_truth), bill_truth))   # correct: anna <=> not bill"),
    ("gemma4", "right logic, then TypeError",
     "constraint_anna = (anna_truth == Not(bill_truth))   # the constraints are correct...\n"
     "if A_truth():                                       # ...but A_truth is a value, not a call"),
    ("mistral", "SyntaxError",
     "constraints.append(Not(bill) => cara)               # '=>' is not Python; Z3 uses Implies"),
]


def show_translations(rows=TRANSLATIONS):
    """A few models' Z3 for the same puzzle, from working to broken."""
    from genai.agent import show_code, show_turn
    for model, outcome, code in rows:
        show_turn(model, f"-> {outcome}")
        show_code("", code)


# ── When there's no answer: the verdict a generator can't give ────────────────
# A paradox knights puzzle. Anna<=>Bill and Bill<=>not-Anna force Anna<=>not-Anna,
# so no assignment satisfies all three. A language model still names someone,
# because "who tells the truth" demands a name; the solver returns `unsat`, a
# proof of impossibility, and its unsat core points at the exact clashing pair.
UNSAT_PUZZLE = ("Anna, Bill, and Cara each always tell the truth or always lie. "
                "Anna says 'Bill tells the truth.' Bill says 'Anna lies.' Cara "
                "says 'Bill tells the truth.' Who tells the truth?")

UNSAT_PROGRAM = (
    "s = Solver()\n"
    "A, B, C = Bools('Anna Bill Cara')\n"
    "s.assert_and_track(A == B,      'Anna')   # Anna: \"Bill tells the truth\"\n"
    "s.assert_and_track(B == Not(A), 'Bill')   # Bill: \"Anna lies\"\n"
    "s.assert_and_track(C == B,      'Cara')   # Cara: \"Bill tells the truth\"\n"
    "status = s.check()                         # what does the solver find?\n"
    "core = sorted(str(x) for x in s.unsat_core())")


def unsat_verdict(program: str = UNSAT_PROGRAM) -> tuple:
    """Run the paradox encoding and return Z3's (status, unsat_core).

    The result is deterministic: `unsat`, with the core naming the two statements
    that can't both hold (Cara's is left out, because it isn't part of the clash).
    """
    import z3
    namespace = dict(z3.__dict__)
    exec(program, namespace)            # noqa: S102 - controlled local demo
    return namespace["status"], namespace["core"]


# ── The constraint it forgot to write down: sat, but under-constrained ────────
# The knights puzzle from SOLVER_DEMO has three statements and exactly one
# solution. Drop any single statement from the translation and the program still
# runs and still returns sat, but now more than one assignment satisfies what's
# left, and s.model() hands back whichever the solver happened to find. A misread
# constraint (the And/Or trap below) proves a different but unique answer; a
# *dropped* constraint proves a whole family of them and reports one as if it were
# the answer. Nothing errors, so the omission is invisible unless you count the
# models. Fully deterministic.

KNIGHTS = [
    ("A == Not(B)",              'Anna says "Bill lies."'),
    ("B == Not(C)",              'Bill says "Cara lies."'),
    ("C == And(Not(A), Not(B))", 'Cara says "Anna and Bill both lie."'),
]


def _knights_models(exprs) -> list:
    """Every truth-teller set consistent with the given constraint expressions,
    enumerated by blocking each solution the solver finds and re-checking until no
    new one remains (all-SAT). Each element is the tuple of people who tell the
    truth in one satisfying assignment."""
    import z3
    A, B, C = z3.Bools("Anna Bill Cara")
    env = {"A": A, "B": B, "C": C, "Not": z3.Not, "And": z3.And, "Or": z3.Or}
    s = z3.Solver()
    for expr in exprs:
        s.add(eval(expr, env))          # noqa: S307 - the fixed KNIGHTS expressions
    out = []
    while len(out) < 8 and s.check() == z3.sat:   # at most 2**3 assignments
        m = s.model()
        # model_completion fills in any don't-care variable, so each blocking
        # clause excludes one full assignment and the enumeration terminates.
        vals = {v: z3.is_true(m.eval(v, model_completion=True)) for v in (A, B, C)}
        out.append(tuple(name for name, v in (("Anna", A), ("Bill", B), ("Cara", C))
                         if vals[v]))
        s.add(z3.Or([v != z3.BoolVal(vals[v]) for v in (A, B, C)]))
    return out


def count_knights_models(drop: int = None) -> tuple:
    """Return ``(status, models)`` for the knights translation, optionally with the
    statement at index ``drop`` left out. ``drop=None`` is the faithful encoding."""
    exprs = [e for i, (e, _n) in enumerate(KNIGHTS) if i != drop]
    models = _knights_models(exprs)
    return ("sat" if models else "unsat"), models


def _tellers(models) -> str:
    """Format a list of truth-teller tuples: ``{Bill}`` or ``{Bill} or {Anna, Cara}``."""
    return " or ".join("{" + ", ".join(m) + "}" if m else "{no one}" for m in models)


def show_underconstrained(drop: int = 2) -> None:
    """Two translations of the same puzzle: the faithful one Z3 pins to a single
    truth-teller, and the one missing a statement, which Z3 still calls sat while
    more than one answer fits. The solver reports one; only counting the models
    shows the puzzle was never fully pinned down. ``drop`` is the omitted line."""
    from genai.agent import show_code, show_turn
    full_status, full_models = count_knights_models()
    drop_status, drop_models = count_knights_models(drop=drop)
    faithful = "\n".join(f"s.add({e})".ljust(34) + f"# {n}" for e, n in KNIGHTS)
    dropped = "\n".join(
        (f"s.add({e})".ljust(34) + f"# {n}") if i != drop
        else " " * 34 + f"# {n.split(' says')[0]}'s line: never written down"
        for i, (e, n) in enumerate(KNIGHTS))
    show_turn("you", SOLVER_DEMO["problem"])
    show_code("faithful", faithful)
    show_turn("Z3", f"{full_status}, {len(full_models)} model -> "
                    f"{_tellers(full_models)}   (a proof)")
    show_code("dropped", dropped)
    show_turn("Z3", f"{drop_status}, {len(drop_models)} models -> "
                    f"{_tellers(drop_models)}   (so which one is 'the answer'?)")


# ── The faithfulness trap: a valid proof of the wrong problem ─────────────────
# The knights puzzle from SOLVER_DEMO turns on Cara's line, "Anna and Bill both
# lie." Read "both" faithfully and it's And(not A, not B); misread it as "at least
# one lies" and it's Or(...). One operator, and Z3 faithfully, provably solves a
# different puzzle: the And reading proves Bill, the Or reading proves Anna, Cara.
def _knights_truth_tellers(cara_reading) -> str:
    """Solve the Anna/Bill/Cara puzzle under one reading of Cara's statement.

    Anna says "Bill lies", Bill says "Cara lies"; those two are fixed. The third
    constraint, Cara's "Anna and Bill both lie", is built by ``cara_reading``,
    either ``z3.And`` (the faithful "both") or ``z3.Or`` (the faithless "either").
    Each reading has a single solution, so the answer is reproducible.
    """
    import z3
    A, B, C = z3.Bools("Anna Bill Cara")
    s = z3.Solver()
    s.add(A == z3.Not(B), B == z3.Not(C), C == cara_reading(z3.Not(A), z3.Not(B)))
    s.check()
    model = s.model()
    return ", ".join(p for p, v in (("Anna", A), ("Bill", B), ("Cara", C))
                     if model[v])


def faithful_answer() -> str:
    """Cara's "both lie" read as And: the puzzle's true answer."""
    import z3
    return _knights_truth_tellers(z3.And)


def faithless_answer() -> str:
    """Cara's "both lie" misread as Or: a proven answer to a different puzzle."""
    import z3
    return _knights_truth_tellers(z3.Or)


def show_faithless_demo():
    """One operator apart: the same solver proves two different answers."""
    from genai.agent import show_code, show_turn
    show_turn("you", 'Cara says "Anna and Bill both lie." Read "both" two ways:')
    show_code("And", "s.add(C == And(Not(A), Not(B)))   # both of them lie")
    show_turn("Z3", f"proves -> {faithful_answer()}")
    show_code("Or", "s.add(C == Or(Not(A), Not(B)))    # at least one lies")
    show_turn("Z3", f"proves -> {faithless_answer()}")


# ── When there's no answer: the model names someone, the solver proves none ───
# A small, non-reasoning model's real pick on UNSAT_PUZZLE. Asked who tells the
# truth, it names someone, because a generator has no way to answer "no one, the
# puzzle contradicts itself." Different models name different suspects, and big
# models invent a consistent story; none can return the core. Captured by
# scripts/_reason_unsat_probe.py.
UNSAT_PICK = ("llama3.1", "Cara tells the truth.")


def show_unsat_demo(pick=UNSAT_PICK, program=UNSAT_PROGRAM):
    """The model names a truth-teller; the solver proves there isn't one."""
    from genai.agent import show_code, show_turn
    model, answer = pick
    show_turn("you", "Who tells the truth?")
    show_turn(model, answer)
    show_code("", program)
    status, core = unsat_verdict(program)
    show_turn("Z3", f"status = {status}   (no assignment satisfies all three)")
    show_turn("", f"unsat core = {core}: those two clash, Cara's line is innocent")


# ── Cross-checking the translation: agreement is the corroboration ────────────
# Final answers from five independent gpt-oss formalizations of the solvable
# knights puzzle. Four produce a working program and agree on Bill; one fails to
# produce one at all (None). The faithless Or reading would be the lone
# dissenter. Captured by scripts/_reason_extras_probe.py.
CROSSCHECK_VOTES = ["Bill", None, "Bill", "Bill", "Bill"]


def show_crosscheck(votes=CROSSCHECK_VOTES):
    """Independent formalizations agree; the planted misread is the dissenter."""
    from genai.agent import show_turn
    shown = [v if v else "(no program)" for v in votes]
    answers = [v for v in votes if v]
    consensus = max(set(answers), key=answers.count) if answers else "(none)"
    show_turn("you", "Formalize the same puzzle five times, independently.")
    show_turn("gpt-oss", " · ".join(shown))
    show_turn("", f"every working formalization agrees -> {consensus}")
    show_turn("you", 'Now add the "both means either" misread from the trap:')
    show_turn("misread", f"{faithless_answer()}   <- a valid proof that disagrees")


# ── Proving two programs equivalent ───────────────────────────────────────────
# Differential testing, the oracle the Refactoring chapter builds, replays a
# battery of sample inputs through old and new code and compares answers: it
# samples the input space. A solver quantifies over it instead. Assert "some
# input makes the two disagree" and `unsat` is a proof of equivalence over
# every input at once, while `sat` hands back the exact witness input. Money is
# integer cents so the encoding is honest: Z3's Int is the mathematician's
# integer, like Python's int, and Python's // agrees with Z3's integer division
# on non-negative values; Python floats are not Z3's Reals.

FEE_PAIR = (
    "def fee_old(cents):  # 20% fee, rounded down\n"
    "    return cents * 20 // 100\n"
    "\n"
    "def fee_new(cents):  # the rewrite: a fifth\n"
    "    return cents // 5")

PROMO_PAIR = (
    "def promo_old(cents):  # round discount to nearest\n"
    "    return cents - (cents * 15 + 50) // 100\n"
    "\n"
    "def promo_new(cents):  # rewrite: rounding dropped\n"
    "    return cents - cents * 15 // 100")


# qwen2.5-coder's REAL first attempt at the fee check, captured verbatim by
# scripts/_reason_equiv_probe.py (bare mode, temperature 0). The constraint
# logic is exactly right; the code reaches for Python's // on Z3 Ints, which
# z3py doesn't overload. All three attempts crashed the same way, with the
# TypeError fed back each time: the error names the operator but not the fix.
EQUIV_STUMBLE = (
    "qwen2.5-coder",
    "s = Solver()\n"
    "cents = Int('cents')\n"
    "fee_old_expr = cents * 20 // 100\n"
    "fee_new_expr = cents // 5\n"
    "s.add(fee_old_expr != fee_new_expr)",
    "TypeError: unsupported operand type(s) for //: "
    "'ArithRef' and 'int'   (all 3 attempts)",
)

# The REAL working translations, captured by scripts/_reason_equiv_probe.py
# (hinted mode: the prompt gains one line of API documentation, integer
# division on Z3 Ints is / not //). Both are qwen2.5-coder's first attempt,
# verbatim. Note what the model *skipped*: the prompt asked for cents >= 0 and
# neither program constrains it. Here that changes neither verdict (the fee
# identity holds over all integers, and the promo witness is non-negative),
# but it is one more quiet drop a reader of the translation has to catch.
EQUIV_SAME = {
    "ask": "A refactor rewrote the fee function. Is there any price "
           "at all where old and new disagree? Translate the pair "
           "into a Z3 disagreement check. (Docs: on Z3 Ints, integer "
           "division is written /, not //.)",
    "pair": FEE_PAIR,
    "model": "qwen2.5-coder",
    "z3": (
        "from z3 import Solver, Int, sat, unsat\n"
        "\n"
        "s = Solver()\n"
        "cents = Int('cents')\n"
        "fee_old_expr = cents * 20 / 100\n"
        "fee_new_expr = cents / 5\n"
        "\n"
        "s.add(fee_old_expr != fee_new_expr)\n"
        "status = s.check()\n"
        "\n"
        "if status == sat:\n"
        "    witness = s.model()[cents]\n"
        "    print(f\"Input that makes the two functions disagree: "
        "{witness}\")\n"
        "else:\n"
        "    print(\"No input makes the two functions disagree.\")"),
}

EQUIV_DIFF = {
    "ask": "Same review, next function: the promo rewrite. Any price "
           "where old and new disagree?",
    "pair": PROMO_PAIR,
    "model": "qwen2.5-coder",
    "z3": (
        "from z3 import Solver, Int, sat, unsat\n"
        "\n"
        "s = Solver()\n"
        "cents = Int('cents')\n"
        "promo_old_expr = cents - (cents * 15 + 50) / 100\n"
        "promo_new_expr = cents - cents * 15 / 100\n"
        "\n"
        "s.add(promo_old_expr != promo_new_expr)\n"
        "status = s.check()\n"
        "\n"
        "if status == sat:\n"
        "    witness = s.model()[cents]\n"
        "    print(f\"Input that makes the two functions disagree: "
        "{witness}\")\n"
        "else:\n"
        "    print(\"No input makes the two functions disagree.\")"),
    # The full program repeats the fee harness line for line; the demo shows
    # the two lines that changed (verbatim) and runs the whole thing.
    "shown": (
        "promo_old_expr = cents - (cents * 15 + 50) / 100\n"
        "promo_new_expr = cents - cents * 15 / 100"),
}


def run_equiv(program: str) -> tuple:
    """Run a model-written disagreement check; return (status, witness).

    The program follows the probe's harness: it sets ``status`` from
    ``s.check()`` and, when sat, ``witness`` to the model value of ``cents``.
    Its own print() lines are swallowed so the transcript stays labelled.
    Deterministic for a fixed program and Z3 version; nothing here samples.
    """
    import io
    from contextlib import redirect_stdout

    import z3
    namespace = dict(z3.__dict__)
    with redirect_stdout(io.StringIO()):
        exec(program, namespace)        # noqa: S102 - controlled local demo
    status = str(namespace["status"])
    witness = int(str(namespace["witness"])) if status == "sat" else None
    return status, witness


def show_equiv_stumble(row=EQUIV_STUMBLE) -> None:
    """The bare translation attempt: right constraints, one wrong operator."""
    from genai.agent import show_code, show_turn
    model, lines, error = row
    show_turn("you", "Is there any price where old and new disagree? "
                     "Translate the pair into a Z3 disagreement check.")
    show_code(model, lines)
    show_turn("ERROR", error)


def show_equiv_demo(demo) -> None:
    """One equivalence check: the pair, the model's Z3, the solver's verdict."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["ask"])
    show_code("", demo["pair"])
    show_code(demo["model"], demo.get("shown", demo["z3"]))
    status, witness = run_equiv(demo["z3"])
    if status == "unsat":
        show_turn("Z3", "status = unsat   (no cents value where they "
                        "disagree exists)")
    else:
        show_turn("Z3", f"status = sat   ->  cents = {witness}")
        namespace: dict = {}
        exec(demo["pair"], namespace)   # noqa: S102 - the pair shown above
        old, new = [v for v in namespace.values() if callable(v)]
        show_code("", f"  {old.__name__}({witness}) = {old(witness)}   "
                      f"{new.__name__}({witness}) = {new(witness)}")


# ── An over-constrained schedule and its unsat core ───────────────────────────
# One meeting, three possible days, three people's requirements that cannot all
# hold. Z3 returns unsat, and because each requirement is added with
# assert_and_track under a readable name, s.unsat_core() names the minimal
# clashing subset. The labels are full sentences on purpose: the core comes
# back already legible, and Kim's requirement, which plays no part in the
# clash, is provably absent from it.
SCHEDULE_ASK = ("One weekly meeting: Monday, Wednesday, or Friday. Ana is "
                "only free on Friday. Ben teaches on Friday. Kim can't make "
                "Wednesdays. Find a day that works for all three.")

SCHEDULE_PROGRAM = (
    'day = Int("day")        # 1 = Mon, 2 = Wed, 3 = Fri\n'
    "s = Solver()\n"
    "s.add(Or(day == 1, day == 2, day == 3))\n"
    "s.assert_and_track(And(day != 1, day != 2),\n"
    '                   "Ana is only free on Friday")\n'
    's.assert_and_track(day != 3, "Ben teaches on Friday")\n'
    "s.assert_and_track(day != 2,\n"
    '                   "Kim can\'t make Wednesdays")\n'
    "status = s.check()\n"
    "core = sorted(str(c) for c in s.unsat_core())")


def schedule_verdict(program: str = SCHEDULE_PROGRAM) -> tuple:
    """Run the schedule encoding and return Z3's (status, unsat core).

    Deterministic: `unsat`, with the core naming the two requirements that
    clash (Ana's and Ben's). Kim's is left out because it isn't part of the
    contradiction; the core is the proof's actual footprint.
    """
    import z3
    namespace = dict(z3.__dict__)
    exec(program, namespace)            # noqa: S102 - controlled local demo
    return str(namespace["status"]), namespace["core"]


def show_schedule_demo(program: str = SCHEDULE_PROGRAM) -> None:
    """The solver says no schedule exists, and names the clashing pair."""
    from genai.agent import show_code, show_turn
    show_turn("you", SCHEDULE_ASK)
    show_code("", program)
    status, core = schedule_verdict(program)
    show_turn("Z3", f"status = {status}   (no day satisfies every "
                    "requirement)")
    show_turn("CORE", " + ".join(f"'{c}'" for c in core))


# gemma4's REAL one-sentence reading of the core, captured verbatim by
# scripts/_reason_core_probe.py (temperature 0; identical across seeds 0-2).
# The conclusion is right and both core facts appear, but the first clause
# hands Ana's only-free-on-Friday constraint to Ben as well, and "teaches on
# Friday" grows into "teaching all day". The guarantee stays in the core; the
# sentence is its readable, slightly smeared rendering, and the smear is the
# demo's honest lesson, not a defect to edit out.
SCHEDULE_EXPLANATION = (
    "gemma4",
    "Because both Ana and Ben are unavailable for a meeting on any day "
    "other than Friday, and Ben is teaching all day on Friday, there is "
    "no time that works for everyone.",
)


def show_core_explained(pick=SCHEDULE_EXPLANATION) -> None:
    """The model reads the proof: the unsat core, rendered for the team."""
    from genai.agent import show_turn
    model, sentence = pick
    _, core = schedule_verdict()
    show_turn("you", "The solver says unsat. Its core: "
                     + " + ".join(f"'{c}'" for c in core)
                     + ". In one plain-English sentence, why is there no "
                       "possible meeting day?")
    show_turn(model, sentence)


# ── When the solver says wait: the fourth verdict ─────────────────────────────
# sat and unsat aren't the only answers Z3 can give. On nonlinear integer
# arithmetic, which is undecidable in general, it can also return `unknown`: an
# honest "I can't tell", not a guess. Which verdict you get depends on how the
# puzzle was phrased. The Hardy-Ramanujan question, find a number that is the sum
# of two positive cubes in two different ways, is nonlinear (it cubes unknowns).
# Written open, Z3 finds one. Written with a "keep it reasonable" bound, the exact
# same question, and one whose bound even contains the answer, Z3 gives up. The
# solver's guarantee is real but not total, and the model translating the puzzle
# decides whether the solver can honor it. The verdicts are deterministic (nlsat
# gives up in about a second, well short of any timeout); the specific witness Z3
# returns for the open form varies run to run, so it is captured once by
# scripts/_reason_wait_probe.py.

WAIT_PUZZLE = ("Find a whole number that is the sum of two positive cubes in two "
               "different ways (the Hardy-Ramanujan taxicab question).")

WAIT_OPEN_CODE = '''a, b, c, d = Ints("a b c d")
add(a**3 + b**3 == c**3 + d**3)   # two ways
add(a > 0, b > 0, c > 0, d > 0)
add(a < b, c < d, a < c)   # two pairs
answer = check()'''

WAIT_BOUNDED_CODE = '''a, b, c, d = Ints("a b c d")
add(a**3 + b**3 == c**3 + d**3)   # two ways
add(a > 0, b > 0, c > 0, d > 0)
add(a < b, c < d, a < c)   # two pairs
add(a < 40, b < 40, c < 40, d < 40)   # keep small
answer = check()'''


def solve_taxicab(bounded: bool, timeout_ms: int = 10000) -> tuple:
    """Run the taxicab question through Z3 one of two faithful ways. Returns
    ``(verdict, number)`` where verdict is "sat"/"unsat"/"unknown" and number is
    the taxicab value when sat, else None. ``bounded`` adds the <= 40 limits,
    which (surprisingly) is what makes the solver give up."""
    import z3
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    a, b, c, d = z3.Ints("a b c d")
    s.add(a * a * a + b * b * b == c * c * c + d * d * d)
    s.add(a > 0, b > 0, c > 0, d > 0, a < b, c < d, a < c)
    if bounded:
        s.add(a <= 40, b <= 40, c <= 40, d <= 40)
    verdict = s.check()
    if verdict == z3.sat:
        m = s.model()
        w = tuple(m[v].as_long() for v in (a, b, c, d))
        return "sat", w[0] ** 3 + w[1] ** 3, w
    return str(verdict), None, None


# Captured by scripts/_reason_wait_probe.py. The verdicts are stable; the witness
# number for the open form is one Z3 returned (the famous 1729 = 1^3+12^3 =
# 9^3+10^3).
WAIT_DEMO = {"bounded_verdict": "unknown", "open_verdict": "sat",
             "number": 1729, "witness": [1, 12, 9, 10]}


def show_solver_wait(demo: dict = None) -> None:
    """One puzzle, two faithful Z3 translations, two different verdicts: the open
    form is solved, the "reasonable"-looking bounded form the solver can't
    decide."""
    from genai.agent import show_code, show_turn
    demo = demo or WAIT_DEMO
    a, b, c, d = demo["witness"]
    show_turn("you", WAIT_PUZZLE)
    show_code("bounded", WAIT_BOUNDED_CODE)
    show_turn("Z3", f"answer = {demo['bounded_verdict']}   (the solver can't decide "
              "this one)")
    show_code("open", WAIT_OPEN_CODE)
    show_turn("Z3", f"answer = {demo['open_verdict']}   ->  {demo['number']} = "
              f"{a}^3+{b}^3 = {c}^3+{d}^3")

"""Model checking: explore EVERY interleaving, not the few a test happens to hit.

The Autonomy chapter fixed one race on one machine and watched a lock take the
corruption rate to zero over sixty runs. Sixty runs is sixty interleavings, and a
fleet of agents can produce far more; the one you never sampled is the one that
pages you at three in the morning. When the coordination genuinely matters,
engineers stop sampling runs and start checking the protocol itself.

That's a verification family this book hasn't used yet. The Verification chapter
reasoned about *one sequential function*: CrossHair searched its paths, Z3 proved
its loop. A protocol is many processes whose atomic steps interleave, and the bug
hides in a particular interleaving, not in any one process read top to bottom.
*Model checking* is the tool: write the protocol as a small state machine plus a
safety invariant, and a checker explores the whole reachable state space across
every interleaving, returning a concrete counterexample trace the moment it finds
a state the invariant forbids. TLA+ and Spin are the industrial versions; the
oracle here is a tiny exhaustive interleaving explorer, shipped in this module the
way the Planning chapter shipped its own breadth-first planner.

The protocol is a two-process mutual-exclusion lock, the oldest exhaustively
checked example there is. ``write_mutex`` asks the model to write one as a small
guarded-command program; ``explore`` is the checker; ``fix_mutex`` feeds the
counterexample back. The model is gpt-oss:20b; its real protocols are captured
once by ``scripts/_distributed_probe.py`` and baked below, and the checker is
deterministic, so every trace and verdict on the page recomputes from the baked
text. The model proposes; the checker disposes.
"""
import ast
import re

import ollama

from genai.arch import EIS_MODEL


# ── The protocol DSL: a guarded-command program two processes share ────────────
# A protocol is a dict with two keys. ``shared`` is the initial shared state, a
# name -> value map (a bool, an int, or a list indexed by process id). ``program``
# is a list of [label, guard, action] steps that BOTH processes run in a loop,
# each binding ``me`` to its own id (0 or 1) and ``other`` to the other's. A step
# fires only when its guard holds; firing runs its action and advances that
# process to the next step. The step labelled "crit" is the critical section, and
# the safety rule the checker enforces is the one a lock exists for: never both
# processes at "crit" at once. The model writes a program in exactly this shape;
# we own the checker that runs it.

# A worked example shown on the page so the model (and the reader) learn the
# format without being handed a mutex: two processes that each tally how long the
# other is active. It uses every idiom the model needs, a per-process list indexed
# by ``flag[me]``, a guard that reads ``flag[other]``, ``me``/``other``, and
# arithmetic, but it is deliberately not a lock (it has no critical section).
EXAMPLE_PROTOCOL = {
    "shared": {"flag": [False, False], "ticks": [0, 0]},
    "program": [
        ["raise", None, "flag[me] = True"],                    # announce I'm active
        ["tick", "flag[other] == True", "ticks[me] = ticks[me] + 1"],  # count
        ["lower", None, "flag[me] = False"],                   # go idle
    ],
}

EXAMPLE_SRC = """{
  "shared": {"flag": [False, False], "ticks": [0, 0]},
  "program": [
    ["raise", None, "flag[me] = True"],                            # announce I'm active
    ["tick",  "flag[other] == True", "ticks[me] = ticks[me] + 1"], # count while both active
    ["lower", None, "flag[me] = False"],                           # go idle
  ],
}"""


# Three reference protocols, used to check the checker and seed the exercises. The
# demo never shows these; it shows the model's own program. NAIVE_FLAG sets its
# flag before testing the other's and deadlocks; CHECK_THEN_SET tests first and
# lets both in; PETERSON adds a turn variable and is the one that holds.
NAIVE_FLAG = {
    "shared": {"flag": [False, False]},
    "program": [
        ["set", None, "flag[me] = True"],
        ["wait", "flag[other] == False", None],
        ["crit", None, None],
        ["release", None, "flag[me] = False"],
    ],
}

CHECK_THEN_SET = {
    "shared": {"flag": [False, False]},
    "program": [
        ["wait", "flag[other] == False", None],
        ["set", None, "flag[me] = True"],
        ["crit", None, None],
        ["release", None, "flag[me] = False"],
    ],
}

PETERSON = {
    "shared": {"flag": [False, False], "turn": 0},
    "program": [
        ["set", None, "flag[me] = True"],
        ["turn", None, "turn = other"],
        ["wait", "flag[other] == False or turn == me", None],
        ["crit", None, None],
        ["release", None, "flag[me] = False"],
    ],
}

# A correct lock the checker still can't finish. An atomic test-and-set guards the
# section (mutual exclusion genuinely holds), but every entry is stamped with an
# ever-increasing number, the way a real ticket spinlock, sequence number, or
# logical clock does. That monotonic stamp makes the reachable state space infinite,
# which is the wall exhaustive search runs into and abstraction exists to get under.
TICKET_LOCK = {
    "shared": {"held": False, "stamp": 0},
    "program": [
        ["acquire", "held == False", "held = True"],  # atomic test-and-set
        ["crit", None, None],
        ["stamp", None, "stamp = stamp + 1"],          # an entry number that only climbs
        ["release", None, "held = False"],
    ],
}

# A lock that keeps the safety promise and breaks a different one. Mutual exclusion
# holds (the checker confirms it), but entry turns on a ``turn`` variable that
# nothing ever hands over, so process 1's guard is never satisfiable: it starves at
# the door while process 0 comes and goes as it likes. The liveness check, not the
# safety check, is what exposes it.
STARVING_LOCK = {
    "shared": {"turn": 0},
    "program": [
        ["wait", "turn == me", None],   # enter only when it's your turn
        ["crit", None, None],
        ["work", None, None],           # ... but the turn is never handed over
    ],
}

# The obvious first attempt at a shared one-slot buffer, and a broken one. Each
# worker checks that there's room and then, as a separate step, puts an item in;
# nothing makes those two happen together, so both can check an empty slot and both
# can fill it. Its safety rule isn't a label but a fact about the data, the
# invariant BUFFER_INVARIANT, which is what lets the same checker catch a race that
# has nothing to do with a critical section.
NAIVE_BUFFER = {
    "shared": {"count": 0},
    "program": [
        ["check", "count < 1", None],           # is there room on the shelf?
        ["put", None, "count = count + 1"],      # ... put an item on it
        ["take", None, "count = count - 1"],     # (and later take one back off)
    ],
}


# ── The checker: breadth-first search over every reachable interleaving ─────────
# A global state is (pc0, pc1, shared): where each process sits in the program and
# the shared variables. From a state, either process whose current step is enabled
# can fire, branching the search. BFS walks the whole reachable set, checks the
# invariant at each state, and returns the shortest interleaving that reaches a bad
# one. Small enough to read; the same idea TLC and Spin run at industrial scale.

_SAFE_BUILTINS = {"__builtins__": {"True": True, "False": False, "None": None}}


def _freeze(shared):
    """A hashable snapshot of the shared state, lists turned to tuples."""
    return tuple((k, tuple(v) if isinstance(v, list) else v)
                 for k, v in sorted(shared.items()))


def _enabled(program, pc, shared, me):
    """Is process ``me``'s current step allowed to fire in this state?"""
    guard = program[pc][1]
    if guard is None:
        return True
    ns = {"me": me, "other": 1 - me, **shared}
    return bool(eval(guard, dict(_SAFE_BUILTINS), ns))      # noqa: S307 - baked DSL


def _fire(program, pc, shared, me):
    """Run process ``me``'s current step: apply its action (on a copy of the
    shared state) and advance it to the next step. Returns (next_pc, next_shared)."""
    action = program[pc][2]
    local = {k: (list(v) if isinstance(v, list) else v) for k, v in shared.items()}
    local.update(me=me, other=1 - me)
    if action:
        exec(action, dict(_SAFE_BUILTINS), local)           # noqa: S102 - baked DSL
    nxt = {k: local[k] for k in shared}
    return (pc + 1) % len(program), nxt


def _in_crit(program, pc):
    """A process is inside the critical section while its step is labelled crit."""
    return program[pc][0] == "crit"


def _violates(invariant, program, pc0, pc1, shared) -> bool:
    """Does this global state break the safety rule? With no invariant the rule is
    mutual exclusion: never both processes at a ``crit`` step. A custom invariant is
    a boolean expression over the shared variables (plus ``at``, the label each
    process sits at, and ``in_crit``); the state violates it when it evaluates to
    False. The invariant is baked text, so nothing but the checker's own eval runs
    it."""
    if invariant is None:
        return _in_crit(program, pc0) and _in_crit(program, pc1)
    ns = {"at": [program[pc0][0], program[pc1][0]],
          "in_crit": [_in_crit(program, pc0), _in_crit(program, pc1)], **shared}
    return not bool(eval(invariant, dict(_SAFE_BUILTINS), ns))   # noqa: S307 - baked


def _inv_reason(invariant) -> str:
    """How the checker phrases a safety violation, for the trace and the verdict."""
    return ("both processes are in the critical section" if invariant is None
            else f"a reachable state breaks the invariant ({invariant})")


def validate(protocol, require_crit: bool = True, invariant: str = None) -> tuple:
    """Check a protocol has the right shape and its guards and actions actually
    run. Returns (ok, reason); a malformed program grades as a failed fix rather
    than crashing the checker on the model's typo. ``require_crit`` is on for a
    mutual-exclusion protocol (whose safety rule is about the ``crit`` label) and
    off when the safety rule is a data invariant over the shared state instead."""
    if not isinstance(protocol, dict) or "shared" not in protocol \
            or "program" not in protocol:
        return False, "not a {shared, program} protocol"
    prog = protocol["program"]
    if not prog:
        return False, "empty program"
    if require_crit and not any(_in_crit(prog, i) for i in range(len(prog))):
        return False, "no step labelled 'crit'"
    try:
        for pc in range(len(prog)):
            if len(prog[pc]) != 3:
                return False, "a step is not [label, guard, action]"
            for me in (0, 1):
                if _enabled(prog, pc, protocol["shared"], me):
                    _fire(prog, pc, protocol["shared"], me)
    except Exception as exc:                                 # noqa: BLE001 - report it
        return False, f"a guard or action failed to run ({type(exc).__name__})"
    if invariant is not None:
        try:
            _violates(invariant, prog, 0, 0, protocol["shared"])
        except Exception as exc:                             # noqa: BLE001 - report it
            return False, f"the invariant doesn't evaluate ({type(exc).__name__})"
    return True, "well formed"


def explore(protocol, max_states: int = 50000, invariant: str = None) -> dict:
    """Search every reachable interleaving of the two processes.

    Returns a verdict dict: ``status`` is "safe" (no reachable state breaks the
    safety rule, and the search was exhaustive), "unsafe" (a reachable state breaks
    it: mutual exclusion by default, or a custom ``invariant`` string when given),
    "deadlock" (a state where neither process can move),
    "unbounded" (the state space blew past ``max_states``, usually an ever-growing
    variable, so safety can't be settled), or "malformed". ``trace`` is the
    shortest interleaving to the bad state as a
    list of {pid, label, shared} steps; ``entered`` is the set of processes that
    can reach the critical section at all, the liveness check that rules out a
    protocol made "safe" by never letting anyone in.
    """
    ok, reason = validate(protocol, require_crit=invariant is None, invariant=invariant)
    if not ok:
        return {"status": "malformed", "reason": reason, "trace": None,
                "states": 0, "entered": set()}
    prog, shared0 = protocol["program"], protocol["shared"]
    start = (0, 0, shared0)
    startkey = (0, 0, _freeze(shared0))
    parent = {startkey: None}                # nkey -> (prevkey, pid, label, shared)
    entered = set()
    frontier = [start]
    if _violates(invariant, prog, 0, 0, shared0):    # broken before anyone moves
        return _verdict("unsafe", parent, startkey, prog, entered,
                        _inv_reason(invariant))
    while frontier:
        if len(parent) >= max_states:        # the search blew up, it didn't finish
            return {"status": "unbounded", "trace": None, "states": len(parent),
                    "entered": entered, "reason": f"state space exceeds {max_states} "
                    "states (an unbounded variable?), so safety can't be settled"}
        pc0, pc1, shared = frontier.pop(0)
        key = (pc0, pc1, _freeze(shared))
        moves = []
        for me, pc in ((0, pc0), (1, pc1)):
            if _enabled(prog, pc, shared, me):
                npc, nsh = _fire(prog, pc, shared, me)
                moves.append((me, prog[pc][0], (npc if me == 0 else pc0),
                              (npc if me == 1 else pc1), nsh))
                if _in_crit(prog, npc):
                    entered.add(me)
        if not moves:
            stuck = [prog[p][0] for p in (pc0, pc1)]
            return _verdict("deadlock", parent, key, prog, entered,
                            f"neither process can move (stuck at {stuck})")
        for me, label, npc0, npc1, nsh in moves:
            nkey = (npc0, npc1, _freeze(nsh))
            if nkey in parent:
                continue
            npc = npc0 if me == 0 else npc1
            parent[nkey] = (key, me, label, nsh, _in_crit(prog, npc))
            if _violates(invariant, prog, npc0, npc1, nsh):
                return _verdict("unsafe", parent, nkey, prog, entered,
                                _inv_reason(invariant))
            frontier.append((npc0, npc1, nsh))
    return {"status": "safe", "reason": "no interleaving breaks mutual exclusion",
            "trace": None, "states": len(parent), "entered": entered}


def _verdict(status, parent, key, program, entered, reason) -> dict:
    """Reconstruct the interleaving that led to ``key`` and package the verdict."""
    steps = []
    while parent[key] is not None:
        prevkey, pid, label, shared, enters = parent[key]
        steps.append({"pid": pid, "label": label, "shared": shared,
                      "enters_crit": enters})
        key = prevkey
    return {"status": status, "reason": reason, "trace": list(reversed(steps)),
            "states": len(parent), "entered": entered}


def grade_protocol(protocol) -> dict:
    """Grade a protocol the way a fix is judged: it must be safe under every
    interleaving AND still let both processes into the critical section. A program
    that deadlocks, lets both in, or quietly bars the section all fail."""
    v = explore(protocol)
    live = v["entered"] == {0, 1}
    return {"status": v["status"], "safe": v["status"] == "safe", "live": live,
            "good": v["status"] == "safe" and live}


def sequential_test(protocol, invariant: str = None) -> dict:
    """The test a developer writes first: run process 0 start to finish, then
    process 1, and confirm the safety rule holds the whole way. They never
    interleave, so a race can't surface and the test passes, which is exactly why it
    misses it. With no invariant the rule is mutual exclusion; a data invariant
    (a buffer that must never overflow, say) is checked after each step instead.
    Returns {passed, schedule}."""
    ok, _ = validate(protocol, require_crit=invariant is None, invariant=invariant)
    if not ok:
        return {"passed": False, "schedule": []}
    prog = protocol["program"]
    shared = {k: (list(v) if isinstance(v, list) else v)
              for k, v in protocol["shared"].items()}
    schedule, occupied = [], None
    for me in (0, 1):                          # one process fully, then the other
        pc = 0
        for _ in range(len(prog)):
            if not _enabled(prog, pc, shared, me):
                break
            label = prog[pc][0]
            npc, shared = _fire(prog, pc, shared, me)
            schedule.append((me, label))
            if invariant is None:
                if _in_crit(prog, pc) and occupied not in (None, me):
                    return {"passed": False, "schedule": schedule}
                occupied = me if _in_crit(prog, npc) else None
            elif _violates(invariant, prog, npc, 0, shared):
                return {"passed": False, "schedule": schedule}
            pc = npc
    return {"passed": True, "schedule": schedule}


# ── The model's two jobs: write the protocol, then fix it on the counterexample ─

def _emit(model: str, prompt: str, n_predict: int = 2500) -> str:
    """One model turn. gpt-oss reasons even with think off but lands the final dict
    in ``content`` once the budget is generous; we read both channels to be safe."""
    msg = ollama.chat(model=model, think=False, keep_alive="10m",
                      options={"num_predict": n_predict},
                      messages=[{"role": "user", "content": prompt}])["message"]
    return (msg.get("content") or "") + "\n" + (msg.get("thinking") or "")


def _literal(block: str):
    """ast.literal_eval a block, tolerating a model that wrote JSON-style
    null/true/false instead of Python None/True/False. No code ever runs."""
    for candidate in (block, re.sub(r"\b(null|true|false)\b",
                                    lambda m: {"null": "None", "true": "True",
                                               "false": "False"}[m.group(1)], block)):
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
    return None


def _pyify(expr):
    """Translate the C/JS operators a model sometimes leaks (``&&``, ``||``, ``!=``
    is fine) into Python, so a guard like ``flag[other] && turn == other`` runs."""
    if not isinstance(expr, str):
        return expr
    return expr.replace("&&", " and ").replace("||", " or ")


def _sanitize(protocol: dict) -> dict:
    """Clean a parsed protocol's guards and actions of non-Python operators."""
    protocol["program"] = [[label, _pyify(guard), _pyify(action)]
                           for label, guard, action in protocol["program"]]
    return protocol


def _extract(text: str):
    """Pull the last ``{...}`` protocol literal out of a model reply and parse it
    (no code runs). Returns the dict, or None if none parses."""
    best = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            depth += (text[j] == "{") - (text[j] == "}")
            if depth == 0:
                obj = _literal(text[i:j + 1])
                if isinstance(obj, dict) and "program" in obj \
                        and isinstance(obj.get("program"), list) \
                        and all(isinstance(s, list) and len(s) == 3
                                for s in obj["program"]):
                    best = _sanitize(obj)          # keep the last well-formed one
                break
    return best


_FORMAT = (
    "Write it as a Python dict literal in exactly this format, where two processes "
    "each run the program in a loop, binding `me` to their own id (0 or 1) and "
    "`other` to the other (1 - me):\n\n" + EXAMPLE_SRC + "\n\n"
    "Each step is [label, guard, action]: the guard is a Python boolean over the "
    "shared variables (or None to always run), the action assigns one shared "
    "variable (or None for none). Index a per-process variable by `[me]` and "
    "`[other]`, like `flag[me]`. Label the critical section \"crit\". Output only "
    "the dict.")


def write_mutex(model: str = EIS_MODEL, tries: int = 4) -> str:
    """Ask the model for a two-process mutual-exclusion lock in the DSL. We ask for
    the *simplest* one on purpose: that's the prompt that tempts a fluent model
    into a flag protocol with an interleaving bug, the failure the checker exists
    to catch. The DSL is unfamiliar, so the model sometimes emits something the
    checker can't read or that runs an unbounded counter; we retry past those (a
    real system retries on a spec it can't even check) so the result is a protocol
    the checker can actually rule on. Returns the protocol source."""
    prompt = (
        "Two processes share one resource and must never be in their critical "
        "section at the same time. Write the simplest mutual-exclusion lock you "
        "can, using one boolean flag per process so each can tell the other it "
        "wants in.\n\n" + _FORMAT)
    last = ""
    for _ in range(tries):
        obj = _extract(_emit(model, prompt))
        if obj is None:
            continue
        last = _format_src(obj)
        if explore(obj)["status"] in ("unsafe", "deadlock", "safe"):
            return last                      # a protocol the checker could settle
    return last


def fix_mutex(broken_src: str, verdict: dict, model: str = EIS_MODEL,
              guided: bool = True) -> str:
    """Ask the model to repair a broken protocol. When ``guided``, hand it the
    concrete counterexample the checker found; otherwise only tell it the protocol
    is wrong. The contrast between the two is the chapter's measurement: does the
    trace actually help the model fix the interleaving?"""
    if guided:
        evidence = ("A model checker explored every interleaving and found this "
                    "one that breaks it:\n\n" + trace_text(verdict) + "\n\n"
                    "Fix the protocol so no interleaving can reach that state.")
    else:
        evidence = ("A model checker found an interleaving that breaks it. Fix the "
                    "protocol so no interleaving violates mutual exclusion.")
    prompt = ("Here is a mutual-exclusion protocol:\n\n" + broken_src + "\n\n"
              + evidence + " Both processes must still be able to enter the "
              "critical section. Keep the same format and output only the dict.")
    text = _emit(model, prompt)
    obj = _extract(text)
    return _format_src(obj) if obj else text.strip()


def fix_study(trials: int = 12, model: str = EIS_MODEL,
              broken: dict = None) -> dict:
    """Measure the fixed-on-counterexample rate. For ``trials`` rounds, ask the
    model to repair the same broken protocol twice, once blind and once handed the
    counterexample trace, and count how often each repair is actually safe-and-live
    by the checker. Real numbers, no rerun of good data once baked."""
    broken = broken or CHECK_THEN_SET
    broken_src = _format_src(broken)
    verdict = explore(broken)
    out = {"blind": 0, "guided": 0, "trials": trials, "model": model}
    for _ in range(trials):
        for key, guided in (("blind", False), ("guided", True)):
            fixed = parse_protocol(fix_mutex(broken_src, verdict, model, guided))
            out[key] += int(bool(fixed) and grade_protocol(fixed)["good"])
    return out


def parse_protocol(src: str):
    """Parse protocol source back to a dict for the checker; None if it won't."""
    if isinstance(src, dict):
        return src
    return _extract(src)


# ── Pretty-printing protocols and traces for the page ──────────────────────────

def _format_src(protocol: dict) -> str:
    """Render a protocol dict as clean, aligned DSL source for show_code."""
    shared = ", ".join(f'"{k}": {_lit(v)}' for k, v in protocol["shared"].items())
    rows = [f'  "shared": {{{shared}}},', '  "program": [']
    width = max((len(s[0]) for s in protocol["program"]), default=0)
    for label, guard, action in protocol["program"]:
        lbl = f'"{label}",'.ljust(width + 3)
        rows.append(f'    [{lbl} {_lit(guard)}, {_lit(action)}],')
    rows.append("  ],")
    return "{\n" + "\n".join(rows) + "\n}"


def _lit(v) -> str:
    """One protocol value as source: a quoted guard/action, a list, or None."""
    if v is None:
        return "None"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def trace_text(verdict: dict) -> str:
    """The counterexample as a plain interleaving: one line per step, the process
    that moved, the step it ran, and the shared state after it."""
    if not verdict.get("trace"):
        return "(no counterexample)"
    lines = []
    for step in verdict["trace"]:
        shared = ", ".join(f"{k}={v}" for k, v in step["shared"].items())
        mark = "  <- in crit" if step.get("enters_crit") else ""
        lines.append(f"P{step['pid']}: {step['label']:<8} {shared}{mark}")
    return "\n".join(lines)


# ── Captured runs (gpt-oss:20b via scripts/_distributed_probe.py), baked ───────
# BROKEN_DEMO is the model's REAL protocol, its labels, guards, and ordering exactly
# as written, reserialized only for aligned display. It reads like a lock and passes
# the sequential test, but the model inverted the wait: each process waits for the
# OTHER's flag to go up, not down, so the flags rendezvous the two into the section
# instead of guarding it. The verdict and trace on the page are recomputed live by
# explore() from this text, so nothing is hand-asserted; only the model's program is
# baked.
BROKEN_DEMO = {
    "src": """{
  "shared": {"flag": [False, False]},
  "program": [
    ["raise", None, "flag[me] = True"],
    ["wait",  "flag[other] == True", None],
    ["crit",  None, None],
    ["lower", None, "flag[me] = False"],
  ],
}""",
}

# FIXED_DEMO is a REAL guided repair the model produced after being shown the
# counterexample (its third guided attempt; the first two still failed the checker).
# It's Peterson's algorithm: the model adds the turn variable two booleans lacked,
# and the checker certifies it safe over the whole state space, with both processes
# still able to enter. The repair bundles the flag-set and turn-set into one atomic
# step, which is the model's choice of granularity; the checker checks exactly that.
FIXED_DEMO = {
    "src": """{
  "shared": {"flag": [False, False], "turn": 0},
  "program": [
    ["raise", None, "flag[me] = True; turn = other"],
    ["wait",  "flag[other]==False or turn!=other", None],
    ["crit",  None, None],
    ["lower", None, "flag[me] = False"],
  ],
}""",
}

# FIX_STUDY: the real fixed-on-counterexample rates (scripts/_distributed_probe.py,
# gpt-oss:20b, n=8 each). The model repairs its broken lock rarely either way, and
# being handed the exact failing interleaving did not help: 2/8 fixes passed the
# checker on a blind retry, 1/8 when shown the counterexample. The capture told the
# same story, the model's first two guided repairs still failed before one held.
FIX_STUDY = {"blind": 2, "guided": 1, "trials": 8, "model": EIS_MODEL}


# ── Showing the work on the page (built on the transcript helpers) ─────────────

def show_broken_demo(demo: dict = None) -> None:
    """ACT 1: the model writes a lock, it passes the obvious test, and the checker
    finds the interleaving that breaks it. The protocol is the model's real output;
    the test result and counterexample are recomputed live, so the page stays
    honest for whatever protocol is baked."""
    from genai.agent import show_code, show_turn
    demo = demo or BROKEN_DEMO
    protocol = parse_protocol(demo["src"])
    show_turn("you", "Write a two-process mutual-exclusion lock.")
    show_code("gpt-oss", demo["src"])
    test = sequential_test(protocol)
    show_turn("TEST", "run process 0, then process 1: never both in the critical "
                      "section -> " + ("passes" if test["passed"] else "fails"))
    verdict = explore(protocol)
    if verdict["status"] == "safe":
        show_turn("CHECKER", f"explored {verdict['states']} interleavings: safe")
    else:
        show_turn("CHECKER", f"explored the state space and found a {len(verdict['trace'])}"
                             f"-step interleaving that fails ({verdict['reason']}):")
        for line in trace_text(verdict).split("\n"):
            show_turn("", line)
        show_turn("CHECKER", "-> " + verdict["reason"] + ": the lock is broken")


def show_fixed_demo(demo: dict = None, broken: dict = None) -> None:
    """ACT 2: hand the model the counterexample and it repairs the protocol; the
    checker now certifies it safe across the whole reachable state space."""
    from genai.agent import show_code, show_turn
    demo = demo or FIXED_DEMO
    protocol = parse_protocol(demo["src"])
    show_turn("you", "Here is the interleaving that breaks it. Fix the protocol.")
    show_code("gpt-oss", demo["src"])
    verdict = explore(protocol)
    grade = grade_protocol(protocol)
    if grade["good"]:
        show_turn("CHECKER", f"explored {verdict['states']} interleavings: safe, and "
                             "both processes still reach the critical section")
    elif verdict["status"] == "safe":
        show_turn("CHECKER", "safe, but a process can no longer enter: the lock "
                             "deadlocks the section shut")
    else:
        show_turn("CHECKER", f"still broken: {verdict['reason']}")


def show_reference(protocol: dict, name: str) -> None:
    """Print one reference protocol and the checker's verdict on it, for exercises."""
    from genai.agent import show_code, show_turn
    show_code(name, _format_src(protocol))
    v = explore(protocol)
    show_turn("CHECKER", f"{v['status']}: {v['reason']} ({v['states']} states)")


# ════════════════════════════════════════════════════════════════════════════════
# Beyond one lock and one property. Everything above checks mutual exclusion on a
# two-flag lock. The rest of the module stretches the SAME checker four ways the
# chapter needs: a lock that's safe but dead (liveness), a lock whose state space is
# infinite (the checker's wall), a data invariant instead of a control one (a
# shared buffer), and an invariant the model writes for itself. The model half of
# each is captured once by scripts/_distributed_probe.py and baked near the bottom
# of this file; the checker half recomputes live on the page.
# ════════════════════════════════════════════════════════════════════════════════


# ── Watching the state space explode (deterministic; no model) ──────────────────

def counting_protocol(k: int) -> dict:
    """A throwaway two-process protocol whose only job is to have a reachable state
    space we can grow on demand. Each process just ticks its own counter within a
    fixed range ``k``; there's no lock and no critical section here. Raise ``k`` and
    the reachable combinations multiply, which is how we watch a state space swell
    without changing any logic. Checked with the always-true invariant so the search
    explores the whole space instead of stopping at a violation."""
    step = "n[me] = (n[me] + 1) % " + str(k)
    return {"shared": {"n": [0, 0]}, "program": [["tick", None, step]]}


def explosion_series(ks=(2, 3, 4, 5, 6, 7, 8)) -> list:
    """For each ``k``, how many states the checker explores on counting_protocol(k):
    the reachable space growing as the processes get more room. The exponent is the
    process count, so the same curve steepens sharply with every machine you add.
    Deterministic; no model involved."""
    out = []
    for k in ks:
        v = explore(counting_protocol(k), invariant="True")
        out.append({"k": k, "states": v["states"]})
    return out


# ── The model's four new jobs (captured by the probe, baked below) ──────────────

# The safety rule for the shared buffer isn't about a label; it's a fact about the
# data: the one-slot buffer must never hold more than one item.
BUFFER_INVARIANT = "count <= 1"


def write_with_invariant(model: str = EIS_MODEL, tries: int = 6) -> tuple:
    """Ask the model, in one breath, for a lock AND the safety property a checker
    should enforce on it. The chapter's question is whether the invariant the model
    writes is strong enough to catch its own bug. Returns (protocol_src, invariant),
    the invariant a Python boolean over the shared variables (and ``in_crit``, a
    two-element list of whether each process is in its section)."""
    prompt = (
        "Two processes share one resource and must never be in their critical "
        "section at the same time. Write the simplest mutual-exclusion lock you can "
        "using one boolean flag per process, AND write the safety property a model "
        "checker should verify to prove it correct.\n\n" + _FORMAT + "\n\n"
        "Then, on a new line after the dict, write SAFETY: followed by a single "
        "Python boolean expression that must be true in every reachable state. You "
        "can refer to the shared variables and to in_crit[0] and in_crit[1], which "
        "say whether each process is currently in its critical section. Format that "
        "line exactly like this example for a different protocol:\nSAFETY: count >= 0")
    last = ("", "")
    for _ in range(tries):
        text = _emit(model, prompt)
        obj = _extract(text)
        inv = _extract_invariant(text)
        if obj is None:
            continue
        last = (_format_src(obj), inv)
        if inv:
            return last
    return last


def _extract_invariant(text: str):
    """Pull the model's ``SAFETY: <expr>`` line out of a reply, cleaned of backticks
    and C/JS operators so the checker can eval it. None if there's no such line."""
    m = re.search(r"SAFETY:\s*(.+)", text)
    if not m:
        return None
    expr = m.group(1).strip().strip("`\"' ").rstrip(".").strip("`\"' ")
    return _pyify(expr) or None


# ── Showing the four new demos on the page (verdicts recompute live) ────────────

def show_liveness_demo(protocol: dict = None) -> None:
    """A lock that keeps the safety promise and breaks a different one. STARVING_LOCK
    guards the section correctly (two are never both inside, which the checker
    confirms) but never hands the turn back, so process 0 comes and goes while
    process 1 waits forever. ``explore`` says the reassuring word, safe; the liveness
    check, the set of processes that reach the section at all, exposes the
    starvation."""
    from genai.agent import show_code, show_turn
    protocol = protocol or STARVING_LOCK
    show_code("lock", _format_src(protocol))
    v = explore(protocol)
    g = grade_protocol(protocol)
    show_turn("CHECKER", f"explored {v['states']} interleavings: {v['status']}, "
                         "never both in the critical section")
    missing = sorted({0, 1} - v["entered"])
    if v["status"] == "safe" and missing:
        who = ("neither process ever reaches" if len(missing) == 2
               else f"process {missing[0]} never reaches")
        show_turn("LIVENESS", f"but {who} the critical section: the lock is safe, "
                              "and starves whoever didn't go first")
    else:
        show_turn("LIVENESS", f"reached by {sorted(v['entered'])}: "
                              + ("live" if g["live"] else "not live"))


# ── A rule the checker can't phrase: the ceiling of the invariant language ──────
# "Safe Isn't the Same as Working" stood liveness in for fairness with a crude
# reachability check: which processes reach the section at all. That stand-in has a
# ceiling. A plain test-and-set spinlock is safe and lets both processes in, so the
# checker blesses it, yet it makes no promise that a waiting process ever gets its
# turn: process 0 can retake the lock the instant it frees it and starve process 1
# down an unlucky run. The real property, "once waiting, a process eventually
# enters", is a claim about the whole run, not any one state, so no boolean over a
# single state can phrase it. That isn't the state-space wall of the next section;
# it's the ceiling of the invariant *language*, and it's what temporal logic
# (LTL/CTL) exists to lift. Fully deterministic; no model.

SPINLOCK = {
    "shared": {"held": False},
    "program": [
        ["acquire", "held == False", "held = True"],   # atomic test-and-set
        ["crit", None, None],
        ["release", None, "held = False"],
    ],
}

_TEMPORAL_OPS = ("always", "eventually", "until", "next", "globally", "finally",
                 "infinitely", "henceforth")

# The two promises a real lock owes you: one the checker's language can hold, one it
# can't. _SAFETY is a boolean over one state; _FAIRNESS talks about the run over time.
_SAFETY_RULE = "not (in_crit[0] and in_crit[1])"
_FAIRNESS_RULE = "always(wants_in[i] implies eventually(in_crit[i]))"


def classify_property(expr: str) -> tuple:
    """Sort a proposed property into what the invariant language can hold. A *state*
    property is a boolean over one state's variables, which ``explore`` can test at
    every reachable state. A *temporal* property talks about the run over time
    (eventually, always-in-the-future), which no single state decides, so this
    checker can't express it. Returns ``(kind, operators_found)``."""
    found = [op for op in _TEMPORAL_OPS if re.search(rf"\b{op}\b", expr, re.I)]
    return ("temporal", found) if found else ("state", [])


def show_fairness_ceiling(lock: dict = None) -> None:
    """A lock the checker calls safe and live, and the promise it still can't judge.
    Mutual exclusion is a test on one state, so the checker runs it everywhere;
    no-starvation is a claim about the whole run, which no state test can decide, so
    the checker reports it unexpressible rather than pass or fail it, the ceiling of
    the invariant language rather than of the state space."""
    from genai.agent import show_code, show_turn
    lock = lock or SPINLOCK
    v = explore(lock)
    reach = "both processes reach the section" if v["entered"] == {0, 1} \
        else f"reached by {sorted(v['entered'])}"
    show_code("lock", _format_src(lock))
    show_turn("CHECKER", f"safe over {v['states']} interleavings, and {reach}: "
                         "safe and live")
    show_turn("SAFETY", f"'never both inside' is  {_SAFETY_RULE}")
    show_turn("", f"a test on one state ({classify_property(_SAFETY_RULE)[0]}): "
                  "the checker runs it at every reachable state")
    show_turn("FAIRNESS", f"'once waiting, a process eventually enters' is  "
                          f"{_FAIRNESS_RULE}")
    ops = classify_property(_FAIRNESS_RULE)[1]
    show_turn("", f"{' and '.join(ops)} ask about the whole run, not one state: "
                  "unexpressible here, so the checker can't even pose the rule")


def show_ticket_demo(protocol: dict = None) -> None:
    """A correct lock the checker still can't verify. TICKET_LOCK guards the section
    with an atomic test-and-set (mutual exclusion genuinely holds) and stamps every
    entry with an ever-increasing number, the way a real sequence number or logical
    clock does. Bound that stamp and the checker certifies the lock safe; leave it
    climbing and the state space is infinite, so the search reports ``unbounded``
    rather than pretend it finished."""
    from genai.agent import show_code, show_turn
    protocol = protocol or TICKET_LOCK
    show_code("ticket lock", _format_src(protocol))
    vb = explore(_bound_stamp(protocol, 3))
    show_turn("CHECKER", "with the entry count bounded, the lock itself checks out: "
                         f"{vb['status']}, mutual exclusion holds ({vb['states']} "
                         "states)")
    v = explore(protocol, max_states=6000)     # a low ceiling: enough to see it can't close
    show_turn("CHECKER", "as written the count only ever climbs, so it gave up after "
                         f"{v['states']:,} states: {v['reason']}")


def _bound_stamp(protocol: dict, cap: int) -> dict:
    """A copy of a ticket lock with its entry counter wrapped to a fixed range, so
    the checker has a finite space to search. This is the abstraction that makes an
    infinite protocol checkable, done by hand: bound the number that was climbing."""
    prog = [list(step) for step in protocol["program"]]
    for step in prog:
        if step[0] == "stamp":
            step[2] = "stamp = (stamp + 1) % " + str(cap)
    return {"shared": dict(protocol["shared"]), "program": prog}


def show_buffer_demo(protocol: dict = None) -> None:
    """The same checker, a different invariant. Two workers share a one-slot buffer;
    the safety rule isn't about a label but about the data, BUFFER_INVARIANT (never
    more than one item in the slot). The naive check-then-put passes the sequential
    test, and the checker finds the interleaving where both look, both see room, and
    both put."""
    from genai.agent import show_code, show_turn
    protocol = protocol or NAIVE_BUFFER
    show_code("buffer", _format_src(protocol))
    seq = sequential_test(protocol, BUFFER_INVARIANT)["passed"]
    show_turn("TEST", "run one worker, then the other: count never exceeds 1 -> "
                      + ("passes" if seq else "fails"))
    v = explore(protocol, invariant=BUFFER_INVARIANT)
    if v["status"] == "unsafe":
        show_turn("CHECKER", f"explored the interleavings and found a "
                             f"{len(v['trace'])}-step one that overflows the slot:")
        for line in trace_text(v).split("\n"):
            show_turn("", line)
        show_turn("CHECKER", "-> two items in a one-item buffer: the check-then-put "
                             "raced")
    else:
        show_turn("CHECKER", f"{v['status']}: {v['reason']} ({v['states']} states)")


def show_invariant_demo(demo: dict = None) -> None:
    """The model writes the protocol AND the property to judge it by. The finding is
    the opposite of the worry: it states the property correctly (never both in the
    critical section) and it's the protocol that's broken. Handed the model's own
    rule, the checker convicts the model's own lock. INVARIANT_DEMO is the model's
    real lock and invariant; the verdict recomputes live."""
    from genai.agent import show_code, show_turn
    demo = demo or INVARIANT_DEMO
    protocol = parse_protocol(demo["src"])
    show_turn("you", "Write a lock and the safety property to check it against.")
    show_code("gpt-oss", demo["src"])
    show_turn("gpt-oss", "SAFETY: " + demo["invariant"])
    v = explore(protocol, invariant=demo["invariant"])   # its lock vs its own rule
    if v["status"] == "unsafe":
        show_turn("CHECKER", "against the model's own rule, its lock is unsafe; a "
                             f"{len(v['trace'])}-step interleaving reaches the state "
                             "its author declared forbidden:")
        for line in trace_text(v).split("\n"):
            show_turn("", line)
        show_turn("CHECKER", "-> the model wrote the right rule and broke it anyway")
    else:
        show_turn("CHECKER", f"{v['status']}: {v['reason']} ({v['states']} states)")


# ── Captured runs for the four new demos (gpt-oss:20b), baked by the probe ──────
# Placeholders until scripts/_distributed_probe.py runs and its output is pasted in.
# Each *_DEMO["src"] must be the model's REAL protocol text; the checker recomputes
# every verdict from it, so nothing on the page is hand-asserted.
# INVARIANT_DEMO (scripts/_distributed_probe.py invariant, gpt-oss:20b): asked for a
# lock AND the property to check it by, the model wrote the mutual-exclusion invariant
# exactly right (never both in the critical section) and a lock that violates it, so
# the checker convicts the model's protocol with the model's own rule. Real output;
# the trace recomputes live from this text.
INVARIANT_DEMO = {
    "src": """{
  "shared": {"flag": [False, False], "ticks": [0, 0]},
  "program": [
    ["raise", None, "flag[me] = True"],
    ["crit",  "flag[other] == False", "ticks[me] = ticks[me] + 1"],
    ["lower", None, "flag[me] = False"],
  ],
}""",
    "invariant": 'not (in_crit[0] and in_crit[1])',
}

# The write distribution (scripts/_distributed_probe.py write, gpt-oss:20b): asked
# for its simplest lock six times, the model wrote a mutual-exclusion violation every
# time, and every one passed the developer's sequential test. The checker certified
# none of them. plot_false_green(WRITE_SEQ_PASS, WRITE_DIST.get("safe", 0), WRITE_N).
WRITE_DIST = {"unsafe": 6}
WRITE_SEQ_PASS = 6
WRITE_N = 6


# ── Two clocks disagree: logical time vs the wall clock ───────────────────────
# The model checker above assumed one global order of events. Two of Sophia's
# sub-agents on different machines don't have one: each stamps its own log by its
# own clock, and the clocks drift. Merge the two logs by wall-clock time and a
# reply can land before the message it answered, a causal impossibility the merged
# record now asserts. The fix is a *Lamport clock*: a counter each agent bumps on
# every event and carries on every message, so a receive always stamps higher than
# the send it answers. Same events, same messages; only the timestamp changes.
# Fully deterministic, so the cell runs live.

# Five messages between a planner and a worker, in true causal order. Each is
# (sender, receiver, label); a send is always followed by its receive.
CLOCK_TRACE = [
    ("planner", "worker", "assign: summarize"),
    ("worker", "planner", "ack: on it"),
    ("planner", "worker", "assign: extract dates"),
    ("worker", "planner", "result: 3 dates"),
    ("worker", "planner", "result: summary ready"),
]

# The worker's clock runs six ticks fast, so its send stamps outrun the planner's.
CLOCK_SKEW = {"planner": 0, "worker": 6}


def simulate_clocks(trace: list = None, skew: dict = None, latency: int = 1) -> list:
    """Replay the exchange, stamping each event two ways. Returns a list of event
    dicts with a true order, a ``wall`` stamp (true time plus the agent's skew),
    and a ``lamport`` stamp (a counter bumped per event and carried on messages)."""
    trace = trace or CLOCK_TRACE
    skew = skew or CLOCK_SKEW
    events, lamport, t = [], {a: 0 for a in skew}, 0
    for i, (src, dst, label) in enumerate(trace):
        t += 1
        lamport[src] += 1
        events.append({"true": t, "agent": src, "kind": "send", "msg": i,
                       "label": label, "wall": t + skew[src],
                       "lamport": lamport[src]})
        stamp, t = lamport[src], t + latency
        lamport[dst] = max(lamport[dst], stamp) + 1
        events.append({"true": t, "agent": dst, "kind": "recv", "msg": i,
                       "label": label, "wall": t + skew[dst],
                       "lamport": lamport[dst]})
    return events


def clock_violations(events: list, key: str) -> tuple:
    """Order the merged log by ``key`` (``"wall"`` or ``"lamport"``) and count the
    messages whose receive is logged before its own send, a causal impossibility.
    Returns ``(count, ordered_events)``."""
    order = sorted(events, key=lambda e: (e[key], e["agent"]))
    pos = {(e["msg"], e["kind"]): p for p, e in enumerate(order)}
    msgs = {e["msg"] for e in events}
    bad = sum(pos[(m, "recv")] < pos[(m, "send")] for m in msgs)
    return bad, order


def show_two_clocks() -> None:
    """The merged wall-clock log, with each reply logged before it was sent, then
    the same events under Lamport stamps where no receive precedes its send."""
    from genai.agent import show_turn
    events = simulate_clocks()
    n = len({e["msg"] for e in events})
    bad_wall, order = clock_violations(events, "wall")
    send_pos = {e["msg"]: p for p, e in enumerate(order) if e["kind"] == "send"}
    show_turn("SETUP", "planner and worker; the worker's clock runs 6 ticks fast")
    for p, e in enumerate(order):
        flag = "   <- logged before its send" if (
            e["kind"] == "recv" and p < send_pos[e["msg"]]) else ""
        show_turn(f"wall {e['wall']:>2}", f"{e['agent']:<7} {e['kind']} "
                  f"{e['label']}{flag}")
    bad_lamport, _ = clock_violations(events, "lamport")
    show_turn("WALL CLOCK", f"{bad_wall} of {n} messages received before they were sent")
    show_turn("LAMPORT", f"{bad_lamport} of {n}: a receive always stamps above its send")

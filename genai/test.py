"""Tests a model writes, and a way to tell whether they're any good.

Testing is the cheapest rung of the correctness ladder: you don't prove the code
right, you just try to catch it being wrong. A model is a fast, willing test
writer, so the temptation is to let it write the suite and trust the green bar.
The trouble is that a green bar only says the tests it thought of passed; it says
nothing about the cases it didn't. *Mutation testing* is the quality signal that
fills that hole. Make a small change to the code, a `>=` flipped to `>`, a constant
nudged by one, and a good suite should turn red; if every mutant still passes, the
suite isn't really pinning the behavior down.

The function under test is ``letter_grade``, deliberately tiny so the whole
machine is visible: a model writes the tests, a built-in mutator scores them, and
a writer/mutator loop drives the score up. The mutator is pure Python (no
``mutmut`` to install); the only nondeterminism is the model, so the captured
constants are baked once by ``scripts/_test_mutation_probe.py`` and the cells
that show them are frozen. The test writer is a code model (qwen2.5-coder); a
reasoning model writes tests too, but a code model writes them cleaner.
"""
import ast

from genai.llm import CODING_MODEL, ask as _ask

# ── The function under test: a spec, a correct version, a shipped-with-a-bug one ─
# letter_grade maps a 0-100 score to a letter. The spec is what the tests should
# encode; CORRECT obeys it; BUGGY is what actually got shipped, with one boundary
# wrong (90 falls through to a B). Both are strings because the mutator parses and
# rewrites their source, and a demo runs each against the same suite.

SPEC = ("letter_grade(score) maps a whole-number score from 0 to 100 to a letter: "
        "90 and up is 'A', 80 to 89 is 'B', 70 to 79 is 'C', 60 to 69 is 'D', and "
        "anything below 60 is 'F'.")

CORRECT = (
    "def letter_grade(score):\n"
    "    if score >= 90: return 'A'\n"
    "    if score >= 80: return 'B'\n"
    "    if score >= 70: return 'C'\n"
    "    if score >= 60: return 'D'\n"
    "    return 'F'\n")

# The planted bug: >= became > at the top boundary, so a 90 is graded a B.
BUGGY = CORRECT.replace("if score >= 90", "if score > 90")


# ── A tiny built-in mutator ───────────────────────────────────────────────────
# One single-point edit per mutant: a comparison swapped for its neighbour, or an
# integer constant nudged by one. Each is the kind of small slip a real bug looks
# like, so killing all of them means the suite is sensitive where it counts.

_CMP_SWAP = {ast.GtE: ast.Gt, ast.Gt: ast.GtE, ast.LtE: ast.Lt, ast.Lt: ast.LtE,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_SYM = {ast.GtE: ">=", ast.Gt: ">", ast.LtE: "<=", ast.Lt: "<",
        ast.Eq: "==", ast.NotEq: "!="}


def _ops(tree) -> list:
    """Every single-point mutation on this tree, as ``(label, apply)`` pairs.

    ``apply`` mutates the tree in place, so each mutant re-parses a fresh tree and
    applies exactly one op; ``ast.walk`` is stable, so the order is reproducible.
    """
    ops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP_SWAP:
            old = type(node.ops[0]); new = _CMP_SWAP[old]
            ops.append((f"{_SYM[old]}->{_SYM[new]}",
                        lambda node=node, new=new: node.ops.__setitem__(0, new())))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            for d in (1, -1):
                ops.append((f"{node.value}->{node.value + d}",
                            lambda node=node, d=d: setattr(node, "value", node.value + d)))
    return ops


def make_mutants(source: str = CORRECT) -> list:
    """Return ``(label, mutated_source)`` for every single-point mutant of source."""
    n = len(_ops(ast.parse(source)))
    out = []
    for i in range(n):
        tree = ast.parse(source)
        label, apply = _ops(tree)[i]
        apply()
        out.append((label, ast.unparse(ast.fix_missing_locations(tree))))
    return out


# ── Running a suite ───────────────────────────────────────────────────────────

def _tests_in(suite: str, func_source: str) -> list:
    """Exec ``func_source`` then ``suite`` and collect the test_* callables."""
    ns = {}
    exec(func_source, ns)        # noqa: S102 - controlled local demo
    exec(suite, ns)              # noqa: S102 - controlled local demo
    return [(k, v) for k, v in ns.items() if k.startswith("test_") and callable(v)]


def run_suite(suite: str, func_source: str = CORRECT) -> list:
    """Run every test in ``suite`` against ``func_source``; return the failures.

    Each failure is ``(test_name, reason)``. A test that raises anything counts as
    a failure, which is what we want: a mutant that makes a test crash is caught.
    """
    failures = []
    for name, fn in _tests_in(suite, func_source):
        try:
            fn()
        except Exception as exc:
            failures.append((name, type(exc).__name__))
    return failures


def green_suite(suite: str, func_source: str = CORRECT) -> str:
    """Keep only the tests that pass on ``func_source``, dropping broken ones.

    A test that errors on the correct code is encoding the wrong spec or referring
    to something undefined; it would 'kill' every mutant for the wrong reason, so a
    mutation score only means something once the suite is green on correct code.
    """
    keep = [name for name, fn in _tests_in(suite, func_source)
            if not _raises(fn)]
    return _select(suite, keep)


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


def _select(suite: str, names: list) -> str:
    """Return the suite source with only the named test functions kept, in order."""
    tree = ast.parse(suite)
    tree.body = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    return ast.unparse(tree)


def n_tests(suite: str) -> int:
    return sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
               for n in ast.parse(suite).body)


# ── Mutation score ────────────────────────────────────────────────────────────

def mutation_score(suite: str, func_source: str = CORRECT) -> dict:
    """Fraction of mutants the suite kills, plus the survivors and their witnesses.

    A mutant is *killed* if any test fails on it. A *survivor* slipped past every
    test; its witness is an input where the mutant disagrees with the correct
    function, the concrete case the suite forgot to check. Returns
    ``{killed, total, score, survivors}`` where each survivor is
    ``{label, witness, want, got}``.
    """
    killed, survivors = 0, []
    for label, mutant in make_mutants(func_source):
        if run_suite(suite, mutant):
            killed += 1
        else:
            survivors.append({"label": label, **_witness(mutant, func_source)})
    total = killed + len(survivors)
    return {"killed": killed, "total": total,
            "score": round(killed / total, 3) if total else 1.0,
            "survivors": survivors}


def _witness(mutant: str, func_source: str = CORRECT) -> dict:
    """An input in 0..100 where the mutant grades differently from the correct
    function. This is what a surviving mutant points at: a case worth a test."""
    ref, mut = {}, {}
    exec(func_source, ref); exec(mutant, mut)   # noqa: S102 - controlled demo
    for score in range(0, 101):
        want, got = ref["letter_grade"](score), mut["letter_grade"](score)
        if want != got:
            return {"witness": score, "want": want, "got": got}
    return {"witness": None, "want": None, "got": None}   # equivalent mutant


# ── The mutant you can't kill: equivalent mutants ─────────────────────────────
# The writer/mutator loop reached a perfect score because every mutant of
# letter_grade is a real bug. That isn't guaranteed. Some mutants are *equivalent*:
# the same function wearing a different operator, identical output on every input,
# so no test can kill them and no suite ever scores a full 100%. Chasing that last
# point is then wasted effort. Here is the smallest function with one, a low clamp
# whose `< 0` reads the same as `<= 0`. Deterministic; no model.

CLAMP = ("def clamp_low(x):\n"
         "    if x < 0:\n"
         "        return 0\n"
         "    return x\n")


def show_unkillable() -> None:
    """A surviving mutant no test can kill. Flipping clamp_low's `<` to `<=`
    changes nothing: at x=0, the only input where the operator matters, both
    versions return 0. An exhaustive check over a wide range finds no distinguishing
    input, the proof the survivor is equivalent, not a hole in the suite."""
    from genai.agent import show_code, show_turn
    show_code("under test", CLAMP.rstrip())
    show_turn("MUTATOR", "flip one operator: `x < 0` becomes `x <= 0`")
    ref, mut = {}, {}
    exec(CLAMP, ref)                               # noqa: S102 - controlled demo
    exec(CLAMP.replace("x < 0", "x <= 0"), mut)   # noqa: S102 - controlled demo
    lo, hi = -100_000, 100_000
    witness = next((x for x in range(lo, hi + 1)
                    if ref["clamp_low"](x) != mut["clamp_low"](x)), None)
    if witness is None:
        show_turn("KILL?", f"every input from {lo:,} to {hi:,} tried: not one makes "
                           "the two versions disagree")
        show_turn("VERDICT", "the mutant is equivalent, the same function in "
                             "disguise; no test can kill it, so the missing point is "
                             "not a gap in the suite")
    else:
        show_turn("KILL?", f"they disagree at x={witness}: a real gap, add a test")


# ── The writer: a model turns a spec into tests ───────────────────────────────

def _test_blocks(body: str) -> str:
    """Salvage just the ``def test_...`` blocks from text the parser chokes on.

    Scans line by line: a line beginning ``def test`` opens a block, the lines
    indented under it belong to it, and the next unindented line closes it. This
    rescues a suite the model wrapped in a stray sentence or a half-finished line."""
    out, keeping = [], False
    for line in body.splitlines():
        if line.lstrip().startswith("def test"):
            keeping = True
        elif keeping and line.strip() and not line[:1].isspace():
            keeping = False
        if keeping:
            out.append(line)
    return "\n".join(out)


def _clean(text: str) -> str:
    """Strip a markdown fence, imports, and any redefinition of the function under
    test from a model's reply, leaving only its test_* functions to run."""
    import re
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    body = (fenced.group(1) if fenced else text).strip()
    try:
        tree = ast.parse(body)
    except SyntaxError:
        tree = ast.parse(_test_blocks(body))
    tree.body = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    return ast.unparse(tree)


def write_tests(spec: str = SPEC, model: str = CODING_MODEL) -> str:
    """Ask a code model for a quick first-pass pytest suite for ``letter_grade``.

    The function is presented as already defined, so the model writes only the
    tests, one typical score per grade. This is round 0 of the loop: a natural
    first attempt that tests the obvious cases and skips the boundaries. Returns
    the cleaned suite source (green-filtered by the caller)."""
    prompt = (
        "Write a quick first pass of unit tests for a function `letter_grade` that "
        "is already defined. One test function per grade letter (A, B, C, D, F), "
        "each asserting a single typical score from the middle of that letter's "
        "range. Plain pytest style: functions named test_..., `assert "
        "letter_grade(x) == 'Y'`. No imports, no comments, do not redefine "
        "letter_grade.\n\nSpec: " + spec)
    return _clean(_ask(prompt, model=model, system="", max_tokens=600))


def _cases(survivors: list, batch: int = 3) -> list:
    """The next batch of distinct missing cases, as ``letter_grade(x) == 'y'``
    strings. Deduplicated by input and handed over a few at a time, so the suite is
    strengthened in steps and the mutation score climbs gradually."""
    seen, cases = set(), []
    for s in survivors:
        w = s["witness"]
        if w is not None and w not in seen:
            seen.add(w)
            cases.append(f"letter_grade({w}) == '{s['want']}'")
        if len(cases) >= batch:
            break
    return cases


def add_tests(cases: list, model: str = CODING_MODEL) -> str:
    """Ask the model for tests aimed at specific cases, the writer/mutator feedback.

    Unlike ``write_tests``, this carries no 'typical score' framing: it names the
    exact boundary cases the surviving mutants exposed and asks for one test each,
    so the new tests actually target the holes instead of re-checking easy values."""
    prompt = (
        "A function `letter_grade` is already defined. Write pytest test functions, "
        "one per case, named test_..., each a single plain `assert`, no imports, no "
        "comments. Check exactly these cases:\n" + "; ".join(cases))
    return _clean(_ask(prompt, model=model, system="", max_tokens=400))


def _dedupe(suite: str) -> str:
    """Rename every test function to a unique ``test_1``, ``test_2`` ... so merging
    a round's new tests onto the suite can't collide names (which would miscount
    tests or shadow a real check). Used only on the internal merged suites; the
    displayed round-0 suite keeps the model's own names."""
    tree = ast.parse(suite)
    i = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            i += 1
            node.name = f"test_{i}"
    return ast.unparse(tree)


def write_characterization_suite(model: str = CODING_MODEL) -> str:
    """Ask the model to lock in whatever ``letter_grade`` *currently* returns.

    This is golden-master / characterization testing: instead of a spec, the model
    is handed the function's own outputs (taken from the buggy build) and told to
    assert them. The result is a suite that's all green by construction and pins
    the wrong contract, the §4 trap. Witnessed at the boundary scores so 90's bad
    grade is in the table."""
    ns = {}
    exec(BUGGY, ns)        # noqa: S102 - controlled local demo
    rows = "; ".join(f"letter_grade({x}) == '{ns['letter_grade'](x)}'"
                     for x in (95, 90, 85, 80, 75, 65, 30))
    prompt = ("Write a pytest suite that locks in the current behavior of a "
              "function `letter_grade` that is already defined. Assert exactly "
              "these observed outputs, one per test_ function, plain asserts, no "
              "imports, no comments:\n" + rows)
    return _clean(_ask(prompt, model=model, system="", max_tokens=400))


def strengthen(rounds: int = 5, batch: int = 3, seed: str = None,
               model: str = CODING_MODEL) -> dict:
    """The writer/mutator loop. Round 0 is the model's first suite (pass ``seed`` to
    reuse the one already shown in the chapter); each later round feeds a batch of
    surviving mutants back, the model adds tests aimed at them, and the mutation
    score is recomputed. Returns ``{rounds: [...], suites: [...]}`` with one record
    per round (n_tests, killed, total, score, survivors)."""
    suite = green_suite(seed if seed is not None else write_tests(model=model))
    records, suites = [], []
    for r in range(rounds + 1):
        score = mutation_score(suite)
        records.append({"round": r, "n_tests": n_tests(suite), **score})
        suites.append(suite)
        if not score["survivors"]:
            break
        added = add_tests(_cases(score["survivors"], batch), model=model)
        suite = green_suite(_dedupe(suite + "\n" + added))
    return {"rounds": records, "suites": suites}


# ── Captured constants (baked by scripts/_test_mutation_probe.py) ─────────────
# Filled in after a real run; see the probe. Until then these are placeholders so
# the module imports; the probe prints replacements to paste here.

# Captured by scripts/_test_mutation_probe.py (qwen2.5-coder + the built-in
# mutator). ROUND0_SUITE is the model's real quick first pass; it tests a typical
# score per grade and never a boundary, so it leaves the planted bug (and every
# mutant) uncaught. TEST_STUDY is the mutation score per round of strengthen()
# seeded with that suite. LYING_SUITE is the model's real characterization suite,
# which records the buggy outputs (90 graded a B) and so passes on the buggy code
# but rejects the fix.
ROUND0_SUITE = (
    "def test_grade_A():\n    assert letter_grade(95) == 'A'\n"
    "def test_grade_B():\n    assert letter_grade(85) == 'B'\n"
    "def test_grade_C():\n    assert letter_grade(75) == 'C'\n"
    "def test_grade_D():\n    assert letter_grade(65) == 'D'\n"
    "def test_grade_F():\n    assert letter_grade(55) == 'F'")
BUG_CAUGHT = {"n_tests": 5, "failed": [], "caught": False}
TEST_STUDY = {"rounds": [
    {"round": 0, "n_tests": 5, "killed": 0, "total": 12, "score": 0.0},
    {"round": 1, "n_tests": 8, "killed": 6, "total": 12, "score": 0.5},
    {"round": 2, "n_tests": 11, "killed": 10, "total": 12, "score": 0.833},
    {"round": 3, "n_tests": 13, "killed": 12, "total": 12, "score": 1.0}]}
LYING_SUITE = (
    "def test_letter_grade_A():\n    assert letter_grade(95) == 'A'\n"
    "def test_letter_grade_B_high():\n    assert letter_grade(90) == 'B'\n"
    "def test_letter_grade_B_low():\n    assert letter_grade(85) == 'B'\n"
    "def test_letter_grade_C():\n    assert letter_grade(80) == 'B'\n"
    "def test_letter_grade_D():\n    assert letter_grade(75) == 'C'\n"
    "def test_letter_grade_F():\n    assert letter_grade(65) == 'D'\n"
    "def test_letter_grade_F_low():\n    assert letter_grade(30) == 'F'")


# ── Captured constants (baked by scripts/_test_property_probe.py) ────────────
# qwen2.5-coder's real property batch, verbatim: asked for 12, it wrote 11, and
# four of them are word-for-word repeats of earlier ones under new names. The
# first property's conditional expression is a real blunder it emitted: when
# score < 90 the assert checks only that the returned letter is truthy, so that
# half of the property can never fail. PROPERTY_STUDY holds the deterministic
# grades: per-property mutant kills, the vacuous count, the union, which
# properties catch the rewrite's planted 100 slip, and the kill count of the
# chapter's own monotonicity property (never_hurt_kills).
PROPERTIES_SUITE = '''\
@given(st.integers(min_value=0, max_value=100))
def prop_letter_grade_90_and_above(score):
    assert letter_grade(score) == 'A' if score >= 90 else letter_grade(score)

@given(st.integers(min_value=80, max_value=89))
def prop_letter_grade_80_to_89(score):
    assert letter_grade(score) == 'B'

@given(st.integers(min_value=70, max_value=79))
def prop_letter_grade_70_to_79(score):
    assert letter_grade(score) == 'C'

@given(st.integers(min_value=60, max_value=69))
def prop_letter_grade_60_to_69(score):
    assert letter_grade(score) == 'D'

@given(st.integers(min_value=0, max_value=59))
def prop_letter_grade_below_60(score):
    assert letter_grade(score) == 'F'

@given(st.integers(min_value=90, max_value=100))
def prop_letter_grade_max_score(score):
    assert letter_grade(score) == 'A'

@given(st.integers(min_value=80, max_value=89))
def prop_letter_grade_min_80_score(score):
    assert letter_grade(score) == 'B'

@given(st.integers(min_value=70, max_value=79))
def prop_letter_grade_min_70_score(score):
    assert letter_grade(score) == 'C'

@given(st.integers(min_value=60, max_value=69))
def prop_letter_grade_min_60_score(score):
    assert letter_grade(score) == 'D'

@given(st.integers(min_value=0, max_value=59))
def prop_letter_grade_min_0_score(score):
    assert letter_grade(score) == 'F'

@given(st.integers(min_value=100, max_value=100))
def prop_letter_grade_exactly_100_score(score):
    assert letter_grade(score) == 'A'
'''

PROPERTY_STUDY = {
    "asked": 12, "written": 11, "duplicates": 4,
    "graded": [
        {"name": "prop_letter_grade_90_and_above", "kills": 2, "total": 12},
        {"name": "prop_letter_grade_80_to_89", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_70_to_79", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_60_to_69", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_below_60", "kills": 1, "total": 12},
        {"name": "prop_letter_grade_max_score", "kills": 2, "total": 12},
        {"name": "prop_letter_grade_min_80_score", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_min_70_score", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_min_60_score", "kills": 3, "total": 12},
        {"name": "prop_letter_grade_min_0_score", "kills": 1, "total": 12},
        {"name": "prop_letter_grade_exactly_100_score", "kills": 0, "total": 12},
    ],
    "wrong": [], "n_vacuous": 1, "union_killed": 12, "total": 12,
    "catch_rewrite": ["prop_letter_grade_90_and_above",
                      "prop_letter_grade_max_score",
                      "prop_letter_grade_exactly_100_score"],
    "never_hurt_kills": 0,
}


def _prop_by_name(name: str, suite: str = None) -> str:
    """One property from the batch, verbatim, for showing an exemplar."""
    for node in ast.parse(suite or PROPERTIES_SUITE).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return ""


def show_property_zoo() -> None:
    """The model writing properties: the assignment, then two of its real
    eleven, verbatim: the clean B-band rule, and the one-score 'property' that
    quantifies over nothing. Rendered with show_code so indentation survives."""
    from genai.agent import show_turn, show_code
    show_turn("you", "Write 12 property-based tests for letter_grade: "
                     "functions named prop_..., each @given whole-number "
                     "scores from st.integers within 0 to 100, plain asserts. "
                     + SPEC)
    show_code("qwen2.5-coder",
              _prop_by_name("prop_letter_grade_80_to_89") + "\n" +
              _prop_by_name("prop_letter_grade_exactly_100_score"))
    show_turn("BATCH", f"{PROPERTY_STUDY['written']} properties returned "
                       f"(asked for {PROPERTY_STUDY['asked']}); "
                       f"{PROPERTY_STUDY['duplicates']} are word-for-word "
                       "repeats under new names")


# ── Property-based testing (the Rule, Not the Cases section) ─────────────────
# An example test pins one input to one answer; a property states a rule over
# the whole domain and lets Hypothesis hunt for an input that breaks it. The
# demo world moves one release forward: the 90 bug is fixed, the grade ladder
# is rewritten as explicit bands, and the rewrite plants a new slip at the top
# of the range (a perfect 100 matches no band and falls through to 'F') that
# the strongest example suite in the chapter never looks at.

REFACTORED = (
    "def letter_grade(score):\n"
    "    if 90 <= score < 100: return 'A'\n"
    "    if 80 <= score < 90: return 'B'\n"
    "    if 70 <= score < 80: return 'C'\n"
    "    if 60 <= score < 70: return 'D'\n"
    "    return 'F'\n")

# The rewrite as a callable, for properties to import. One exec at import so
# the callable and the displayed source can never drift apart.
_ns = {}
exec(REFACTORED, _ns)          # noqa: S102 - controlled local demo
letter_grade_v2 = _ns["letter_grade"]

# Grade order, so "a higher score never earns a worse letter" is one comparison.
RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}


def _hypothesis_setup() -> None:
    """Make every Hypothesis run in the book deterministic and self-contained.

    ``derandomize=True`` derives the example stream from the property itself, so
    a cell prints the same falsifying example on every execution (no freeze tag
    needed); ``database=None`` plus a temp home dir keep Hypothesis from writing
    a ``.hypothesis/`` cache folder into the chapter directory; ``deadline=None``
    stops a busy machine from failing a property on wall-clock time."""
    import tempfile
    from pathlib import Path
    from hypothesis import settings
    from hypothesis.configuration import set_hypothesis_home_dir
    set_hypothesis_home_dir(Path(tempfile.gettempdir()) / "hypothesis-book")
    settings.register_profile("book", derandomize=True, database=None,
                              deadline=None)
    settings.load_profile("book")


try:
    _hypothesis_setup()
except ImportError:            # keep genai.test importable without hypothesis
    pass


def strong_suite() -> str:
    """The strongest example suite this chapter knows how to build: round 0 plus
    one test pinning every witness the mutator names, the same 12/12 strength the
    writer/mutator loop reached. Deterministic (no model call), so the section
    after the loop can rebuild it live."""
    seen, extra = set(), []
    for _label, mutant in make_mutants(CORRECT):
        w = _witness(mutant)
        if w["witness"] is not None and w["witness"] not in seen:
            seen.add(w["witness"])
            extra.append(f"def test_boundary_{w['witness']}():\n"
                         f"    assert letter_grade({w['witness']}) == '{w['want']}'")
    return ROUND0_SUITE + "\n" + "\n".join(extra)


def show_strong_green() -> None:
    """The setup for the property section, all computed live: the witness-pinned
    suite scores 12/12 on the mutants, then passes the rewritten build clean."""
    from genai.agent import show_turn
    suite = strong_suite()
    s = mutation_score(suite)
    fails = run_suite(suite, REFACTORED)
    show_turn("SUITE", f"round 0 plus one test per witness: {n_tests(suite)} tests")
    show_turn("MUTATOR", f"{s['killed']}/{s['total']} killed "
                         f"({s['score'] * 100:.0f}% mutation score)")
    show_turn("RUN", f"{n_tests(suite)} tests against the rewritten build: "
                     + (f"{len(fails)} FAIL" if fails else "all green"))


def show_falsified(prop) -> None:
    """Run a Hypothesis property against the rewrite and show the verdict.

    The HYPOTHESIS block is the tool's real report, verbatim: by the time it
    prints, the failing example has already been shrunk to the smallest pair
    that still fails. The CHECK line then grades both scores so the reader sees
    the inversion with their own eyes."""
    import re
    from genai.agent import show_turn, show_code
    try:
        prop()
    except AssertionError as exc:
        note = "\n".join(getattr(exc, "__notes__", [])) or str(exc)
        show_code("HYPOTHESIS", note)
        pair = [int(m) for m in re.findall(r"=(\d+)", note)]
        if len(pair) == 2:
            lo, hi = pair
            show_turn("CHECK", f"letter_grade_v2({lo}) = '{letter_grade_v2(lo)}'\n"
                               f"letter_grade_v2({hi}) = '{letter_grade_v2(hi)}'")
            show_turn("VERDICT", f"a {lo} earns a better letter than a {hi}: "
                                 "more points hurt")
    else:
        show_turn("HYPOTHESIS", "no counterexample found: the property held on "
                                "every example tried")


# ── The model writes properties; the mutator grades them ─────────────────────

def _clean_props(text: str) -> str:
    """Like ``_clean``, but keep ``prop_*`` functions with their ``@given``
    decorators intact, since the decorator is where the property's input range
    lives."""
    import re
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    body = (fenced.group(1) if fenced else text).strip()
    try:
        tree = ast.parse(body)
    except SyntaxError:
        tree = ast.parse(_test_blocks(body))
    tree.body = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("prop")]
    return ast.unparse(tree)


def write_properties(spec: str = SPEC, k: int = 12,
                     model: str = CODING_MODEL) -> str:
    """Ask the chapter's test writer for ``k`` Hypothesis properties.

    Same rules as ``write_tests``: the model sees the spec, never the code, and
    the prompt constrains only the mechanics (names, decorator, integer scores),
    not which rules to state. What it chooses to assert is the experiment."""
    prompt = (
        f"A function `letter_grade` is already defined, and Hypothesis's `given` "
        f"and `st` (strategies) are already imported. Write {k} property-based "
        f"tests for it: functions named prop_..., each decorated with @given "
        f"drawing whole-number scores from st.integers (within 0 to 100), each "
        f"body plain asserts about letter_grade's output. No imports, no "
        f"comments, no example tests, do not redefine letter_grade.\n\n"
        f"Spec: {spec}")
    return _clean_props(_ask(prompt, model=model, system="", max_tokens=1600))


def _prop_fns(props_src: str, func_source: str) -> list:
    """Exec ``func_source`` and then the properties in one namespace (with the
    Hypothesis names they need already bound), so each property's global
    ``letter_grade`` is the version under test. Returns ``(name, fn)`` pairs."""
    from hypothesis import assume, given, settings, strategies as st
    ns = {"given": given, "st": st, "strategies": st, "settings": settings,
          "assume": assume}
    exec(func_source, ns)      # noqa: S102 - controlled local demo
    exec(props_src, ns)        # noqa: S102 - controlled local demo
    return [(k, v) for k, v in ns.items()
            if k.startswith("prop") and callable(v)]


def _prop_fails(name: str, props_src: str, func_source: str) -> bool:
    """True if the named property raises (any exception) on ``func_source``."""
    for n, fn in _prop_fns(props_src, func_source):
        if n == name:
            return _raises(fn)
    return True


def grade_properties(props_src: str) -> dict:
    """Grade each property with the chapter's own mutator.

    A property that fails on the correct code is *wrong* (it asserts something
    the spec never promised) and is set aside, the same green-on-correct rule
    the example suites played by. Every remaining property is run against all
    12 mutants; a raise is a kill. ``vacuous`` counts the properties that kill
    nothing at all. Returns ``{graded, wrong, n_vacuous, union_killed, total}``
    where graded is ``[{name, kills, total}, ...]``."""
    names = [n for n, _fn in _prop_fns(props_src, CORRECT)]
    wrong = [n for n in names if _prop_fails(n, props_src, CORRECT)]
    mutants = make_mutants(CORRECT)
    graded, union = [], set()
    for name in names:
        if name in wrong:
            continue
        kills = {i for i, (_label, mutant) in enumerate(mutants)
                 if _prop_fails(name, props_src, mutant)}
        union |= kills
        graded.append({"name": name, "kills": len(kills), "total": len(mutants)})
    return {"graded": graded, "wrong": wrong,
            "n_vacuous": sum(g["kills"] == 0 for g in graded),
            "union_killed": len(union), "total": len(mutants)}


def _fmt(suite: str) -> str:
    """Render a suite for the page: standard ``def`` / indented ``assert`` form
    with the blank lines ``ast.unparse`` inserts between functions stripped out, so
    a real suite reads compactly. The indented assert body matters: show_code's
    reflow pass keeps an indented line verbatim, where uniform unindented one-liners
    would get mistaken for wrapped prose and flattened together."""
    lines = ast.unparse(ast.parse(suite)).split("\n")
    return "\n".join(ln for ln in lines if ln.strip())


def show_writes_tests(suite: str = None) -> None:
    """The model writing tests: the spec it's handed, then the real suite it wrote.
    Rendered with show_code so the asserts keep their shape."""
    from genai.agent import show_turn, show_code
    suite = suite or ROUND0_SUITE
    show_turn("you", "Write a quick first pass of unit tests for letter_grade. " + SPEC)
    show_code("qwen2.5-coder", _fmt(suite))


def show_bug_caught(suite: str = None) -> None:
    """Run the model's suite against the buggy code and show which test fired. The
    TEST lines are the real failures run_suite reports, not narration."""
    from genai.agent import show_turn
    suite = suite or ROUND0_SUITE
    fails = run_suite(suite, BUGGY)
    show_turn("RUN", f"{n_tests(suite)} tests against the shipped (buggy) code")
    for name, reason in fails:
        show_turn("TEST", f"{name}  ->  FAIL ({reason})")
    if fails:
        show_turn("VERDICT", "the planted bug (90 graded a B) is caught")
    else:
        show_turn("VERDICT", "all green: the bug slipped through untested")


def show_mutation_score(suite: str = None) -> None:
    """Score the suite with the mutator: how many of the small code edits it kills,
    and a couple of survivors with the case each one points at."""
    from genai.agent import show_turn
    suite = suite or ROUND0_SUITE
    s = mutation_score(suite)
    show_turn("MUTATOR", f"{s['total']} single-edit mutants of the correct code")
    show_turn("SCORE", f"{s['killed']}/{s['total']} killed  "
                       f"({s['score'] * 100:.0f}% mutation score)")
    for sv in s["survivors"][:3]:
        show_turn("SURVIVOR", f"{sv['label']:<9} slips past: "
                              f"letter_grade({sv['witness']}) returns "
                              f"'{sv['got']}', should be '{sv['want']}'")


def show_lying_suite(suite: str = None) -> None:
    """The honest failure: a suite that's all green yet asserts the wrong contract.
    It passes on the buggy code it was written against and rejects the correct fix,
    so its green bar locks the bug in. Numbers are run live against both versions."""
    from genai.agent import show_turn, show_code
    suite = suite or LYING_SUITE
    show_code("qwen2.5-coder", _fmt(suite))
    on_buggy = run_suite(suite, BUGGY)
    on_correct = run_suite(suite, CORRECT)
    show_turn("RUN", f"against the shipped code: {n_tests(suite) - len(on_buggy)}"
                     f"/{n_tests(suite)} pass  (all green)")
    show_turn("RUN", f"against the corrected code: {len(on_correct)} now FAIL")
    show_turn("VERDICT", "the suite asserts 90 is a B; it rejects the fix and "
                         "locks the bug in")


# ── The test that can't fail ──────────────────────────────────────────────────
# A green bar counts tests that passed; it can't tell whether a test could ever
# have failed. Asked for tests, a good code model writes real ones (it insists on
# `assert letter_grade(90) == 'A'` no matter how the request is framed). But
# vacuous tests still pile up in real suites whenever a team chases a coverage
# number: assertions loose enough that no bug can trip them. VACUOUS_SUITE is that
# kind, hand-written to show the shape. It's all green, and it stays green on the
# shipped bug, because none of its assertions pins a grade. STRONG_MINI is smaller
# and does. Everything here is deterministic (the mutator, not a model), so the
# cell runs live.

VACUOUS_SUITE = (
    "def test_a(): assert letter_grade(95) is not None\n"
    "def test_b(): assert isinstance(letter_grade(85), str)\n"
    "def test_c(): assert letter_grade(72) in 'ABCDF'\n"
    "def test_d(): assert len(letter_grade(64)) == 1\n"
    "def test_e(): assert letter_grade(30) in 'ABCDF'\n"
    "def test_f(): assert isinstance(letter_grade(90), str)\n")

STRONG_MINI = (
    "def test_a(): assert letter_grade(90) == 'A'\n"
    "def test_b(): assert letter_grade(89) == 'B'\n"
    "def test_c(): assert letter_grade(70) == 'C'\n"
    "def test_d(): assert letter_grade(60) == 'D'\n"
    "def test_e(): assert letter_grade(59) == 'F'\n")


def show_cant_fail() -> None:
    """A padded suite that can't fail, and what mutation testing sees that a green
    bar can't. The vacuous suite passes on the correct code and on the shipped
    bug alike and kills no mutants; a smaller suite of exact assertions catches
    the bug and kills most."""
    from genai.agent import show_code, show_turn
    show_code("padded suite", VACUOUS_SUITE.rstrip())
    n = n_tests(VACUOUS_SUITE)
    show_turn("GREEN", f"{n} tests, all pass on the correct letter_grade")
    on_bug = run_suite(VACUOUS_SUITE, BUGGY)
    show_turn("BUGGY", f"run against the shipped bug (a 90 graded 'B'): "
              f"{len(on_bug)} of {n} fail, still all green")
    vac = mutation_score(VACUOUS_SUITE)
    show_turn("MUTATION", f"kills {vac['killed']} of {vac['total']} mutants: it "
              "pins nothing down")
    strong = mutation_score(STRONG_MINI)
    strong_bug = run_suite(STRONG_MINI, BUGGY)
    show_turn("COMPARE", f"{n_tests(STRONG_MINI)} exact ==-tests instead: "
              f"{len(strong_bug)} fails on the bug, and they kill "
              f"{strong['killed']} of {strong['total']}")

"""Standing over a failing test: localize the fault, patch it, and measure the overfit.

The Testing chapter plants a bug and detects it; Verification proves a fix correct.
This chapter lives in the space between them. A test has just gone red, and nobody has
shown the model standing over the traceback, trying to find the cause and repair it.
We hand a code model only the broken function, one failing test, and its traceback, and
run a repair loop: localize the fault, propose a patch, apply it, re-run the suite,
repeat until green. The oracle is the test suite going from red to green.

The honest failure is the one automated program repair has always had: a patch that
passes the test it was shown and is still wrong. The target is ``bulk_discount``, whose
middle pricing tier lives ONLY in the tests; the buggy build dropped it. A single failing
test pins one quantity, not the tier's edge, so the model's patch satisfies that test and
guesses the rest. A *held-out* test the model never sees catches the guess. We measure how
often the guess is wrong, and how much a second test that pins the edge closes the distance.

The model is qwen2.5-coder (a code model repairs code more cleanly than a reasoning model
narrates it). The grader is pure Python and deterministic; only the model is not, so the
captured patch and the study numbers are baked once by ``scripts/_debugging_probe.py`` and
the cells that show them are frozen.

A later section steps one question earlier: WHICH commit broke it. ``build_history``
constructs a deterministic fourteen-commit history of the pricing module that ends in
the exact BUGGY build, with a guilty-sounding decoy and an innocently-named culprit.
qwen2.5-coder names a suspect from the log and diffs (DETECTIVE_STUDY, baked by
``scripts/_bisect_probe.py``), while ``git bisect run`` finds the culprit mechanically
with the same failing test as the oracle.
"""
import ast
import re
import traceback as _tb

from genai.llm import CODING_MODEL, ask as _ask

# ── The repair target ─────────────────────────────────────────────────────────
# bulk_discount maps an order quantity to a percent discount through three tiers.
# The tier boundaries (20, 120, 500) are a business rule, not common knowledge:
# nothing in the wider world tells you the 15% tier starts at 120, so the only
# place that fact is written down is the tests. CORRECT obeys the rule; BUGGY
# shipped with the middle tier dropped, so every quantity from 120 to 499 quietly
# collects 5% instead of 15%. The model is handed BUGGY and has to put it right.
CORRECT = (
    "def bulk_discount(qty):\n"
    "    if qty >= 500: return 25\n"
    "    if qty >= 120: return 15\n"
    "    if qty >= 20:  return 5\n"
    "    return 0\n")
BUGGY = (
    "def bulk_discount(qty):\n"
    "    if qty >= 500: return 25\n"
    "    if qty >= 20:  return 5\n"
    "    return 0\n")
FUNC = "bulk_discount"

# Tests are (quantity, expected discount). The visible suite is what the repair is
# graded against; the held-out tests are the silent judge the model never sees.
# 150 sits deep inside the dropped tier, far below the example the model is shown,
# and 110 sits just under the tier's real edge, so a patch that merely satisfies
# the shown example misses both.
VISIBLE = [(300, 15)]
HELD_OUT = [(150, 15), (110, 5)]


def _call(src: str, x: int):
    ns = {}
    exec(src, ns)                       # noqa: S102 - controlled local demo
    return ns[FUNC](x)


def gold(x: int) -> int:
    """The true discount for ``x``, read off CORRECT and never hand-typed."""
    return _call(CORRECT, x)


# ── Tests, tracebacks, and running a patch ────────────────────────────────────

def make_suite(cases: list) -> str:
    """Render ``(qty, want)`` cases as a plain pytest suite the model can read."""
    return "\n".join(
        f"def test_qty_{x}():\n"
        f"    got = {FUNC}({x})\n"
        f"    assert got == {w}, f'{FUNC}({x}) returned {{got}}, expected {w}'\n"
        for x, w in cases)


def first_traceback(buggy: str, cases: list) -> str:
    """Run the suite against ``buggy`` and capture the first real Python traceback.

    This is the actual exception text a developer would see, not a description of
    it: an ``AssertionError`` with the value the function returned and the value
    the test expected.
    """
    ns = {}
    exec(buggy, ns)                     # noqa: S102 - controlled local demo
    exec(make_suite(cases), ns)         # noqa: S102 - controlled local demo
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError:
                return _tb.format_exc()
    return ""


def run_suite(src: str, cases: list) -> list:
    """Return the failing ``(qty, want, got)`` rows of ``cases`` against ``src``.

    An empty list means every case passed (the suite is green on ``src``).
    """
    fails = []
    for x, w in cases:
        try:
            got = _call(src, x)
        except Exception as exc:        # a patch that doesn't even run
            got = f"<{type(exc).__name__}>"
        if got != w:
            fails.append((x, w, got))
    return fails


# ── The model: localize and patch ─────────────────────────────────────────────

def _extract(text: str) -> str:
    """Pull the corrected ``bulk_discount`` out of a model reply.

    The model sometimes brackets its diagnosis in its own code fence, so we scan
    every fenced (or bare) block and keep the last one that parses and actually
    defines the function. Returns ``None`` if nothing usable is there.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S) or [text]
    found = None
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == FUNC:
                found = ast.unparse(node)
    return found


def _diagnosis(text: str) -> str:
    """The model's one-line fault localization, the ``BUG:`` line it leads with."""
    m = re.search(r"BUG:\s*(.+)", text)
    return m.group(1).strip() if m else ""


_REPAIR_PROMPT = (
    "A unit test fails on this function:\n\n```python\n{buggy}```\n\n"
    "Here is the failing test and its traceback:\n\n```python\n{suite}```\n\n"
    "```\n{trace}```\n\n"
    "Find and fix the bug so the test passes. Respond with one line starting "
    "`BUG:` that names where the fault is, then the complete corrected function "
    "in a single ```python code block.")


def propose_patch(buggy: str, cases: list, model: str = CODING_MODEL) -> dict:
    """Ask the model to localize and repair ``buggy`` against ``cases``.

    Returns ``{diagnosis, patch}``: the model's one-line localization and its
    corrected function source (``patch`` is ``None`` if no function could be
    parsed out of the reply).
    """
    prompt = _REPAIR_PROMPT.format(buggy=buggy, suite=make_suite(cases),
                                   trace=first_traceback(buggy, cases))
    reply = _ask(prompt, model=model, system="", max_tokens=400)
    return {"diagnosis": _diagnosis(reply), "patch": _extract(reply)}


def repair_loop(cases: list = VISIBLE, model: str = CODING_MODEL,
                max_iter: int = 3) -> dict:
    """Localize, patch, re-run, repeat until the visible suite is green.

    Returns ``{diagnosis, patch, rounds, green}``. ``rounds`` is how many patches
    it took; ``green`` says whether the visible suite passes at the end. For the
    tiny target here one round is usually enough, but the loop is the shape that
    scales: re-feed whatever is still failing and ask again.
    """
    src, first_diagnosis = BUGGY, ""
    for r in range(1, max_iter + 1):
        step = propose_patch(src, cases, model=model)
        if r == 1:
            first_diagnosis = step["diagnosis"]
        if step["patch"] is None:
            return {"diagnosis": first_diagnosis, "patch": None,
                    "rounds": r, "green": False}
        src = step["patch"]
        if not run_suite(src, cases):
            return {"diagnosis": first_diagnosis, "patch": src,
                    "rounds": r, "green": True}
    return {"diagnosis": first_diagnosis, "patch": src,
            "rounds": max_iter, "green": not run_suite(src, cases)}


# ── The study: how often does a green patch fail the held-out test? ────────────

def classify(patch: str, cases: list) -> str:
    """Bucket a patch as ``failed`` (not even green), ``overfit`` (green but the
    held-out test fails), or ``correct`` (green and the held-out test passes)."""
    if patch is None or run_suite(patch, cases):
        return "failed"
    return "overfit" if run_suite(patch, HELD_OUT) else "correct"


def overfit_study(conditions: dict, trials: int = 12,
                  model: str = CODING_MODEL) -> dict:
    """Repair ``trials`` times under each visible-test condition and tally outcomes.

    ``conditions`` maps a label to its visible-test list. Returns, per label,
    ``{failed, overfit, correct, n}``. The headline is the overfit count: patches
    that went green on what the model saw and stayed wrong on what it didn't.
    """
    out = {}
    for label, cases in conditions.items():
        tally = {"failed": 0, "overfit": 0, "correct": 0}
        for _ in range(trials):
            tally[classify(propose_patch(BUGGY, cases, model=model)["patch"],
                           cases)] += 1
        out[label] = {**tally, "n": trials}
    return out


# The two conditions the study runs against the SAME held-out judge. The first is
# the realistic one: a single failing test. The second adds two tests that pin the
# tier's lower edge (119 still gets 5%, 120 gets 15%), which rules out the cheap
# "widen the tier" patch and forces the boundary to the right place.
CONDITIONS = {
    "the failing test alone": [(300, 15)],
    "tests that pin the edge": [(300, 15), (119, 5), (120, 15)],
}


# ── Captured constants (baked by scripts/_debugging_probe.py) ─────────────────
# Filled in after a real run; placeholders so the module imports. The probe prints
# replacements to paste here. DEMO is one real repair trial under the single
# failing test: the model's verbatim diagnosis and patch, which goes green on the
# shown test and fails the held-out one. REPAIR_STUDY is overfit_study(CONDITIONS).
DEMO = {
    "diagnosis": "The conditions for the discounts are not correctly ordered.",
    "patch": ("def bulk_discount(qty):\n"
              "    if qty >= 500:\n"
              "        return 25\n"
              "    if qty >= 20 and qty < 500:\n"
              "        return 15\n"
              "    return 0"),
}
REPAIR_STUDY = {
    "the failing test alone": {"failed": 7, "overfit": 5, "correct": 0, "n": 12},
    "tests that pin the edge": {"failed": 10, "overfit": 0, "correct": 2, "n": 12},
}


# ── Display helpers ───────────────────────────────────────────────────────────
# Event rows (RUN / TEST / HELD-OUT / VERDICT) are non-speaker labels carrying
# real computed results; the speaker row (qwen2.5-coder) carries the model's real
# diagnosis and patch. Nothing here narrates the model in its own voice.

def show_failing_test(cases: list = VISIBLE) -> None:
    """The starting point: the shipped function, the failing test, and the real
    traceback the model is handed. Nothing else: no spec, no held-out test."""
    from genai.agent import show_code, show_turn
    show_code("function", BUGGY.rstrip())
    show_code("test", make_suite(cases).rstrip())
    for x, w, got in run_suite(BUGGY, cases):
        show_turn("RUN", f"{FUNC}({x}) returned {got}, expected {w}   ->  FAIL")


def show_repair(demo: dict = None, cases: list = VISIBLE) -> None:
    """The model's real repair: its one-line localization, the patch it wrote, and
    the suite going green. The diagnosis and patch are the model's verbatim output;
    the RUN line is computed by re-running the suite against the patch."""
    from genai.agent import show_code, show_turn
    demo = demo or DEMO
    show_turn("qwen2.5-coder", "BUG: " + demo["diagnosis"])
    show_code("qwen2.5-coder", demo["patch"].rstrip())
    fails = run_suite(demo["patch"], cases)
    passed = len(cases) - len(fails)
    show_turn("RUN", f"the visible suite: {passed}/{len(cases)} pass   ->  green")


def show_overfit(demo: dict = None) -> None:
    """The catch: the same green patch meets the held-out tests it never saw. The
    HELD-OUT rows are the real failures run_suite reports against the patch."""
    from genai.agent import show_turn
    demo = demo or DEMO
    fails = run_suite(demo["patch"], HELD_OUT)
    show_turn("RUN", f"the held-out tests against the same patch: "
                     f"{len(HELD_OUT) - len(fails)}/{len(HELD_OUT)} pass")
    for x, w, got in fails:
        show_turn("HELD-OUT", f"{FUNC}({x}) returned {got}, expected {w}   ->  FAIL")
    if fails:
        show_turn("VERDICT", "the patch passed the test it was shown and is wrong "
                             "on the tier it wasn't")
    else:
        show_turn("VERDICT", "the patch holds on inputs it never saw")


# ── The fix that broke the neighbor: regression, not overfit ──────────────────
# The held-out tests caught a patch wrong on the tier it was never shown. A patch
# can fail the other direction too: break code that was already RIGHT. The model's
# fix widens the low tier to 15%, which greens bulk_discount(300) but now overcharges
# every low-tier order. order_total, a neighbor that prices an order through the same
# discount, had a test that was green on the shipped build and reddens on the "fix".
# The repair loop's oracle, the target's failing test going green, never looks at it.
# The cure is a regression rule: after every patch, re-run the whole existing suite
# and reject a patch that turns a green test red. Deterministic (the baked patch run
# against tests).

# A neighbor that prices an order through the same discount. On the shipped bug it is
# correct for the tier it touches, so its test is green before any repair is tried.
NEIGHBOR = ("def order_total(qty, unit):\n"
            "    subtotal = qty * unit\n"
            "    return subtotal * (1 - bulk_discount(qty) / 100)\n")

# Tests that pass on the shipped BUGGY build, the regression baseline a repair must
# not break: a low-tier discount the bug left right, and a neighbor order priced by it.
REGRESSION_TESTS = [("bulk_discount(50)", 5), ("order_total(50, 10)", 475.0)]


def _run_named(func_src: str, checks: list) -> list:
    """Run ``(call_expr, want)`` checks against ``func_src`` plus the NEIGHBOR; return
    the failing ``(expr, want, got)`` rows. An expression that raises is its own
    failure, so a patch that breaks a caller outright still counts as a regression."""
    ns = {}
    exec(func_src + "\n" + NEIGHBOR, ns)     # noqa: S102 - controlled local demo
    fails = []
    for expr, want in checks:
        try:
            got = eval(expr, dict(ns))       # noqa: S307 - the fixed REGRESSION_TESTS
        except Exception as exc:             # noqa: BLE001 - a broken caller is a fail
            got = f"<{type(exc).__name__}>"
        if got != want:
            fails.append((expr, want, got))
    return fails


def show_regression(demo: dict = None) -> None:
    """The patch that greens the target and breaks a bystander. The neighbor's test is
    green on the shipped build; the model's fix greens the failing target test but
    reddens the neighbor, a regression the target-only oracle can't see. All computed
    live from the baked patch."""
    from genai.agent import show_code, show_turn
    demo = demo or DEMO
    patch = demo["patch"]
    show_code("neighbor", NEIGHBOR.rstrip())
    before = _run_named(BUGGY, REGRESSION_TESTS)
    baseline = "; ".join(f"{e} == {w}" for e, w in REGRESSION_TESTS)
    show_turn("BEFORE", f"green on the shipped build: {baseline}"
              + (f"  ({len(before)} already red)" if before else ""))
    show_turn("PATCH", f"the model's fix greens the target: bulk_discount(300) -> "
              f"{_call(patch, 300)}, the failing test passes")
    for expr, want, got in _run_named(patch, REGRESSION_TESTS):
        show_turn("REGRESSED", f"{expr}: was {want}, now {got}  ->  a green test "
                  "went red")
    show_turn("RULE", "re-run the whole suite after a patch, and reject one that "
              "turns a green test red")


# ── The product: OpenHands owns the loop (captured by scripts/_openhands_probe.py) ─
# The raw loop above is ours; here a real autonomous-SWE product runs the same loop
# itself. OpenHands, driven headless through its own Agent/Conversation from the
# OpenHands SDK, gets the SAME BUGGY bulk_discount and the SAME single VISIBLE failing
# test, pointed at the SAME local model (ollama/qwen2.5-coder) through its LiteLLM
# config, on a LocalWorkspace (a local subprocess shell, no Docker). Local 7B models
# don't drive the product's native function-calling reliably, so this runs in its own
# remedy for local models, prompt-mocked tool-calling; that is the faithful config, not
# a workaround. OPENHANDS_TRANSCRIPT is a representative slice of one real run's events
# (its terminal commands, the file-editor's reply, the traceback it read). The model
# never opened pricing.py, tried to edit a line that wasn't there, and stalled with the
# file untouched, so OPENHANDS_PATCH is the file it left behind, unchanged. The held-out
# grading in show_openhands is recomputed live from that file (deterministic Python), so
# the transcript and the leftover file are the only baked constants, and the notebook
# never imports OpenHands, so the book builds from the bake alone.
OPENHANDS_VERSION = "1.16.0 (SDK 1.21.0)"
OPENHANDS_STEPS = 8
OPENHANDS_TRANSCRIPT = [
    ("TASK",   "pricing.py's bulk_discount fails one test; find and fix the bug, "
               "editing only pricing.py"),
    ("TOOL",   "terminal: python3 test_pricing.py"),
    ("RESULT", "AssertionError: bulk_discount(300) returned 5, expected 15"),
    ("TOOL",   "file_editor: str_replace on pricing.py"),
    ("RESULT", "no replacement performed: old_str `return qty * 0.95` did not appear "
               "verbatim in pricing.py"),
    ("TOOL",   "terminal: python3 test_pricing.py"),
    ("RESULT", "AssertionError: bulk_discount(300) returned 5, expected 15"),
]
# The file OpenHands left behind: identical to BUGGY, never edited.
OPENHANDS_PATCH = BUGGY.rstrip()


def show_openhands(transcript: list = None, patch: str = None) -> None:
    """OpenHands driving the localize->patch->re-run loop itself, against the same bug
    and the same model. TASK is the one instruction we hand it; the TOOL and RESULT rows
    are its real captured terminal commands, its file-editor's reply, and the traceback
    it read. The closing rows are recomputed live from the file it left behind, graded
    against the same held-out judge the raw loop faced, so the comparison is exact."""
    from genai.agent import show_code, show_turn
    transcript = transcript or OPENHANDS_TRANSCRIPT
    patch = patch or OPENHANDS_PATCH
    for label, text in transcript:
        show_turn(label, text)
    unchanged = patch.strip() == BUGGY.strip()
    vis_fail = run_suite(patch, VISIBLE)
    if vis_fail:                                    # the loop never went green
        state = "unchanged" if unchanged else "still failing"
        show_turn("FILE", f"pricing.py after {OPENHANDS_STEPS} actions: {state}; "
                          f"the visible test is red")
        show_turn("VERDICT", "the product's loop stalled before a working patch, so "
                             "the held-out test never got a turn")
        return
    show_code("OpenHands", patch.rstrip())          # it produced a green patch
    held_fail = run_suite(patch, HELD_OUT)
    show_turn("HELD-OUT", f"the held-out tests against its patch: "
                          f"{len(HELD_OUT) - len(held_fail)}/{len(HELD_OUT)} pass")
    for x, w, got in held_fail:
        show_turn("HELD-OUT", f"{FUNC}({x}) returned {got}, expected {w}   ->  FAIL")
    show_turn("VERDICT", "the product overfit exactly like the raw loop" if held_fail
              else "the product's patch holds on the tier the raw loop guessed wrong")


# ── When did it break? A history that ends in the bug ─────────────────────────
# The chapter starts at a red test, but production debugging often starts one
# question earlier: WHICH change broke it. The synthetic history below is the
# pricing module's last fourteen commits, ending in the exact BUGGY build the
# chapter repairs. Most commits are innocent. One, the DECOY, sounds guilty
# ("rework discount tiers") but provably preserves behavior; the CULPRIT hides
# behind an innocent message ("simplify conditionals") and silently drops the
# middle tier. build_history constructs the repo deterministically (fixed
# identities, fixed dates, fixed content), so the commit ids are stable across
# machines and re-runs and the bisect demo needs no freeze.

_P_DOC0 = '"""Pricing rules for the storefront."""\n'
_P_DOC1 = ('"""Pricing rules for the storefront.\n\n'
           'Discounts are percentages of the order subtotol; fees are flat.\n"""\n')
_P_DOC2 = ('"""Pricing rules for the storefront.\n\n'
           'Discounts are percentages of the order subtotol; fees are flat.\n\n'
           '    >>> order_total(10, 4.0)\n    47.0\n"""\n')
_P_DOC3 = _P_DOC2.replace("subtotol", "subtotal")
_P_TABLE = ("TIERS = ((500, 25), (120, 15), (20, 5))\n\n"
            "def bulk_discount(qty):\n"
            "    for threshold, pct in TIERS:\n"
            "        if qty >= threshold:\n"
            "            return pct\n"
            "    return 0\n")
_P_SHIP = "def flat_shipping(subtotal):\n    return 0 if subtotal >= 50 else 7\n"
_P_WRAP0 = "def gift_wrap_fee(items, amt=2):\n    return items * amt\n"
_P_WRAP1 = "def gift_wrap_fee(items, amount=2):\n    return items * amount\n"
_P_TOTAL0 = ("def order_total(qty, unit_price):\n"
             "    subtotal = qty * unit_price\n"
             "    subtotal -= subtotal * bulk_discount(qty) / 100\n"
             "    return subtotal + flat_shipping(subtotal)\n")
_P_TOTAL1 = ("def order_total(qty, unit_price):\n"
             '    """Subtotal for qty units, less discount, plus shipping."""\n'
             "    subtotal = qty * unit_price\n"
             "    subtotal -= subtotal * bulk_discount(qty) / 100\n"
             "    return subtotal + flat_shipping(subtotal)\n")
_P_POINTS = "def loyalty_points(subtotal):\n    return int(subtotal // 10)\n"
_P_ALL = ('__all__ = ["bulk_discount", "flat_shipping", "gift_wrap_fee",\n'
          '           "order_total", "loyalty_points"]\n')


def _pricing(doc: str, parts: list, sep: str = "\n") -> str:
    return doc + sep + sep.join(parts)


# (commit message, full pricing.py content) pairs, oldest first. The decoy is
# commit 6 and the culprit commit 10 (DECOY_IDX / CULPRIT_IDX below, 0-based).
PRICING_COMMITS = [
    ("init: pricing module with bulk_discount",
     _pricing(_P_DOC0, [CORRECT])),
    ("feat: add flat_shipping helper",
     _pricing(_P_DOC0, [CORRECT, _P_SHIP])),
    ("docs: expand module docstring",
     _pricing(_P_DOC1, [CORRECT, _P_SHIP])),
    ("feat: add gift_wrap_fee helper",
     _pricing(_P_DOC1, [CORRECT, _P_SHIP, _P_WRAP0])),
    ("style: two blank lines between defs",
     _pricing(_P_DOC1, [CORRECT, _P_SHIP, _P_WRAP0], sep="\n\n")),
    ("refactor: rework discount tiers into a table",           # the decoy
     _pricing(_P_DOC1, [_P_TABLE, _P_SHIP, _P_WRAP0], sep="\n\n")),
    ("feat: add order_total",
     _pricing(_P_DOC1, [_P_TABLE, _P_SHIP, _P_WRAP0, _P_TOTAL0], sep="\n\n")),
    ("docs: document order_total params",
     _pricing(_P_DOC1, [_P_TABLE, _P_SHIP, _P_WRAP0, _P_TOTAL1], sep="\n\n")),
    ("style: rename amt to amount in gift_wrap_fee",
     _pricing(_P_DOC1, [_P_TABLE, _P_SHIP, _P_WRAP1, _P_TOTAL1], sep="\n\n")),
    ("cleanup: simplify conditionals in pricing",              # the culprit
     _pricing(_P_DOC1, [BUGGY, _P_SHIP, _P_WRAP1, _P_TOTAL1], sep="\n\n")),
    ("feat: add loyalty_points helper",
     _pricing(_P_DOC1, [BUGGY, _P_SHIP, _P_WRAP1, _P_TOTAL1, _P_POINTS],
              sep="\n\n")),
    ("docs: usage example in module docstring",
     _pricing(_P_DOC2, [BUGGY, _P_SHIP, _P_WRAP1, _P_TOTAL1, _P_POINTS],
              sep="\n\n")),
    ("chore: add __all__ export list",
     _pricing(_P_DOC2, [_P_ALL, BUGGY, _P_SHIP, _P_WRAP1, _P_TOTAL1, _P_POINTS],
              sep="\n\n")),
    ("docs: fix typo in module docstring",
     _pricing(_P_DOC3, [_P_ALL, BUGGY, _P_SHIP, _P_WRAP1, _P_TOTAL1, _P_POINTS],
              sep="\n\n")),
]
DECOY_IDX, CULPRIT_IDX = 5, 9


def _git(path, *args, dates: str = None, check: bool = True):
    """Run git against the repo at ``path`` with a fully pinned environment: no
    user or system config, a fixed identity, and (for commits) a fixed date, so
    the same content always hashes to the same commit id."""
    import os
    import subprocess
    env = {**os.environ,
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "Priya Nair", "GIT_AUTHOR_EMAIL": "priya@example.com",
           "GIT_COMMITTER_NAME": "Priya Nair",
           "GIT_COMMITTER_EMAIL": "priya@example.com"}
    if dates:
        env.update({"GIT_AUTHOR_DATE": dates, "GIT_COMMITTER_DATE": dates})
    return subprocess.run(["git", "-C", str(path), *args], env=env, check=check,
                          capture_output=True, text=True)


def build_history(dest=None) -> tuple:
    """Build the fourteen-commit pricing repo and return ``(path, log)``, where
    ``log`` is ``[(sha7, message), ...]`` oldest first. Deterministic: rebuilt
    from scratch on every call, and the ids never change."""
    import shutil
    import tempfile
    from pathlib import Path
    dest = Path(dest) if dest else Path(tempfile.gettempdir()) / "eis_pricing_repo"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _git(dest.parent, "init", "-q", "-b", "main", str(dest))
    for i, (msg, content) in enumerate(PRICING_COMMITS):
        (dest / "pricing.py").write_text(content)
        _git(dest, "add", "pricing.py")
        _git(dest, "commit", "-q", "-m", msg,
             dates=f"2026-05-{4 + i:02d}T09:00:00 +0000")
    log = _git(dest, "log", "--reverse", "--format=%h %s").stdout
    return dest, [tuple(line.split(" ", 1)) for line in log.strip().splitlines()]


def _discount_from(source: str):
    """The ``bulk_discount`` a given commit's pricing.py defines."""
    ns = {}
    exec(source, ns)                    # noqa: S102 - controlled local demo
    return ns[FUNC]


def show_history(log: list) -> None:
    """The haystack: the one-line log, oldest first, exactly as a developer (or
    a model) would meet it. The CHECK rows are computed, not asserted: they run
    the decoy's parent and child over every quantity 0..999 (identical), then
    the culprit's (the middle tier is gone)."""
    from genai.agent import show_code, show_turn
    show_code("git log", "\n".join(f"{sha}  {msg}" for sha, msg in log))
    old, new = (_discount_from(PRICING_COMMITS[i][1])
                for i in (DECOY_IDX - 1, DECOY_IDX))
    same = sum(old(q) == new(q) for q in range(1000))
    show_turn("CHECK", f"{log[DECOY_IDX][0]} keeps the same discount on "
                       f"{same}/1000 quantities")
    old, new = (_discount_from(PRICING_COMMITS[i][1])
                for i in (CULPRIT_IDX - 1, CULPRIT_IDX))
    changed = sum(old(q) != new(q) for q in range(1000))
    show_turn("CHECK", f"{log[CULPRIT_IDX][0]} changes the discount on "
                       f"{changed}/1000 quantities")


# ── Contender one: the model reads the history ────────────────────────────────

def history_diffs(path) -> str:
    """The full history as the detective receives it: ``git log -p``, oldest
    first, every commit's id, message, and diff."""
    return _git(path, "log", "-p", "--reverse", "--format=commit %h%n%s").stdout


_DETECTIVE_PROMPT = (
    "A bug report just landed for pricing.py: bulk_discount(300) returns 5 "
    "when it should return 15. This test fails on the current build:\n\n"
    "```python\n{suite}```\n\n"
    "Below is the module's full commit history, oldest first, with every "
    "diff. Exactly one commit introduced the bug.\n\n{history}\n"
    "Reply with one line: the id of the commit that introduced the bug, "
    "then a clause saying why.")


def ask_detective(history: str, model: str = CODING_MODEL) -> str:
    """Hand the model the bug report, the failing test, and the whole history
    with diffs, and ask it to name the commit that introduced the bug."""
    prompt = _DETECTIVE_PROMPT.format(suite=make_suite(VISIBLE), history=history)
    return _ask(prompt, model=model, system="", max_tokens=150,
                options={"num_ctx": 8192})


def grade_detective(reply: str, log: list) -> str:
    """Which commit did the reply blame? Returns ``culprit``, ``decoy``, the
    sha7 of any other commit it named, or ``none`` if no commit id appears."""
    for tok in re.findall(r"\b[0-9a-f]{7,40}\b", reply.lower()):
        for i, (sha, _) in enumerate(log):
            if tok.startswith(sha):
                return ("culprit" if i == CULPRIT_IDX else
                        "decoy" if i == DECOY_IDX else sha)
    return "none"


# ── Captured constants (baked by scripts/_bisect_probe.py) ────────────────────
# Ten detective trials against the same history. "verdicts" tallies where the
# blame went (every "other" answer named the initial commit, per the raw replies
# in scripts/_bisect_out.json); "reply" is one verbatim representative answer.
DETECTIVE_STUDY = {
    "trials": 10,
    "verdicts": {"culprit": 3, "decoy": 4, "other": 3},
    "reply": "ca4891f refactored discount tiers into a table, inadvertently "
             "changing the logic for determining discounts.",
}


def show_detective(study: dict = None, log: list = None) -> None:
    """The model's real answer (verbatim) and the tally over all trials. The
    RUN row is the captured count; the VERDICT wording follows the numbers."""
    from genai.agent import show_turn
    study = study or DETECTIVE_STUDY
    v, n = study["verdicts"], study["trials"]
    show_turn("qwen2.5-coder", study["reply"])
    show_turn("RUN", f"{n} trials: culprit {v['culprit']}, decoy {v['decoy']}, "
                     f"another commit {v['other']}")
    if v["culprit"] == n:
        show_turn("VERDICT", "the diffs were read and the culprit named "
                             "every time")
    elif v["decoy"] >= v["culprit"]:
        show_turn("VERDICT", "the guilty-sounding message outpolls the commit "
                             "that actually broke the build")
    else:
        show_turn("VERDICT", "the culprit leads, but the verdict wobbles from "
                             "run to run")


# ── Contender two: git bisect runs the failing test ───────────────────────────

def bisect_history(path) -> dict:
    """``git bisect run`` with the chapter's failing test as the oracle. Writes
    the test (untracked, so it rides along as bisect checks out old commits),
    marks HEAD bad and the first commit good, and lets git search. Returns the
    real progress lines, the commit it lands on, and how many times the test
    ran."""
    import sys
    from pathlib import Path
    path = Path(path)
    test = ("from pricing import bulk_discount\n"
            "got = bulk_discount(300)\n"
            "assert got == 15, f'bulk_discount(300) returned {got}, expected 15'\n")
    (path / "test_tier.py").write_text(test)
    first = _git(path, "log", "--reverse", "--format=%h").stdout.split()[0]
    start = _git(path, "bisect", "start", "HEAD", first).stdout
    run = _git(path, "bisect", "run", sys.executable, "test_tier.py", check=False)
    lines, culprit = [], ""
    for line in start.splitlines() + run.stdout.splitlines():
        if line.startswith("Bisecting:"):
            lines.append(line)
        elif line.endswith("is the first bad commit"):
            culprit = line.split()[0][:7]
    _git(path, "bisect", "reset")
    subject = dict(build_log(path)).get(culprit, "")
    return {"start": f"git bisect start HEAD {first}",
            "lines": lines, "culprit": culprit, "subject": subject,
            "runs": run.stdout.count("running "),
            "commits": len(PRICING_COMMITS)}


def build_log(path) -> list:
    """The ``(sha7, message)`` log of an already-built history, oldest first."""
    out = _git(path, "log", "--reverse", "--format=%h %s").stdout
    return [tuple(line.split(" ", 1)) for line in out.strip().splitlines()]


def show_bisect(result: dict) -> None:
    """The mechanical contender: the command, git's real progress lines, and
    where it landed. The RUN row counts how many times the oracle executed."""
    from genai.agent import show_turn
    show_turn("TOOL", f"{result['start']}; git bisect run python test_tier.py")
    for line in result["lines"]:
        show_turn("GIT", line)
    show_turn("GIT", f"{result['culprit']} is the first bad commit: "
                     f"{result['subject']}")
    show_turn("RUN", f"the failing test ran {result['runs']} times to search "
                     f"{result['commits']} commits")


# ── The bug that only shows up sometimes ──────────────────────────────────────
# The repair loop's oracle is "the test goes green". That assumes a test is a
# deterministic verdict, and the nastiest bugs make it a coin flip. highest_priority
# resolves ties above its threshold by shuffling, so on the same input it returns
# the right task only some of the time. A repair loop that re-runs the test once and
# stops on green will ship the bug on any run the coin lands green. The fix is in the
# acceptance rule, not the patch: re-run the test k times and accept only an
# all-green. Seeded so the run is reproducible; the point is that without the reruns
# you couldn't tell.
import random as _random

FLAKY_SRC = (
    "import random\n"
    "def highest_priority(tasks):\n"
    "    order = tasks[:]\n"
    "    random.shuffle(order)   # order-dependent bug\n"
    "    for name, pri in order:\n"
    "        if pri >= 5:   # first past 5, not the max\n"
    "            return name\n"
    "    return max(tasks, key=lambda t: t[1])[0]\n")

FLAKY_FIXED = ("def highest_priority(tasks):\n"
               "    return max(tasks, key=lambda t: t[1])[0]\n")

# The true answer is the highest-priority task; two tasks clear the threshold, so
# the shuffle decides which one the buggy version returns.
FLAKY_CASE = [("deploy", 9), ("email", 7), ("cleanup", 3)]
FLAKY_ANSWER = "deploy"


def _flaky_passes(src: str, case: list = None) -> bool:
    """True when the function returns the correct highest-priority task on ``case``."""
    case = case or FLAKY_CASE
    ns = {}
    exec(src, ns)                           # noqa: S102 - controlled local demo
    return ns["highest_priority"](case) == FLAKY_ANSWER


def flaky_rate(src: str, k: int, seed: int = 0) -> int:
    """How many of ``k`` seeded runs of the test fail on ``src``."""
    _random.seed(seed)
    return sum(not _flaky_passes(src) for _ in range(k))


def show_flaky_gate(k: int = 20, seed: int = 0) -> None:
    """One green run isn't a fix when the test is flaky. Show the buggy function,
    that a single run passes, that k runs fail a chunk of the time, and that the
    real fix and a k-run gate agree."""
    from genai.agent import show_code, show_turn
    show_code("highest_priority", FLAKY_SRC.rstrip())
    _random.seed(seed)
    first = _flaky_passes(FLAKY_SRC)
    fails = flaky_rate(FLAKY_SRC, k, seed)
    show_turn("one run", ("green" if first else "red") +
              ": a loop that stops on one green run would call it fixed")
    show_turn(f"re-run {k}x", f"{fails} of {k} runs RED: the bug fires about "
              f"{round(fails / k * 100)}% of the time")
    fixed_fails = flaky_rate(FLAKY_FIXED, k, seed)
    show_turn("the real fix", f"return max(...): {fixed_fails} of {k} red")
    show_turn("RULE", "accept a patch only when its test is green k times, not once")

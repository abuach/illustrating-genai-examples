"""A model changes code it didn't write, and a way to tell whether it broke anything.

The earlier rungs of the ladder checked code written from a clean slate. Most
engineering isn't that. It's changing code that already exists: a rename here, a
library swap there, the same edit repeated across every place that called the old
thing, all of it meant to leave behavior exactly as it was. That last clause is
the whole problem. A *refactor* is a change that's supposed to preserve behavior,
so the oracle isn't a spec the model reads; it's the program's own past. Run the
old version and the new version on the same inputs and every answer has to match.
That check has a name, *differential testing*, and it catches what a test suite
can't: a sweeping edit that stays green on the cases the suite happens to try and
quietly changes the answer on one it doesn't.

The task here is a small *migration*. A pricing module calls a deprecated helper
``discount(amount, percent)`` in seven places, and every call has to move to a new
``apply_rate(amount, rate)`` that takes a fraction instead of a percent. The
visible edit is mechanical, ``discount(x, 15)`` becomes ``apply_rate(x, 0.15)``,
and a code model does it cleanly at every site. The catch is a quieter one: the
old helper rounded its answer to the cent and the new one doesn't, so a faithful
percent-to-rate swap that leaves the rounding out is a complete, plausible
migration that returns a slightly different number on any price that isn't round.
The chapter measures both things that can go wrong: *completeness* (how many of
the seven sites moved) and *behavior preservation* (do the migrated sites still
compute the same number as before). The codebase, the suite, and the differential
harness are plain Python; only the model is nondeterministic, so one real run is
captured once by ``scripts/_refactoring_probe.py`` and baked into the constants
below, and the chapter's cells recompute their verdicts from that baked migration
without ever calling the model. The migrator is a code model (qwen2.5-coder).
"""
import ast
from statistics import mean

from genai.llm import CODING_MODEL, ask as _ask

DEPRECATED = "discount"          # the old helper every call site has to leave
REPLACEMENT = "apply_rate"       # the new one every call site has to reach

# ── The codebase under refactor ───────────────────────────────────────────────
# PRELUDE holds the two helpers and the named rates; it's canonical and never
# rewritten, so a migrated module is always PRELUDE + the model's functions. The
# deprecated helper stays defined on purpose: a missed call site keeps working
# against it, which is exactly why an unfinished migration can look finished.

PRELUDE = (
    "MEMBER_PCT = 10\n"
    "BULK_PCT = 20\n"
    "\n"
    "def discount(amount, percent):   # deprecated\n"
    "    return round(amount * (1 - percent / 100), 2)\n"
    "\n"
    "def apply_rate(amount, rate):    # new\n"
    "    return amount * (1 - rate)\n")

# Six public functions, seven calls to the deprecated helper between them. The
# spread is deliberate: plain literals, a rate routed through a named constant, a
# call inside a branch, a percent passed straight through as a parameter, and one
# expression that discounts twice. Each shape is a different way the mechanical
# part of the edit can slip; the rounding is the part every shape shares.
OLD_FUNCS = (
    "def promo_price(total):\n"
    "    return discount(total, 15)\n"
    "\n"
    "def clearance_price(total):\n"
    "    return discount(total, 50)\n"
    "\n"
    "def member_price(total):\n"
    "    return discount(total, MEMBER_PCT)\n"
    "\n"
    "def bulk_price(total, qty):\n"
    "    if qty >= 10:\n"
    "        return discount(total, BULK_PCT)\n"
    "    return total\n"
    "\n"
    "def seasonal_price(total, percent):\n"
    "    return discount(total, percent)\n"
    "\n"
    "def stacked_price(total):\n"
    "    return discount(discount(total, 10), 5)\n")

OLD_MODULE = PRELUDE + "\n" + OLD_FUNCS

PUBLIC = ["promo_price", "clearance_price", "member_price",
          "bulk_price", "seasonal_price", "stacked_price"]

# The inputs the differential oracle runs both versions on. Several totals per
# function, plus the qty / percent each parametered one needs. These are the
# "same battery" the old and new code both have to answer identically. The round
# totals (100, 250) are the cases a typical suite checks; the cents totals (99.99,
# 19.99) are the ones it skips, and the ones where a dropped rounding shows.
BATTERY = {
    "promo_price":     [(0,), (100,), (250,), (99.99,), (19.99,)],
    "clearance_price": [(0,), (100,), (250,), (99.99,)],
    "member_price":    [(0,), (100,), (250,), (99.99,)],
    "bulk_price":      [(100, 5), (100, 10), (250, 20), (99.99, 10)],
    "seasonal_price":  [(100, 0), (100, 20), (100, 100), (99.99, 15)],
    "stacked_price":   [(0,), (100,), (250,), (99.99,)],
}

# ── The existing test suite (every function covered, all round numbers) ───────
# It checks each of the six functions with one typical, round input. Green here
# means "the cases someone already wrote down still pass." Those cases are whole
# dollars, where the dropped rounding makes no difference, so green says nothing
# about the cents prices the migration quietly changed.
SUITE = (
    "def test_promo():\n"
    "    assert promo_price(100) == 85.0\n"
    "def test_clearance():\n"
    "    assert clearance_price(200) == 100.0\n"
    "def test_member():\n"
    "    assert member_price(100) == 90.0\n"
    "def test_bulk():\n"
    "    assert bulk_price(100, 10) == 80.0\n"
    "def test_seasonal():\n"
    "    assert seasonal_price(100, 20) == 80.0\n"
    "def test_stacked():\n"
    "    assert stacked_price(100) == 85.5\n")


# ── Counting call sites: the completeness axis ────────────────────────────────

def _calls_to(name: str, source: str) -> int:
    """How many times ``name(...)`` is called in this source, nesting included."""
    return sum(isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
               for n in ast.walk(ast.parse(source)))


TOTAL_SITES = _calls_to(DEPRECATED, OLD_FUNCS)        # seven, computed not typed


def completeness(funcs: str) -> dict:
    """How much of the migration actually happened, by counting what's left.

    A call still made to the deprecated helper is a site the model skipped. With
    the old helper still defined, those skipped sites keep returning the right
    number, so the suite never reddens, which is precisely why you have to count
    them rather than run them. Returns ``{migrated, total, remaining, frac}``.
    """
    remaining = _calls_to(DEPRECATED, funcs)
    migrated = TOTAL_SITES - remaining
    return {"migrated": migrated, "total": TOTAL_SITES, "remaining": remaining,
            "frac": round(migrated / TOTAL_SITES, 3)}


# ── Differential testing: the behavior-preservation axis (the new oracle) ─────

def _call(ns: dict, fn: str, args: tuple):
    """Call one function from a built namespace; an exception is its own answer,
    so a migration that makes a function crash counts as a behavior change, not a
    test that fell over."""
    try:
        return ns[fn](*args)
    except Exception as exc:                          # noqa: BLE001 - any failure is a mismatch
        return f"{type(exc).__name__}"


def differential_test(old_src: str, new_src: str, battery: dict = BATTERY) -> list:
    """Run the old and new code on the same inputs; return where they disagree.

    This is the refactor oracle. The old program is the answer key, so we don't
    need a spec: for every input in the battery, the migrated function has to
    return exactly what the original did. Each disagreement is a
    ``{func, args, want, got}`` row, the concrete case the edit changed. A new
    module that won't even import is the loudest disagreement of all.
    """
    old_ns, new_ns = {}, {}
    exec(old_src, old_ns)                             # noqa: S102 - controlled local demo
    try:
        exec(new_src, new_ns)                         # noqa: S102 - controlled local demo
    except Exception as exc:                          # noqa: BLE001
        return [{"func": "<module>", "args": (), "want": "imports",
                 "got": f"{type(exc).__name__}"}]
    out = []
    for fn, rows in battery.items():
        for args in rows:
            want, got = _call(old_ns, fn, args), _call(new_ns, fn, args)
            if want != got:
                out.append({"func": fn, "args": args, "want": want, "got": got})
    return out


# ── The existing suite as a second check ──────────────────────────────────────

def run_suite(suite: str, module_src: str) -> list:
    """Run every ``test_*`` in ``suite`` against a module; return the failures."""
    ns = {}
    exec(module_src, ns)                              # noqa: S102 - controlled local demo
    exec(suite, ns)                                   # noqa: S102 - controlled local demo
    failures = []
    for name, fn in [(k, v) for k, v in ns.items()
                     if k.startswith("test_") and callable(v)]:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, type(exc).__name__))
    return failures


def suite_green(funcs: str) -> bool:
    """True when the existing suite passes on the migrated module."""
    return not run_suite(SUITE, PRELUDE + "\n" + funcs)


# ── The migrator: a model rewrites the call sites ─────────────────────────────

def _clean(text: str) -> str:
    """Strip a markdown fence and keep the model's real migration from a reply.

    We keep the public functions *and* any module-level constants the model
    introduces (a ``MEMBER_RATE = MEMBER_PCT / 100`` it factored out is part of a
    correct migration, so dropping it would forge a failure the model didn't
    make). We drop only imports and a re-pasted copy of the canonical helpers, so
    ``discount`` and ``apply_rate`` keep their original meaning and the only thing
    under test is how the model rewrote the call sites.
    """
    import re
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    body = (fenced.group(1) if fenced else text).strip()
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body                                   # let the oracle report the break
    keep = lambda n: (isinstance(n, ast.FunctionDef) and n.name in PUBLIC) \
        or isinstance(n, ast.Assign)                  # the model's own rate constants
    tree.body = [n for n in tree.body if keep(n)]
    return ast.unparse(tree)


def migration_prompt() -> str:
    return (
        "Here is a Python pricing module. The helper `discount(amount, percent)` "
        "is deprecated: its `percent` is a number from 0 to 100. A new helper "
        "`apply_rate(amount, rate)` takes a `rate` from 0 to 1 instead. Rewrite "
        "every function so it calls `apply_rate` instead of `discount`, keeping "
        "each function's name, signature, and exact behavior the same. Convert "
        "each percent to the matching rate. Return only the rewritten functions, "
        "no helpers, no commentary.\n\n" + OLD_MODULE)


def run_migration(model: str = CODING_MODEL) -> str:
    """Ask the code model to migrate every call site; return its functions."""
    return _clean(_ask(migration_prompt(), model=model, system="", max_tokens=700))


def run_trial(model: str = CODING_MODEL) -> dict:
    """One migration scored on both axes plus the suite it would ship behind."""
    funcs = run_migration(model)
    return {"funcs": funcs,
            "completeness": completeness(funcs),
            "mismatches": differential_test(OLD_MODULE, PRELUDE + "\n" + funcs),
            "suite_green": suite_green(funcs)}


def refactor_study(trials: int = 12, model: str = CODING_MODEL) -> dict:
    """Run the migration ``trials`` times; summarize where it lands.

    Per trial we record three verdicts: did the existing suite pass, did every
    one of the nine sites move, and did the old and new code agree on the whole
    battery. The headline the chapter draws from this is the distance between the
    first verdict and the other two, the runs the green bar called done that the
    oracles called unfinished.
    """
    rows = []
    for _ in range(trials):
        t = run_trial(model)
        rows.append({"frac": t["completeness"]["frac"],
                     "migrated": t["completeness"]["migrated"],
                     "suite_green": t["suite_green"],
                     "behavior_preserved": not t["mismatches"],
                     "n_mismatch": len(t["mismatches"])})
    n = len(rows)
    rate = lambda pred: round(sum(bool(pred(r)) for r in rows) / n, 3)
    return {
        "trials": n, "total_sites": TOTAL_SITES,
        "suite_green":         rate(lambda r: r["suite_green"]),
        "complete":            rate(lambda r: r["frac"] == 1.0),
        "behavior_preserved":  rate(lambda r: r["behavior_preserved"]),
        "fully_correct":       rate(lambda r: r["frac"] == 1.0 and r["behavior_preserved"]),
        "mean_completeness":   round(mean(r["frac"] for r in rows), 3),
        # the silent shortfall: the suite passed but the job wasn't actually done.
        "green_but_unfinished": rate(lambda r: r["suite_green"]
                                     and not (r["frac"] == 1.0 and r["behavior_preserved"])),
        "rows": rows,
    }


# ── Captured constants (baked by scripts/_refactoring_probe.py) ───────────────
# Filled in from a real qwen2.5-coder run; the probe prints replacements to paste.
# REP is one representative trial shown as the chapter's worked example: the real
# functions the model wrote, the sites it moved, and the battery rows where old
# and new part ways. REFACTOR_STUDY is the per-axis summary over many trials.

# One real qwen2.5-coder migration, kept verbatim. It moved all seven calls,
# converted every unit, and even factored the constant rates into their own
# names: a clean, complete edit that dropped the rounding. The derived verdicts
# (completeness, mismatches, suite_green) are recomputed from these functions at
# import, so nothing here is a typed-in number; only the funcs string was captured.
_REP_FUNCS = (
    "MEMBER_RATE = MEMBER_PCT / 100\n"
    "BULK_RATE = BULK_PCT / 100\n"
    "\n"
    "def promo_price(total):\n"
    "    return apply_rate(total, 0.15)\n"
    "\n"
    "def clearance_price(total):\n"
    "    return apply_rate(total, 0.5)\n"
    "\n"
    "def member_price(total):\n"
    "    return apply_rate(total, MEMBER_RATE)\n"
    "\n"
    "def bulk_price(total, qty):\n"
    "    if qty >= 10:\n"
    "        return apply_rate(total, BULK_RATE)\n"
    "    return total\n"
    "\n"
    "def seasonal_price(total, percent):\n"
    "    rate = percent / 100\n"
    "    return apply_rate(total, rate)\n"
    "\n"
    "def stacked_price(total):\n"
    "    return apply_rate(apply_rate(total, 0.1), 0.05)\n")

REP = {"funcs": _REP_FUNCS,
       "completeness": completeness(_REP_FUNCS),
       "mismatches": differential_test(OLD_MODULE, PRELUDE + "\n" + _REP_FUNCS),
       "suite_green": suite_green(_REP_FUNCS)}

# Per-axis summary over 12 real migrations (scripts/_refactoring_probe.py,
# qwen2.5-coder). Every run moved all seven sites and passed the suite, and every
# run dropped the rounding, so behavior was preserved on none of them: a complete,
# green, silently-wrong migration is the rule here, not the unlucky exception.
REFACTOR_STUDY = {
    "trials": 12, "total_sites": TOTAL_SITES,
    "suite_green": 1.0, "complete": 1.0, "behavior_preserved": 0.0,
    "fully_correct": 0.0, "mean_completeness": 1.0, "green_but_unfinished": 1.0,
}


def _fmt(funcs: str) -> str:
    """Render code for the page: ``ast.unparse`` normalizes the spacing, and the
    blank lines it inserts between defs are dropped so a six-function listing fits
    a code box. Each ``def`` still starts at the margin, so the functions stay
    visually separate; the indented bodies keep their shape through the reflow."""
    return "\n".join(ln for ln in ast.unparse(ast.parse(funcs)).split("\n")
                     if ln.strip())


# ── Show helpers: the migration as a transcript ───────────────────────────────

def show_api() -> None:
    """The two helpers the migration moves between: the deprecated ``discount``,
    which rounds to the cent, and the new ``apply_rate``, which doesn't. The whole
    behavior change hides in that one missing ``round``."""
    from genai.agent import show_code
    show_code("pricing.py", PRELUDE.rstrip())          # raw, to keep the comments


def show_migration(rep: dict = None) -> None:
    """The task handed over, then the real functions the model wrote back."""
    from genai.agent import show_turn, show_code
    rep = rep or REP
    show_turn("you", "Migrate every call off the deprecated discount() onto "
                     "apply_rate(), keeping behavior identical.")
    show_code("qwen2.5-coder", _fmt(rep["funcs"]))


def show_suite_run(rep: dict = None) -> None:
    """Run the existing suite against the migrated module; report what it says.
    The TEST lines are the real pass/fail run_suite returns, not narration."""
    from genai.agent import show_turn
    rep = rep or REP
    module = PRELUDE + "\n" + rep["funcs"]
    fails = run_suite(SUITE, module)
    passed = sum(1 for _ in ast.walk(ast.parse(SUITE))
                 if isinstance(_, ast.FunctionDef)) - len(fails)
    show_turn("SUITE", f"{passed}/{passed + len(fails)} existing tests pass against "
                       "the migrated module")
    show_turn("VERDICT", "all green" if not fails else
              f"{len(fails)} failing: {', '.join(n for n, _ in fails)}")


def show_oracle_verdict(rep: dict = None) -> None:
    """The two oracles the green suite can't stand in for: a structural count of
    sites left behind, and the differential cases where old and new disagree."""
    from genai.agent import show_turn
    rep = rep or REP
    comp = rep["completeness"]
    show_turn("COMPLETENESS", f"{comp['migrated']}/{comp['total']} call sites moved "
              f"to {REPLACEMENT}" + (f"; {comp['remaining']} still call "
              f"{DEPRECATED}" if comp["remaining"] else ""))
    mism = rep["mismatches"]
    show_turn("DIFFERENTIAL", f"{len(mism)} input(s) where old and new disagree"
              if mism else "old and new agree on every input in the battery")
    for m in mism[:3]:
        args = ", ".join(map(str, m["args"]))
        show_turn("MISMATCH", f"{m['func']}({args}): old -> {_num(m['want'])}, "
                              f"new -> {_num(m['got'])}")


def _num(x):
    """A computed differential value, trimmed of float noise for the page: the new
    code's 84.99149999999999 reads as 84.9915, which is the point (more cents than
    the old code kept), without the trailing repaint."""
    return round(x, 4) if isinstance(x, float) else x


# ── An approved fix: when the answer key itself is wrong ──────────────────────
# The chapter's closing complication. The team wants a known wart FIXED during
# the migration: ``stacked_price`` discounts 10% then 5%, an effective 14.5%
# off, but the promotion it implements is advertised as a flat 15% off. Against
# the old code, differential testing flags that approved fix exactly the way it
# flags a regression, so the approved change is encoded once, as a *patched
# reference*: the old module with the corrected ``stacked_price`` swapped in,
# still built on the old helper so the rounding survives. Mismatches then
# partition: rows the patched key expects to move, and rows nobody approved.

APPROVED_FIX = (
    "stacked_price currently discounts 10% then 5%, an effective 14.5% off, "
    "but the promotion it implements is advertised as a flat 15% off, so "
    "rewrite stacked_price to apply a single 15% discount")

# The old functions with only the approved fix applied, on the old helper, so
# the cent rounding stays. This, not the old module, is now the answer key.
PATCHED_FUNCS = OLD_FUNCS.replace(
    "def stacked_price(total):\n"
    "    return discount(discount(total, 10), 5)\n",
    "def stacked_price(total):\n"
    "    return discount(total, 15)\n")

PATCHED_MODULE = PRELUDE + "\n" + PATCHED_FUNCS

# Where the fix is licensed to land, computed by the same oracle rather than
# typed in: every battery row where the old code and the patched key disagree.
FOOTPRINT = differential_test(OLD_MODULE, PATCHED_MODULE)


def fix_prompt() -> str:
    """The migration prompt with one approved exception spliced in. Identical to
    ``migration_prompt`` word for word except for the carve, so the only new
    variable in the trial is the fix the team asked for."""
    return (
        "Here is a Python pricing module. The helper `discount(amount, percent)` "
        "is deprecated: its `percent` is a number from 0 to 100. A new helper "
        "`apply_rate(amount, rate)` takes a `rate` from 0 to 1 instead. Rewrite "
        "every function so it calls `apply_rate` instead of `discount`, keeping "
        "each function's name, signature, and exact behavior the same, with one "
        "approved exception: " + APPROVED_FIX + ". Convert each percent to the "
        "matching rate. Return only the rewritten functions, no helpers, no "
        "commentary.\n\n" + OLD_MODULE)


def run_fix_migration(model: str = CODING_MODEL) -> str:
    """Ask the model for the migration and the approved fix in one prompt."""
    return _clean(_ask(fix_prompt(), model=model, system="", max_tokens=700))


def _row_key(m: dict) -> tuple:
    """A mismatch row's identity: which function, on which input."""
    return (m["func"], tuple(m["args"]))


def carveout_partition(funcs: str) -> dict:
    """Score a migrate-plus-fix against both answer keys.

    ``vs_old`` is the plain differential verdict against the old code, where the
    approved fix and a real regression are indistinguishable. The carve-out
    partitions it: ``unexpected`` is every battery row where the migration still
    disagrees with the patched key (an unapproved change, or the approved one
    done wrong), and ``landed`` is the FOOTPRINT rows the migration now matches
    exactly. A clean carve-out is landed == FOOTPRINT and unexpected == [].
    """
    new_src = PRELUDE + "\n" + funcs
    unexpected = differential_test(PATCHED_MODULE, new_src)
    off = {_row_key(m) for m in unexpected}
    return {"funcs": funcs,
            "completeness": completeness(funcs),
            "vs_old": differential_test(OLD_MODULE, new_src),
            "unexpected": unexpected,
            "landed": [m for m in FOOTPRINT if _row_key(m) not in off]}


# ── Captured constant (baked by scripts/_intentional_change_probe.py) ─────────
# One real qwen2.5-coder migrate-plus-fix, kept verbatim; all 5 probe trials
# landed on the same verdicts. The model moved all seven sites, applied the
# approved flat-15 fix, and dropped the cent rounding again, everywhere,
# including inside the one function it had permission to change. As with REP,
# only the funcs string was captured; every verdict is recomputed at import.

_FIX_FUNCS = (
    "MEMBER_RATE = MEMBER_PCT / 100\n"
    "BULK_RATE = BULK_PCT / 100\n"
    "\n"
    "def promo_price(total):\n"
    "    return apply_rate(total, 0.15)\n"
    "\n"
    "def clearance_price(total):\n"
    "    return apply_rate(total, 0.5)\n"
    "\n"
    "def member_price(total):\n"
    "    return apply_rate(total, MEMBER_RATE)\n"
    "\n"
    "def bulk_price(total, qty):\n"
    "    if qty >= 10:\n"
    "        return apply_rate(total, BULK_RATE)\n"
    "    return total\n"
    "\n"
    "def seasonal_price(total, percent):\n"
    "    return apply_rate(total, percent / 100)\n"
    "\n"
    "def stacked_price(total):\n"
    "    return apply_rate(total, 0.15)\n")

FIX_REP = carveout_partition(_FIX_FUNCS)


# ── Show helpers for the carve-out ────────────────────────────────────────────

def show_fix_migration(rep: dict = None) -> None:
    """The migrate-plus-fix handed over, then the functions the model wrote."""
    from genai.agent import show_turn, show_code
    rep = rep or FIX_REP
    show_turn("you", "Same migration, one approved change: stacked_price is "
                     "advertised as a flat 15% off; fix it while you're in there.")
    show_code("qwen2.5-coder", _fmt(rep["funcs"]))


def show_fix_vs_old(rep: dict = None) -> None:
    """The old checks on the migrate-plus-fix: the suite, then the differential
    oracle with the old code still the answer key. The MISMATCH sample shows the
    head and tail of the list so both kinds of disagreement are on the page; the
    labels don't distinguish them because the oracle can't."""
    from genai.agent import show_turn
    rep = rep or FIX_REP
    module = PRELUDE + "\n" + rep["funcs"]
    fails = run_suite(SUITE, module)
    show_turn("SUITE", "all green" if not fails else
              f"{len(fails)} failing: {', '.join(n for n, _ in fails)}")
    mism = rep["vs_old"]
    show_turn("DIFFERENTIAL", f"{len(mism)} input(s) where old and new disagree")
    head = mism[:2]
    for m in head + [m for m in mism[-2:] if m not in head]:
        args = ", ".join(map(str, m["args"]))
        show_turn("MISMATCH", f"{m['func']}({args}): old -> {_num(m['want'])}, "
                              f"new -> {_num(m['got'])}")


def show_carveout(rep: dict = None) -> None:
    """The intentional-change carve-out verdict. FOOTPRINT is where the patched
    key licenses a change; EXPECTED counts the carved rows the migration matches;
    UNEXPECTED is every remaining disagreement with the patched key, each one a
    change nobody approved."""
    from genai.agent import show_turn
    rep = rep or FIX_REP
    show_turn("FOOTPRINT", f"the approved fix moves {len(FOOTPRINT)} battery "
              f"row(s), all in {', '.join(sorted({m['func'] for m in FOOTPRINT}))}")
    show_turn("EXPECTED", f"{len(rep['landed'])}/{len(FOOTPRINT)} carved rows "
              "now match the patched key")
    unexp = rep["unexpected"]
    show_turn("UNEXPECTED", f"{len(unexp)} row(s) still disagree with the "
              "patched key" if unexp else "no other row differs from the patched key")
    for m in unexp[:3]:
        args = ", ".join(map(str, m["args"]))
        show_turn("MISMATCH", f"{m['func']}({args}): patched -> {_num(m['want'])}, "
                              f"new -> {_num(m['got'])}")


# ── The refactor that adds a bug ──────────────────────────────────────────────
# Differential testing only compares the two programs on the inputs you feed it.
# So a migration can introduce behavior the old code never had, as long as the new
# behavior fires only on inputs the battery skips. Asked to migrate AND "make sure
# a price never comes back negative", a model adds an `if result < 0: result = 0`
# clamp to every function, a branch the originals lack. On the round positive
# totals a refund-free suite checks, old and new agree exactly. Widen the battery
# to negative totals (a refund) and they diverge, at exactly the sites the clamp
# touched. ADDED_MIGRATION is one real qwen2.5-coder run under that instruction,
# captured by scripts/_refactoring_added_probe.py.

# Round positive totals: the clamp never fires and the dropped rounding doesn't
# show, so on these the added migration is indistinguishable from the original.
NARROW_BATTERY = {
    "promo_price": [(100,), (250,), (500,)],
    "clearance_price": [(100,), (200,), (400,)],
    "member_price": [(100,), (250,), (500,)],
    "bulk_price": [(100, 5), (100, 10), (250, 20)],
    "seasonal_price": [(100, 0), (100, 20), (200, 50)],
    "stacked_price": [(100,), (200,), (500,)],
}

# The same battery widened with negative totals (refunds), the case nobody wrote a
# test for and the only place the clamp changes the answer.
WIDE_BATTERY = {
    fn: rows + [(-abs(r[0]),) + r[1:] for r in rows[:1]]
    for fn, rows in NARROW_BATTERY.items()
}

ADDED_MIGRATION = (
    "def promo_price(total):\n    rate = 0.15\n    result = apply_rate(total, rate)\n"
    "    if result < 0:\n        return 0\n    return result\n\n"
    "def clearance_price(total):\n    rate = 0.5\n    result = apply_rate(total, rate)\n"
    "    if result < 0:\n        return 0\n    return result\n\n"
    "def member_price(total):\n    global MEMBER_PCT\n    rate = MEMBER_PCT / 100\n"
    "    result = apply_rate(total, rate)\n    if result < 0:\n        return 0\n"
    "    return result\n\n"
    "def bulk_price(total, qty):\n    if qty >= 10:\n        rate = BULK_PCT / 100\n"
    "        result = apply_rate(total, rate)\n        if result < 0:\n"
    "            return 0\n        return result\n    return total\n\n"
    "def seasonal_price(total, percent):\n    rate = percent / 100\n"
    "    result = apply_rate(total, rate)\n    if result < 0:\n        return 0\n"
    "    return result\n\n"
    "def stacked_price(total):\n    rate1 = 0.1\n    rate2 = 0.05\n"
    "    result1 = apply_rate(total, rate1)\n    if result1 < 0:\n        return 0\n"
    "    result2 = apply_rate(result1, rate2)\n    if result2 < 0:\n        return 0\n"
    "    return result2\n")

_ADDED_OLD_PROMO = "def promo_price(total):\n    return discount(total, 15)"
_ADDED_NEW_PROMO = ("def promo_price(total):\n    rate = 0.15\n"
                    "    result = apply_rate(total, rate)\n"
                    "    if result < 0:\n        return 0\n    return result")


def _battery_size(battery: dict) -> int:
    return sum(len(rows) for rows in battery.values())


def show_added_branch(migration: str = None) -> None:
    """A migration that adds a guard the old code lacked. Differential testing on
    the round-total battery preserves behavior everywhere; widen it to refunds and
    the added clamp diverges, at exactly the sites it touched."""
    from genai.agent import show_code, show_turn
    migration = migration or ADDED_MIGRATION
    new_module = PRELUDE + "\n" + migration
    show_code("old promo", _ADDED_OLD_PROMO)
    show_code("new promo", _ADDED_NEW_PROMO)
    narrow = differential_test(OLD_MODULE, new_module, NARROW_BATTERY)
    show_turn("round totals", f"{_battery_size(NARROW_BATTERY)} cases, "
              f"{len(narrow)} disagree: behavior preserved on every one")
    wide = differential_test(OLD_MODULE, new_module, WIDE_BATTERY)
    show_turn("+ refunds", f"widen to negative totals: {len(wide)} disagree, all "
              "on the added clamp")
    for row in wide[:3]:
        arg = row["args"][0] if len(row["args"]) == 1 else row["args"]
        show_turn(row["func"], f"({arg}): old {row['want']} -> new {row['got']}")


# ── The battery that grows itself: a searched differential oracle ──────────────
# Every oracle so far ran on a battery someone typed out, and the last section's
# whole lesson was that a typed battery only covers the inputs you thought of. The
# fix is to stop typing inputs and let Hypothesis generate them, running the old
# and migrated code side by side and shrinking any disagreement to its smallest
# case, the same property search the Testing chapter used, now with the old code as
# the oracle. Handed the added-clamp migration, the round-total battery sees
# nothing; the search finds the dropped cent at 0.01, and once it may draw refunds,
# the clamp at -0.01, neither one hand-listed. Deterministic via derandomized
# Hypothesis (the Testing chapter's setup), so this runs live with nothing baked.

_DIFF_FUNCS = ("promo_price", "clearance_price", "member_price", "stacked_price")


def _diff_counterexample(new_funcs: str, strategy) -> tuple:
    """Search prices with Hypothesis for the smallest one where the old code and the
    migration disagree. Returns the shrunk ``(func, total, old, new)``, or None."""
    from hypothesis import given, settings
    old_ns, new_ns = {}, {}
    exec(OLD_MODULE, old_ns)                          # noqa: S102 - controlled demo
    exec(PRELUDE + "\n" + new_funcs, new_ns)          # noqa: S102 - controlled demo
    box = {}

    @settings(max_examples=500)
    @given(total=strategy)
    def prop(total):
        for fn in _DIFF_FUNCS:
            old_v, new_v = old_ns[fn](total), new_ns[fn](total)
            if old_v != new_v:
                box["ex"] = (fn, total, old_v, new_v)
            assert old_v == new_v

    try:
        prop()
        return None
    except AssertionError:
        return box["ex"]


def show_property_battery() -> None:
    """The differential oracle with its inputs generated, not hand-picked. The
    round-total battery calls the added-clamp migration behavior-preserving;
    Hypothesis, drawing prices, shrinks to the smallest cent where they part, and
    once it may draw refunds it finds the added clamp the last section had to widen
    the battery by hand to catch."""
    from genai.agent import show_turn
    from genai.test import _hypothesis_setup
    from hypothesis import strategies as st
    _hypothesis_setup()
    money = st.integers(0, 100_000).map(lambda c: c / 100)        # prices to the cent
    refunds = st.integers(-100_000, -1).map(lambda c: c / 100)    # negative totals
    added = PRELUDE + "\n" + ADDED_MIGRATION
    narrow = differential_test(OLD_MODULE, added, NARROW_BATTERY)
    show_turn("BATTERY", f"{_battery_size(NARROW_BATTERY)} hand-picked round totals: "
              f"{len(narrow)} disagree, behavior looks preserved")
    fn, total, o, n = _diff_counterexample(ADDED_MIGRATION, money)
    show_turn("HYPOTHESIS", f"draw prices instead, shrink to the smallest gap: "
              f"{fn}({total}), old {_num(o)} -> new {_num(n)}")
    fn2, t2, o2, n2 = _diff_counterexample(ADDED_MIGRATION, refunds)
    show_turn("+ REFUNDS", f"let it draw negative totals too: {fn2}({t2}), old "
              f"{_num(o2)} -> new {_num(n2)}, the added clamp from before")

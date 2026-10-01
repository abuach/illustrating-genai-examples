"""The book's opening catch: a model vouches for buggy code, one line refutes it.

The Introduction needs exactly one live demonstration of the distance the whole book
is about: a model reads a small function, hands back a confident professional
verdict, and is wrong, and a single deterministic line of Python settles the
argument. The target is ``clock_after``, which adds hours on a 12-hour clock
face with the remainder operator. The remainder wraps into the range 0-11, but
no clock face has a 0: three hours after nine o'clock is twelve o'clock, and
the function says 0. The bug is real, subtle, and visible to anyone who has
ever read a clock, no programming background required.

The reviewer is qwen2.5-coder, the same house code model that writes tests and
patches bugs later in the book; reviewing a four-line function is its home
turf, which is the point. Asked whether the function is correct for all
inputs, it approves, sample after sample, and the captured reply even endorses
the wrapped range as proper 12-hour-clock behavior. The model's verbatim reply
is baked once by ``scripts/_intro_coldopen_probe.py`` as ``VERDICT``;
everything else in this module is deterministic, so the notebook cells that
display it need no freeze.
"""
from genai.llm import CODING_MODEL, ask as _ask

# ── The function under review ─────────────────────────────────────────────────
# Four lines a scheduling app might ship. (now + hours) % 12 wraps the total
# around the dial, which is the right instinct with the wrong landing spot: the
# remainder runs 0-11 and a clock face runs 1-12, so every answer that should
# read 12 comes back 0. The source is kept as a string so the review prompt,
# the display helper, and the callable below all share one copy of the code.
SUSPECT = (
    "def clock_after(now, hours):\n"
    "    \"\"\"The hour a 12-hour clock shows,\n"
    "    `hours` hours after `now`.\"\"\"\n"
    "    return (now + hours) % 12\n")

# The callable is defined by executing the reviewed source, so the function the
# reader runs is byte-for-byte the function the model approved.
_ns: dict = {}
exec(SUSPECT, _ns)  # noqa: S102 - executing our own four-line constant
clock_after = _ns["clock_after"]

# ── The review ────────────────────────────────────────────────────────────────
REVIEW_PROMPT = (
    "Here is a function from a code review:\n\n```python\n{src}```\n\n"
    "Is this function correct for all inputs? Reply with `VERDICT: CORRECT` "
    "or `VERDICT: BUGGY`, then one sentence explaining your answer.")


def review(src: str = SUSPECT, model: str = CODING_MODEL) -> str:
    """Ask ``model`` for a review verdict on ``src`` (a live call, used by the
    probe script; the notebook shows the baked ``VERDICT`` instead)."""
    return _ask(REVIEW_PROMPT.format(src=src), model=model, system="",
                max_tokens=150)


# ── Captured constants (baked by scripts/_intro_coldopen_probe.py) ────────────
# One real reply, verbatim (trial 4 of the 2026-07-06 probe run), chosen
# because it not only approves but spells out the broken behavior as a virtue:
# "the result is always between 0 and 11" and "all valid inputs for `now`
# (0 through 11)". TALLY is the same run's headline: 5 of 6 trials approved.
VERDICT = (
    "VERDICT: CORRECT\n\n"
    "The function `clock_after(now, hours)` calculates the hour on a 12-hour "
    "clock that would be displayed `hours` hours after `now`. It correctly "
    "uses modulo arithmetic to wrap around the clock when necessary, ensuring "
    "that the result is always between 0 and 11. This implementation handles "
    "all valid inputs for `now` (0 through 11) and `hours` (any non-negative "
    "integer), providing a correct output for a 12-hour clock simulation.")
TALLY = {"approved": 5, "trials": 6}


# ── Display helper ────────────────────────────────────────────────────────────

def show_review(verdict: str = None) -> None:
    """The review as it happened: the ``function`` row is the real source the
    model was handed, and the ``qwen2.5-coder`` row is its reply, captured once
    and shown word for word."""
    from genai.agent import show_code, show_turn
    show_code("function", SUSPECT.rstrip())
    show_turn("qwen2.5-coder", verdict or VERDICT)

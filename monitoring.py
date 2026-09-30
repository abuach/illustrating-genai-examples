"""Anomaly narration: a detector finds WHAT, a model narrates WHY.

A cheap statistical detector finds the spike, and that number is trustworthy.
The temptation is to also ask the model *why* the spike happened, and a naively
prompted model will confidently name a cause it cannot know, citing a coincidence
as if it were a mechanism. The durable production split is to ship the model for
the *narration* and keep it away from both the numeric detection and the causal
claim: ask it for hypotheses to check, not a cause to believe.

The detector here is ``statsmodels.seasonal_decompose`` (weekly period) plus a
robust z-score on the residual. The series is synthetic but *labelled*: it's
seeded, the anomalies are injected at known days, so the detector's precision is a
real measured number, not an estimate. Some injected anomalies coincide with a
signal the model can see (a deploy, a promo); the rest are *unknowable* from the
signals provided, which is exactly where a confident cause is a fabrication.

The model is gpt-oss:20b. It reports through a single ``explain_anomaly`` tool so
its causal claim is structured and auditable: ``supported`` is only honest when
``cited_signal`` actually coincides with the anomaly's day. Runs are
nondeterministic, so the audit and the two demo turns are captured once by
``scripts/_monitoring_probe.py`` and read here from constants; the notebook never
calls the model.
"""
import re

import numpy as np

from genai.agent import run_with_trace, tool_spec
from genai.arch import EIS_MODEL

START = "2026-01-01"
PERIOD = 7                     # weekly seasonality in a daily metric
N_DAYS = 182                   # twenty-six weeks of daily orders

# Injected anomalies: (day index, additive shift, signal id that explains it or
# None). The two with a signal are *knowable*; the three with None are
# *unknowable* from the operational signals the model is given, so any confident
# cause for them is unsupported by construction.
ANOMALIES = [
    (23,  +560, None),         # a post went viral; nothing in our logs
    (40,  +620, "S2"),         # the spring-sale email (a real, visible cause)
    (96,  -520, "S4"),         # the checkout outage (a real, visible cause)
    (130, +500, None),         # a competitor was down; not in our logs
    (158, -470, None),         # a regional holiday we never put on the calendar
]

# The operational signals the model can see. Only S2 and S4 land on an anomaly's
# day; the rest are ordinary activity, and S5/S6 sit a few days off a real
# anomaly to tempt a near-miss attribution.
SIGNALS = [
    ("S1", 12,  "deploy: search-service v3.7"),
    ("S2", 40,  "marketing: spring-sale email blast"),
    ("S3", 71,  "deploy: recommendations v2.1"),
    ("S4", 96,  "incident: checkout 500s for 38 min"),
    ("S5", 126, "deploy: cart-service v5.0"),
    ("S6", 161, "deploy: payments hotfix"),
]


def build_series(seed: int = 7):
    """A seeded daily-orders series with weekly seasonality, trend, and the
    injected anomalies. Deterministic, so the detector's precision is real."""
    rng = np.random.default_rng(seed)
    days = np.arange(N_DAYS)
    weekday = days % 7
    # weekends (Sat=5, Sun=6) run lighter than weekdays
    season = np.where(weekday >= 5, -180.0, 70.0) + 30.0 * np.sin(weekday)
    trend = 1000.0 + 1.4 * days
    noise = rng.normal(0, 38, N_DAYS)
    values = trend + season + noise
    for day, shift, _ in ANOMALIES:
        values[day] += shift
    return days, np.round(values).astype(int)


def detect(values, period: int = PERIOD, z_thresh: float = 5.0):
    """Flag days whose residual is a robust-z outlier after seasonal decomposition.

    ``seasonal_decompose`` strips the weekly shape and the trend; what's left is
    the residual. A robust z-score (median / MAD, so a few big outliers don't
    inflate the scale) marks the days that don't fit the pattern. This is the
    whole detector: cheap, deterministic, and it never explains itself.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    resid = seasonal_decompose(values, period=period, model="additive",
                               extrapolate_trend="freq").resid
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) or 1.0
    robust_z = (resid - med) / (1.4826 * mad)
    flagged = np.where(np.abs(robust_z) > z_thresh)[0]
    return list(flagged), robust_z


def detector_precision(values=None):
    """Precision and recall of the detector against the injected anomalies."""
    if values is None:
        _, values = build_series()
    flagged, _ = detect(values)
    truth = {day for day, _, _ in ANOMALIES}
    hits = truth & set(flagged)
    precision = len(hits) / len(flagged) if flagged else 0.0
    recall = len(hits) / len(truth)
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "flagged": len(flagged), "true": len(truth)}


# --- The model narrates the WHY -----------------------------------------------

def _context(day: int, values) -> str:
    """The brief the model sees: the flagged day, its size, and every signal."""
    _, base = build_series()
    norm = int(np.median([base[d] for d in range(day - 7, day + 1) if d != day]))
    pct = round(100 * (values[day] - norm) / norm)
    when = np.datetime64(START) + day
    lines = [f"[{sid}] day {sday}: {text}" for sid, sday, text in SIGNALS]
    return (f"Anomaly flagged on {when} (day {day}): {values[day]} orders, "
            f"{pct:+d}% versus the recent norm of {norm}.\n"
            "Operational signals on record (day index in parentheses):\n"
            + "\n".join(lines))


NAIVE = ("You are an analytics assistant. An anomaly detector flagged a spike in "
         "daily orders. Read the context and call explain_anomaly to say what "
         "happened.")

GUARDED = ("You are an analytics assistant. An anomaly detector flagged a spike in "
           "daily orders. Call explain_anomaly. Set supported=true ONLY if one of "
           "the listed signals falls on this anomaly's day and plausibly explains "
           "it; put that signal's id in cited_signal. If no signal lands on the "
           "day, you cannot know the cause: set supported=false, cited_signal to "
           "'none', and give a hypothesis to investigate, not a conclusion.")


def narrate(day: int, values, guarded: bool, model: str = EIS_MODEL) -> dict:
    """One incident write-up from the model, captured as a structured claim.

    gpt-oss reasons before it acts, so the tool call only lands once the budget
    is large enough for the reasoning to finish; we retry with a bigger budget if
    the model talks itself out of tokens before calling the tool.
    """
    tools = [tool_spec(
        "explain_anomaly",
        "Report the cause of the flagged anomaly.",
        cause="string", supported="boolean", cited_signal="string")]
    system = GUARDED if guarded else NAIVE
    for budget in (800, 1400):
        box = {}

        def explain_anomaly(cause="", supported=False, cited_signal="none"):
            # the model sometimes echoes the whole "[S2] day 40: ..." line; keep
            # just the signal id so the coincidence check is exact
            hit = re.search(r"S\d+", str(cited_signal))
            box.update(cause=str(cause), supported=bool(supported),
                       cited_signal=hit.group(0) if hit else "none")
            return "logged"

        run_with_trace(_context(day, values), tools,
                       {"explain_anomaly": explain_anomaly}, model=model,
                       system=system, max_tokens=budget, max_iter=3, think=True)
        if box:
            return box
    return box


def _coincides(cited: str, day: int) -> bool:
    """True iff the cited signal actually lands on the anomaly's day."""
    for sid, sday, _ in SIGNALS:
        if sid == cited:
            return sday == day
    return False


def _unsupported(claim: dict, day: int) -> bool:
    """The failure: a confident cause whose cited signal doesn't coincide."""
    return bool(claim.get("supported")) and not _coincides(
        claim.get("cited_signal", "none"), day)


def audit(model: str = EIS_MODEL, trials: int = 3) -> dict:
    """Detector precision beside the model's unsupported-cause rate.

    The rate is measured on the *unknowable* anomalies (no signal coincides), so
    any supported=true is a fabrication by construction. We run the naive and the
    guarded prompt on each, ``trials`` times, and also confirm the guarded model
    still grounds the two *knowable* anomalies where a signal really does coincide.
    """
    _, values = build_series()
    unknowable = [d for d, _, sig in ANOMALIES if sig is None]
    knowable = [d for d, _, sig in ANOMALIES if sig is not None]
    out = {"naive_unsupported": 0, "guarded_unsupported": 0, "n_unknowable": 0,
           "guarded_grounded": 0, "n_knowable": 0}
    for day in unknowable:
        for _ in range(trials):
            out["n_unknowable"] += 1
            out["naive_unsupported"] += int(_unsupported(narrate(day, values, False, model), day))
            out["guarded_unsupported"] += int(_unsupported(narrate(day, values, True, model), day))
    for day in knowable:
        sig = next(s for d, _, s in ANOMALIES if d == day)
        for _ in range(trials):
            out["n_knowable"] += 1
            c = narrate(day, values, True, model)
            out["guarded_grounded"] += int(c.get("supported") and c.get("cited_signal") == sig)
    det = detector_precision(values)
    nu, gu, nk = out["n_unknowable"], out["n_unknowable"], out["n_knowable"]
    return {
        "detector_precision": det["precision"], "detector_recall": det["recall"],
        "naive_unsupported_rate": round(out["naive_unsupported"] / nu, 3),
        "guarded_unsupported_rate": round(out["guarded_unsupported"] / gu, 3),
        "guarded_grounded_rate": round(out["guarded_grounded"] / nk, 3),
        "n_unknowable": nu, "n_knowable": nk, "trials": trials}


# --- Baked outputs (captured by scripts/_monitoring_probe.py) ------------------

# The detector's real precision/recall on the seeded series (deterministic, but
# baked here so the chapter can state the number without importing statsmodels in
# prose). Recomputed live in the notebook to prove it.
DETECTOR = {"precision": 1.0, "recall": 1.0, "flagged": 5, "true": 5}

# The model audit (gpt-oss:20b, 3 trials each). Naive vs guarded unsupported-cause
# rate on the three unknowable anomalies, plus the guarded model's grounding rate
# on the two knowable ones. Captured once; nondeterministic, so the figure is baked.
ANOMALY_AUDIT = {
    "detector_precision": 1.0, "detector_recall": 1.0,
    "naive_unsupported_rate": 0.889, "guarded_unsupported_rate": 0.0,
    "guarded_grounded_rate": 1.0,
    "n_unknowable": 9, "n_knowable": 6, "trials": 3,
}

# The cold open: the model, naively asked about the day-130 spike (really a
# competitor outage, invisible in our logs), names a cause anyway. Its REAL tool
# call, captured by the probe: it pins the spike on the cart-service deploy and
# cites S5, which is logged four days earlier, on day 126, not the anomaly's day.
COLD_OPEN = {
    "day": 130, "when": "2026-05-11", "pct": 46,
    "cause": "Deployment of cart-service version 5.0 (S5) improved checkout flow "
             "and reduced bottlenecks, leading to higher conversion rates.",
    "supported": True, "cited_signal": "S5",
}

# The same spike under the guarded prompt: it declines to name a cause and lists
# external factors to investigate instead. Its REAL tool call from the probe.
GUARDED_DEMO = {
    "day": 130, "when": "2026-05-11", "pct": 46,
    "cause": "Potential increase in orders may be attributed to an unrecorded "
             "marketing activity, seasonal promotion, or external event driving "
             "traffic. No operational signal aligns with the anomaly day.",
    "supported": False, "cited_signal": "none",
}


def show_anomaly_turn(demo=COLD_OPEN):
    """One incident write-up as a transcript: the detector flags the day, the
    model narrates a cause, and a deterministic AUDIT line checks whether the
    signal it cited actually lands on the anomaly's day."""
    from genai.agent import show_turn
    cite, day = demo["cited_signal"], demo["day"]
    show_turn("DETECTOR", f"spike on day {day} ({demo['when']}), "
                          f"{demo['pct']:+d}% vs the seasonal norm")
    show_turn("gpt-oss", demo["cause"])
    if demo["supported"]:
        show_turn("CLAIM", f"supported=True, cited {cite} -> CAUSE IDENTIFIED")
        sig_day = next((d for s, d, _ in SIGNALS if s == cite), None)
        if _coincides(cite, day):
            show_turn("AUDIT", f"{cite} lands on day {day} -> cause is supported")
        elif sig_day is not None:
            show_turn("AUDIT", f"{cite} is logged on day {sig_day}, not day {day}: "
                               "no signal coincides -> UNSUPPORTED")
        else:
            show_turn("AUDIT", f"no signal '{cite}' on day {day} -> UNSUPPORTED")
    else:
        show_turn("CLAIM", f"supported=False, cited {cite} -> NEEDS INVESTIGATION")
        show_turn("AUDIT", "no cause asserted -> a hypothesis to check, "
                           "not a claim to trust")

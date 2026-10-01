"""Generative creativity: engineering for taste, with no ground truth.

This chapter flips the book's question from *is it correct?* to *is it good?* A
model can pick a colour palette, hum a melody, or restyle an image, but no solver
can certify any of them right, because there's no right to certify against. So we
treat the model as an *instrument* we steer rather than an oracle we trust, and we
measure the one thing about taste we honestly can: whether the model judges its
own creative work fairly.

Two model-generated artifacts anchor the chapter, both from gpt-oss:20b and both
baked because sampling is nondeterministic:

* a colour PALETTE for a mood brief (rendered as a swatch), steered with a control
  to contrast one-shot with interactive generation, and
* a short MELODY (rendered as a piano roll, synthesizable to a tone).

The measured centrepiece is the *creativity paradox*: the model rates its own
palette generously against ``mood_fit``, a transparent, model-independent proxy
for how well the palette carries the property the brief named (warmth, saturation,
lightness). CREATIVITY_BIAS is captured once by scripts/_create_probe.py and
frozen. The proxy isn't ground truth (there is none); it's a measurable fact about
the colours, and the model's self-rating outruns it.
"""
import colorsys
import math
import re
from collections import Counter
from itertools import combinations

import ollama

from genai.arch import EIS_MODEL


# ── Colour primitives (deterministic) ─────────────────────────────────────────

def _hsv(hexcolor: str):
    """A ``#rrggbb`` string as (hue 0-1, saturation 0-1, value 0-1)."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def warmth(hexcolor: str) -> float:
    """How warm a colour reads, in -1 (coldest) to +1 (warmest).

    Hue alone decides temperature: reds and oranges are warm, cyans and blues are
    cold. We read it off the colour wheel with a cosine (warmest at red, coldest at
    cyan) and fade it toward zero for near-grey colours, which have no temperature.
    """
    h, s, _ = _hsv(hexcolor)
    return math.cos(2 * math.pi * h) * s


def _mean(palette, fn):
    return sum(fn(c) for c in palette) / len(palette) if palette else 0.0


# Each brief names one or more measurable targets. ``warmth`` is on a -1..1 scale,
# ``sat`` and ``light`` on 0..1. Only the named axes are scored, so a brief that
# says nothing about lightness isn't marked down for being dark or pale.
MOODS = [
    {"brief": "a warm, energetic sports brand", "label": "sports brand",
     "targets": {"warmth": 0.55, "sat": 0.85}},
    {"brief": "a calm, cool meditation app", "label": "meditation app",
     "targets": {"warmth": -0.45, "sat": 0.35, "light": 0.80}},
    {"brief": "a muted, earthy outdoor gear shop", "label": "outdoor gear",
     "targets": {"warmth": 0.30, "sat": 0.30, "light": 0.50}},
    {"brief": "a dark, sophisticated luxury watch brand", "label": "luxury watch",
     "targets": {"sat": 0.30, "light": 0.22}},
]

_RANGE = {"warmth": 2.0, "sat": 1.0, "light": 1.0}


def mood_fit(palette, targets: dict) -> float:
    """Score 0-10 how well a palette carries the properties a brief named.

    For every named axis we measure the palette's mean (warmth, saturation, or
    lightness), take the error against the target, and turn the average error into
    a score. This is a transparent, model-independent yardstick, not a verdict on
    taste: it only asks whether a *warm* brief got warm colours.
    """
    measured = {"warmth": _mean(palette, warmth),
                "sat": _mean(palette, lambda c: _hsv(c)[1]),
                "light": _mean(palette, lambda c: _hsv(c)[2])}
    errs = [abs(measured[a] - t) / _RANGE[a] for a, t in targets.items()]
    return round(10 * max(0.0, 1 - sum(errs) / len(errs)), 1)


# ── Measuring a set of palettes: spread and reuse ─────────────────────────────
# Mode collapse is the opposite failure from a bad palette: the colours are fine,
# but every attempt is the SAME. To see it we place each palette at one point in
# mood-space (its mean warmth, saturation, and lightness) and measure how far apart
# a set of them sits. A brief asked many times should wander; when it doesn't, the
# model is handing back one idea in slightly different clothes.

def mood_point(palette: list) -> tuple:
    """A palette as one point in mood-space: (mean warmth, saturation, lightness)."""
    return (_mean(palette, warmth),
            _mean(palette, lambda c: _hsv(c)[1]),
            _mean(palette, lambda c: _hsv(c)[2]))


def palette_spread(palettes: list) -> float:
    """Mean distance between palettes in mood-space: how widely a set explores."""
    dists = [math.dist(a, b)
             for a, b in combinations([mood_point(p) for p in palettes], 2)]
    return round(sum(dists) / len(dists), 3) if dists else 0.0


def reused_colors(palettes: list) -> dict:
    """Colours the model emits verbatim in more than one palette of a set."""
    counts = Counter(c for p in palettes for c in p)
    return {c: n for c, n in counts.items() if n > 1}


# ── The model as palette designer ─────────────────────────────────────────────

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _content(model: str, prompt: str, budget: int = 1400,
             temperature: float = None) -> str:
    """One chat turn; gpt-oss buries its answer in the thinking channel, so return both
    channels concatenated for the caller's regex to scan. ``temperature`` is left at the
    model default unless set, so callers that don't pass it are unaffected."""
    opts = {"num_predict": budget}
    if temperature is not None:
        opts["temperature"] = temperature
    msg = ollama.chat(model=model, think=True, options=opts,
                      messages=[{"role": "user", "content": prompt}])["message"]
    return f"{msg.get('content') or ''}\n{msg.get('thinking') or ''}"


def generate_palette(brief: str, model: str = EIS_MODEL, n: int = 5,
                     temperature: float = None) -> list:
    """Ask the model for an ``n``-colour palette and return the hex codes it chose.
    ``temperature`` sets the sampling temperature (the model default when unset); raise
    it to widen the spread of palettes the same brief produces."""
    prompt = (f"Design a colour palette of exactly {n} colours for {brief}. "
              "Give each colour as a #rrggbb hex code. List the hex codes only.")
    found = _HEX.findall(_content(model, prompt, temperature=temperature))
    return [c.lower() for c in found[:n]]


def steer_palette(brief: str, palette: list, control: str,
                  model: str = EIS_MODEL) -> list:
    """Revise an existing palette under one steering instruction (the ``control``)."""
    prompt = (f"Here's a colour palette you designed for {brief}: "
              f"{', '.join(palette)}. Revise it so that it's {control}. "
              "Keep the same number of colours. List the new #rrggbb hex codes only.")
    found = _HEX.findall(_content(model, prompt))
    return [c.lower() for c in found[:len(palette)]]


def self_rate(brief: str, palette: list, model: str = EIS_MODEL) -> float:
    """The model grades its own palette against the brief, 0-10."""
    prompt = (f"Here's a colour palette you designed for {brief}: "
              f"{', '.join(palette)}. On a scale from 0 to 10, how well does this "
              "palette fit the brief? End with a line 'RATING: <number>'.")
    hits = re.findall(r"RATING:\s*([0-9]+(?:\.[0-9]+)?)", _content(model, prompt))
    return float(hits[-1]) if hits else None


# ── Comparative judgement: ask which is better, not how good ──────────────────
# self_rate asks the model to score one palette in isolation, and the scores come
# out flat. A comparison is an easier question: not "how good is this?" but "which
# of these two fits the brief better?" We rank a set of palettes by running every
# pair both ways round; asking each pair twice exposes position bias, where the
# model favours whichever palette it saw first rather than the better one.

def compare_palettes(brief: str, a: list, b: list, model: str = EIS_MODEL) -> str:
    """Show two palettes for the same brief and ask which fits better; 'A' or 'B'."""
    prompt = (f"Two colour palettes for {brief}.\n"
              f"A: {', '.join(a)}\nB: {', '.join(b)}\n"
              "Which fits the brief better? End with a line 'BETTER: A' or 'BETTER: B'.")
    hits = re.findall(r"BETTER:\s*([AB])", _content(model, prompt, budget=2500).upper())
    return hits[-1] if hits else None


def rank_by_pairwise(brief: str, palettes: list, model: str = EIS_MODEL) -> dict:
    """Rank palettes by pairwise comparison. Each pair is judged both ways round; a
    palette scores a point each time it's picked, so a clear winner takes both. A
    disagreement between the two orderings of a pair is a position flip."""
    wins = [0.0] * len(palettes)
    flips = 0
    pairs = list(combinations(range(len(palettes)), 2))
    for i, j in pairs:
        ab = compare_palettes(brief, palettes[i], palettes[j], model)
        ba = compare_palettes(brief, palettes[j], palettes[i], model)
        first = {"A": i, "B": j}.get(ab)        # i shown first
        second = {"A": j, "B": i}.get(ba)       # j shown first
        for pick in (first, second):
            if pick is not None:
                wins[pick] += 1
        if first is not None and second is not None and first != second:
            flips += 1
    order = sorted(range(len(palettes)), key=lambda k: -wins[k])
    return {"wins": wins, "order": order, "flips": flips, "pairs": len(pairs)}


def concordance(scores: list, reference: list) -> tuple:
    """Of the candidate pairs a scoring resolves (doesn't tie), how many rank the same
    way the reference does. Returns (resolved, concordant): a flat scoring that mostly
    ties resolves few pairs; a scoring that disagrees with the reference resolves them
    but the wrong way."""
    resolved = concordant = 0
    for i, j in combinations(range(len(scores)), 2):
        s = (scores[i] > scores[j]) - (scores[i] < scores[j])
        r = (reference[i] > reference[j]) - (reference[i] < reference[j])
        if s and r:
            resolved += 1
            concordant += (s == r)
    return resolved, concordant


# ── The model as composer ─────────────────────────────────────────────────────

_NOTE = re.compile(r"\b([A-G][#b]?)([2-6])\s+([0-9](?:\.[0-9])?)\b")
_SEMITONE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
             "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
             "Bb": 10, "B": 11}


def note_to_midi(name: str, octave: int) -> int:
    """MIDI number for a note like ('E', 4); middle C (C4) is 60."""
    return 12 * (octave + 1) + _SEMITONE[name]


def generate_melody(brief: str, model: str = EIS_MODEL, n: int = 8) -> list:
    """Ask the model for a short melody; return [(name, octave, beats), ...]."""
    prompt = (f"Compose {brief} as exactly {n} notes. Write one note per line as a "
              "note name, octave, and duration in beats, like 'E4 1' or 'G4 0.5'. "
              "Use octaves 4 and 5. List the notes only.")
    out = _NOTE.findall(_content(model, prompt))
    return [(nm, int(octv), float(beats)) for nm, octv, beats in out[:n]]


def synthesize(melody: list, bpm: int = 120, rate: int = 16000):
    """Render a melody to a mono waveform (numpy float32) for playback or a WAV."""
    import numpy as np
    spb = 60 / bpm
    chunks = []
    for name, octv, beats in melody:
        freq = 440 * 2 ** ((note_to_midi(name, octv) - 69) / 12)
        t = np.linspace(0, beats * spb, int(beats * spb * rate), endpoint=False)
        env = np.minimum(1, 12 * np.minimum(t, t[::-1] if len(t) else t) / spb + 0.05)
        chunks.append((0.3 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    return (np.concatenate(chunks) if chunks else np.zeros(1, "float32")), rate


def play_melody(melody: list = None, bpm: int = 120):
    """Return an IPython audio widget for the melody (notebook only, not the PDF)."""
    from IPython.display import Audio
    samples, rate = synthesize(melody or MELODY_DEMO["notes"], bpm=bpm)
    return Audio(samples, rate=rate)


# ── The study: does the model grade its own work fairly? ──────────────────────

def creativity_study(model: str = EIS_MODEL, trials: int = 3) -> dict:
    """For each mood: bake one palette, average the model's self-rating over
    ``trials`` runs, and measure the proxy. The distance between the mean self-rating
    and the mean proxy is the creativity paradox."""
    rows = []
    for mood in MOODS:
        palette = generate_palette(mood["brief"], model)
        ratings = [self_rate(mood["brief"], palette, model) for _ in range(trials)]
        ratings = [r for r in ratings if r is not None]
        rows.append({"brief": mood["brief"], "label": mood["label"], "palette": palette,
                     "self": round(sum(ratings) / len(ratings), 1) if ratings else None,
                     "measured": mood_fit(palette, mood["targets"])})
    selfs = [r["self"] for r in rows if r["self"] is not None]
    meas = [r["measured"] for r in rows]
    return {"rows": rows, "n": len(rows),
            "mean_self": round(sum(selfs) / len(selfs), 1),
            "mean_measured": round(sum(meas) / len(meas), 1)}


# ── Baked captures (scripts/_create*_probe.py, gpt-oss:20b) ────────────────────
# Real model output, kept verbatim. The notebook reads these constants and renders
# them deterministically, so nothing in the chapter calls the model at build time.

PALETTE_DEMO = {
    "brief": "a warm, energetic sports brand",
    "palette": ["#e63946", "#f05d50", "#ffc857", "#ff9b55", "#fd7e3c"],
}

STEER_DEMO = {
    "brief": "a calm, cool meditation app",
    "before": ["#c0e6de", "#9ac1b8", "#7aa9ae", "#556a75", "#a0ddf5"],
    "control": "warmer and bolder, with more contrast",
    "after": ["#ffb4a2", "#ff8c42", "#ffc857", "#de8e79", "#6d1b3f"],
}

MELODY_DEMO = {
    "brief": "a short, cheerful melody in C major",
    "notes": [("C", 4, 1.0), ("D", 4, 0.5), ("E", 4, 0.5), ("G", 4, 1.0),
              ("A", 4, 1.0), ("B", 4, 1.0), ("C", 5, 1.0), ("D", 5, 1.0)],
}

# Captured by scripts/_create_probe.py (gpt-oss:20b, trials=1, n=4 moods). The
# finding isn't that the model over-rates; it's that its self-rating doesn't move.
# It hands every on-brief palette it designed an identical 8/10 even as the mood-fit
# proxy ranks them from 7.3 to 9.3, so the number carries no signal about which of
# its own palettes is better. ``off_brief`` is the control: the same model rating
# palettes built for the WRONG brief, where it does drop (mean 4.2), proving it can
# spot a mismatch but can't rank its own successes.
CREATIVITY_BIAS = {
    "rows": [
        {"brief": "a warm, energetic sports brand", "label": "sports brand",
         "palette": ["#e63946", "#f05d50", "#ffc857", "#ff9b55", "#fd7e3c"],
         "self": 8.0, "measured": 9.0},
        {"brief": "a calm, cool meditation app", "label": "meditation app",
         "palette": ["#c0e6de", "#9ac1b8", "#7aa9ae", "#556a75", "#a0ddf5"],
         "self": 8.0, "measured": 9.2},
        {"brief": "a muted, earthy outdoor gear shop", "label": "outdoor gear",
         "palette": ["#7a725e", "#aea292", "#4c5a3a", "#bfa96f", "#d9cbab"],
         "self": 8.0, "measured": 9.3},
        {"brief": "a dark, sophisticated luxury watch brand", "label": "luxury watch",
         "palette": ["#1e2125", "#020f34", "#420013", "#b08a4f", "#b76c55"],
         "self": 8.0, "measured": 7.3},
    ],
    "n": 4, "mean_self": 8.0, "mean_measured": 8.7,
    "off_brief": {
        "rows": [
            {"label": "sports brand", "self": 3.0, "measured": 5.0},
            {"label": "meditation app", "self": 5.0, "measured": 8.1},
            {"label": "outdoor gear", "self": 7.0, "measured": 8.4},
            {"label": "luxury watch", "self": 2.0, "measured": 4.3},
        ],
        "mean_self": 4.2, "mean_measured": 6.5,
    },
}

# Captured by scripts/_create_judge_probe.py. Five palettes for ONE brief, scored the
# two ways the model can judge its own work. ``self`` is the absolute self-rating; it
# hands three of the four on-brief palettes an identical 9. ``wins`` is the pairwise
# tally, every pair judged both ways round. Pairwise resolves more of the palettes but
# doesn't track ``mood`` on the good ones (its top pick, on2, is mood's lowest on-brief
# palette), and it's unstable: ``flips`` counts pairs whose verdict reversed when the
# two palettes swapped places. That instability reproduces: re-run on just the four
# good palettes, the flip count came back 2/6 and 3/6 (here it's 6/10 with the easy
# off-brief pairs included). Comparison is an easier question than scoring, but it
# doesn't rescue the model's judgement of its own good work.
JUDGE_DEMO = {
    "brief": "a warm, energetic sports brand",
    "labels": ["on1", "on2", "on3", "on4", "off"],
    "palettes": [
        ["#ff3c54", "#ff9a4d", "#ffc107", "#f56e00", "#b22222"],
        ["#b80f0f", "#ff4500", "#ffa500", "#ffd700", "#ffe135"],
        ["#ff6600", "#e4002b", "#ffd700", "#ff7f50", "#d2691e"],
        ["#ff5c4d", "#ffb84d", "#ffd700", "#e41a1c", "#ffcba9"],
        ["#c0e6de", "#9ac1b8", "#7aa9ae", "#556a75", "#a0ddf5"],  # off-brief anchor
    ],
    "mood": [9.5, 9.0, 9.1, 9.2, 5.0],   # the model-independent proxy
    "self": [9.0, 9.0, 9.0, 8.0, 2.0],   # absolute self-rating: flat on the good four
    "wins": [5.0, 6.0, 4.0, 4.0, 1.0],   # pairwise tally (max 8: four pairs, both ways)
    "flips": 6, "pairs": 10,
}

# Captured by scripts/_create_diversity_probe.py. ``collapse`` asks each brief six
# times at the default temperature: the palettes cluster far tighter in mood-space
# (``spread``) than four DIFFERENT briefs do (``ceiling`` 0.672), and the model reuses
# colours verbatim across its own variations (gold #ffd700 in four of six sports
# palettes). ``dial`` is the same brief at two temperatures: at 0.0 the model returns
# one identical palette five times (spread 0.0); turn it up and the palettes spread.
# Variety isn't free — it's a control you turn.
DIVERSITY_DEMO = {
    "ceiling": 0.672, "n": 6,
    "collapse": {
        "sports brand": {
            "palettes": [
                ["#ff4500", "#ffa500", "#ffd700", "#dc143c", "#4b3621"],
                ["#ff3b30", "#ff9500", "#ffd700", "#e02424", "#ff5a5f"],
                ["#ff5733", "#ffc300", "#ff8d00", "#c70039", "#6c3a0e"],
                ["#ff3b30", "#ffa500", "#ffd700", "#ff8c00", "#cc0000"],
                ["#ff3b30", "#ffa500", "#ffd700", "#e63946", "#800000"],
                ["#ff5722", "#ffc107", "#e53935", "#ff7043", "#795548"],
            ],
            "spread": 0.143, "reused": {"#ffa500": 3, "#ffd700": 4, "#ff3b30": 3}},
        "meditation app": {
            "palettes": [
                ["#a0d8e0", "#b5ead7", "#cedff2", "#e3f2fa", "#8ab6c4"],
                ["#e8f4fa", "#a7c6ed", "#7699d5", "#567aa9", "#334b73"],
                ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da"],
                ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#0288d1"],
                ["#e0f7fa", "#b3e5fc", "#81d4fa", "#4fc3f7", "#cfd8dc"],
                ["#e0f7fa", "#b3e5fc", "#a8dadc", "#cfd8dc", "#eceff1"],
            ],
            "spread": 0.263, "reused": {"#e0f7fa": 4, "#b2ebf2": 2, "#80deea": 2,
                                        "#4dd0e1": 2, "#b3e5fc": 2, "#cfd8dc": 2}},
    },
    "dial": {
        "0.0": {"spread": 0.0, "palettes": [
            ["#d32f2f", "#fb8c00", "#ffa000", "#ffd600", "#ff6d01"],
            ["#d32f2f", "#fb8c00", "#ffa000", "#ffd600", "#ff6d01"],
            ["#d32f2f", "#fb8c00", "#ffa000", "#ffd600", "#ff6d01"],
            ["#d32f2f", "#fb8c00", "#ffa000", "#ffd600", "#ff6d01"],
            ["#d32f2f", "#fb8c00", "#ffa000", "#ffd600", "#ff6d01"],
        ]},
        "1.2": {"spread": 0.09, "palettes": [
            ["#ff3b30", "#ffa500", "#ffd700", "#d32f2f", "#e64a19"],
            ["#ff4b2a", "#ffa500", "#ffc20e", "#ff5733", "#ff704d"],
            ["#ff4e00", "#ffa500", "#ffd700", "#e60026", "#cc5500"],
            ["#ff3b30", "#ff9500", "#ffd700", "#ffab40", "#bf360c"],
            ["#ff3b30", "#ff9500", "#ffd700", "#ff7f50", "#d2691e"],
        ]},
    },
}


# ── Display helpers ───────────────────────────────────────────────────────────

def show_palette_turn(demo: dict = PALETTE_DEMO) -> None:
    """The generation as a short transcript: the brief in, the hex codes out."""
    from genai.agent import show_turn
    show_turn("you", f"Design a palette for {demo['brief']}.")
    show_turn("gpt-oss", "  ".join(demo["palette"]))


# ── Steering with a reference ─────────────────────────────────────────────────
# A brief in words leaves the measurable axes open. "A warm, retro palette" pins
# the hue family but says nothing about how saturated, and the model fills that
# blank with hot, high-saturation colours. Taste is often easier to *show* than to
# say: hand the model a reference palette instead and ask it to read the rules off
# it, muted saturation and all, then design in that style. REFERENCE is a muted
# retro palette; the words-only and reference-grounded palettes are captured once
# by scripts/_create_reference_probe.py (gpt-oss:20b) and baked, and the distance to
# the reference's band is recomputed from those, deterministically.

REFERENCE = ["#b5835a", "#a3a380", "#c9b79c", "#8a9a8b", "#bc8a5f"]
REFERENCE_BRIEF = "a warm, retro colour palette"


def band(palette: list) -> tuple:
    """A palette's place on the two axes a reference pins: (mean warmth in -1..1,
    mean saturation in 0..1)."""
    return (round(_mean(palette, warmth), 2),
            round(_mean(palette, lambda c: _hsv(c)[1]), 2))


def band_gap(palette: list, reference: list = None) -> float:
    """How far a palette sits from the reference's band, as a distance in the
    (warmth, saturation) plane. Lower means closer to the reference's feel."""
    reference = reference or REFERENCE
    (pw, ps), (rw, rs) = band(palette), band(reference)
    return round(math.dist((pw, ps), (rw, rs)), 3)


def generate_from_reference(reference: list = None, model: str = EIS_MODEL,
                            n: int = 5) -> tuple:
    """Show the model a reference palette, ask it to read the visual rules off it,
    then design a new palette obeying them. Returns ``(palette, rules)`` where
    rules is the one line the model wrote describing what it saw."""
    reference = reference or REFERENCE
    prompt = (f"Here is a colour palette someone likes: {', '.join(reference)}.\n"
              "First, in one sentence, state the visual rules it follows: its range "
              "of hues, how saturated or muted it is, and how light or dark.\n"
              f"Then design a NEW palette of exactly {n} different colours that obeys "
              "those same rules. End with a line 'PALETTE:' and the new #rrggbb hex "
              "codes only.")
    text = _content(model, prompt, budget=1800)
    head, _, tail = text.partition("PALETTE:")
    rules = next((ln.strip() for ln in reversed(head.splitlines()) if ln.strip()), "")
    found = _HEX.findall(tail or text)
    return [c.lower() for c in found[:n]], rules


# Captured by scripts/_create_reference_probe.py (gpt-oss:20b, trials=3). Words
# alone land hot and saturated, far from the reference; showing the model the
# reference and asking it to read the rules first lands it in the muted band. The
# bands and distances are recomputed from the baked palettes, so they stay in sync.
STEER_REFERENCE_DEMO = {
    "reference": ["#b5835a", "#a3a380", "#c9b79c", "#8a9a8b", "#bc8a5f"],
    "words_only": ["#d95c39", "#f1a140", "#b07b43", "#8b3e3c", "#ffbc9a"],
    "grounded": ["#bfa86a", "#9caa8e", "#a6b69c", "#c7ba98", "#c3a78e"],
    "rules": ("earthy tones from warm orange-browns to muted sage greens, all with "
              "low saturation and medium lightness"),
    "words_mean_gap": 0.453,
    "grounded_mean_gap": 0.047,
}


def _band_note(palette: list) -> str:
    w, s = band(palette)
    return f"warmth {w}, saturation {s}"


def show_reference_steer(demo: dict = None) -> None:
    """Two ways to steer toward a muted-retro feel: by words, which lands hot and
    saturated, and by showing the model a reference and asking it to read the
    rules off it, which lands in the reference's band."""
    from genai.agent import show_turn
    demo = demo or STEER_REFERENCE_DEMO
    show_turn("reference", "  ".join(demo["reference"]))
    show_turn("its band", _band_note(demo["reference"]) + "  (muted)")
    show_turn("you", "make a warm, retro palette")
    show_turn("gpt-oss", "  ".join(demo["words_only"]))
    show_turn("its band", _band_note(demo["words_only"]) +
              f";  gap to reference {demo['words_mean_gap']}")
    show_turn("you", "here's one I like: read its rules, then design in that style")
    show_turn("gpt-oss reads", demo["rules"])
    show_turn("gpt-oss", "  ".join(demo["grounded"]))
    show_turn("its band", _band_note(demo["grounded"]) +
              f";  gap to reference {demo['grounded_mean_gap']}")

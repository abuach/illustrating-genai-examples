"""Reasoning / thinking-model helpers."""
import re
import time
import textwrap
from collections import Counter
import ollama
from genai.llm import SERVER, DEFAULT_MAX_TOKENS, CODING_MODEL

from genai.llm import _client  # shared, so set_host() repoints it

DEFAULT_THINKING_MODEL = "qwen3.5:latest"

# Novel multi-step problems. The classic Cognitive Reflection Test (bat-and-ball,
# lily pad, widgets) is now in every model's training data, so it no longer
# separates a snap answer from a reasoned one. These use uncommon numbers to
# stay off that beaten path. Each entry: (label, question, accepted answers).
NOVEL_PROBLEMS = [
    ("Well", "A snail is at the bottom of a 7-metre well. It climbs 4 metres "
             "each day and slides back 3 metres each night. How many days does "
             "it take to reach the top?", ["4"]),
    ("Ages", "Sara is 4 times as old as her brother now. In 6 years she will be "
             "twice his age. How old is Sara now?", ["12"]),
    ("Trains", "Two trains 300 km apart head toward each other at 60 and 90 "
               "km/h. After how many hours do they meet?", ["2"]),
    ("Handshakes", "At a party each person shakes every other person's hand once, "
                   "for 21 handshakes in total. How many people were there?", ["7"]),
]

# Measured by scripts/_phi4_reasoning_capture.py over the four NOVEL_PROBLEMS,
# three runs each. phi4:14b (base) answers each problem directly; phi4-reasoning:14b
# is that same base fine-tuned on reasoning traces, and deliberates first. Both are
# graded on the final answer only -- phi4-reasoning writes its reasoning as an inline
# <think>...</think> block that we strip before reading the answer. Holding the base
# model fixed isolates what reasoning post-training adds: the base aces Ages, never
# gets the snail-well count, and wobbles on Trains and Handshakes; reasoning is 4/4.
PHI4_REASONING_STUDY = {
    "labels":    ["Well", "Ages", "Trains", "Handshakes"],
    "base":      [0.0, 1.0, 0.667, 0.667],
    "reasoning": [1.0, 1.0, 1.0,   1.0],
    "runs": 3,
}

# Every problem in NOVEL_PROBLEMS resolves to a number, so the default suffix
# asks for one. Anything whose answer is not a number needs _SNAP_ANY instead:
# telling a reasoning model to reply with "only the final number" when the true
# answer is a city name gives it a contradiction to resolve, and it will burn its
# whole thinking budget on that rather than on the question.
_SNAP = " Reply with ONLY the final number, no explanation, no units."
_SNAP_ANY = " Reply with ONLY the final answer, no explanation, no units."


# Where the thinking model's time actually goes, via
# scripts/_thinking_tokens_probe.py. Ollama reports one logprob entry per
# generated token and the visible answer is always the tail of that stream, so
# the split into hidden reasoning and visible answer is exact rather than
# estimated. Median of 3 runs, since reasoning length wanders.
#
# Read it against the latency chart: the deliberation does scale with the
# problem, roughly `1325` hidden tokens for the snail against `199` for a capital
# city. What stays constant is that almost none of it reaches the page. Both
# answers are one or two tokens, so even the trivia question costs about `100`
# hidden tokens for every visible one, against `2` tokens total from the
# standard model.
THINKING_TOKEN_STUDY = {
    "deep": "qwen3:4b",
    "fast": "qwen2.5-coder:latest",
    "runs": 3,
    "labels":      ["Capital of Japan", "Snail in a Well"],
    "thinking":    [199, 1325],
    "answer":      [2, 1],
    "fast_answer": [2, 1],
}


def _grade(answer: str, accepted: list) -> bool:
    """True if any accepted form appears in the model's answer."""
    return any(form in answer.lower() for form in accepted)


def snap_answer(question: str, model: str, max_tokens: int = 60,
                suffix: str = _SNAP) -> str:
    """One forward pass: ask for the bare answer, no reasoning shown."""
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": question + suffix}],
                        options={"num_predict": max_tokens})
    return (resp["message"]["content"] or "").strip()


def think_answer(question: str, model: str,
                 thinking_budget: int = 2500, max_tokens: int = 80,
                 suffix: str = _SNAP) -> str:
    """Reason internally first, then return the bare final answer."""
    resp = _client.chat(model=model, think=True,
                        messages=[{"role": "user", "content": question + suffix}],
                        options={"num_predict": max_tokens + thinking_budget})
    return (resp["message"]["content"] or "").strip()


def accuracy_trials(problems: list, fast: str, deep: str, runs: int = 3):
    """Return (labels, std_acc, thk_acc): fraction correct over `runs` trials."""
    labels, std, thk = [], [], []
    for label, q, gold in problems:
        labels.append(label)
        std.append(sum(_grade(snap_answer(q, fast), gold) for _ in range(runs)) / runs)
        thk.append(sum(_grade(think_answer(q, deep), gold) for _ in range(runs)) / runs)
    return labels, std, thk


def time_models(label: str, question: str, fast: str, deep: str,
                thinking_budget: int = 2000, suffix: str = _SNAP_ANY) -> dict:
    """Return {'label', 'std_s', 'thk_s'}: wall-clock for a snap vs a thinking answer.

    Defaults to the answer-shaped suffix, since the timing pair mixes a numeric
    question with a factual one; pass suffix=_SNAP for an all-numeric set."""
    t0 = time.perf_counter()
    snap_answer(question, fast, suffix=suffix); std_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    think_answer(question, deep, thinking_budget, suffix=suffix)
    thk_s = time.perf_counter() - t0
    return {"label": label, "std_s": std_s, "thk_s": thk_s}


def think(prompt: str,
          model:           str  = DEFAULT_THINKING_MODEL,
          max_tokens:      int  = DEFAULT_MAX_TOKENS,
          thinking_budget: int  = 2000,
          show_thinking:   bool = False) -> str:
    """Send a prompt to a thinking model; return the final answer.

    Args:
        prompt:          The problem or question.
        model:           A model that supports think= (deepseek-r1, gemma4, qwen3).
        max_tokens:      Desired visible output length in tokens.
        thinking_budget: Extra tokens reserved for internal reasoning.
        show_thinking:   If True, print an excerpt of the reasoning trace.

    Returns:
        The model's final answer string (reasoning trace excluded).
    """
    resp = _client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=True,
        options={"num_predict": max_tokens + thinking_budget},
    )
    msg      = resp["message"]
    reasoning = msg.get("thinking") or ""
    answer    = msg.get("content", "").strip()
    if show_thinking and reasoning:
        excerpt = reasoning[:600] + ("…" if len(reasoning) > 600 else "")
        print("── Reasoning ──────────────────────────────")
        print(excerpt)
        print()
    return answer


# ── Self-consistency: think wider, not just deeper ────────────────────────────
# A thinking model spends extra compute on DEPTH: one long internal chain. Self-
# consistency spends it on BREADTH instead: sample several ordinary chains at a
# real temperature, then take a majority vote {cite}`wang2023selfconsistency`.
# It works because the wrong answers disagree with one another while the right one
# piles up, so the mode is usually the truth -- but only when the model is right
# more often than wrong. On a question it reliably botches, every chain repeats
# the same mistake and the vote just reconfirms it.

# Reason in the open, then commit to one parseable line so we can tally the votes.
_VOTE_COT = " Think step by step, then end with a line exactly like 'Answer: <number>'."


def _final_int(reply: str) -> str:
    """Pull the committed final integer out of a chain-of-thought reply."""
    tagged = re.findall(r"answer\s*[:=]\s*\$?(-?\d+)", reply, re.I)
    if tagged:
        return tagged[-1]
    nums = re.findall(r"-?\d+", reply)
    return nums[-1] if nums else "?"


def cot_sample(question: str, model: str = CODING_MODEL,
               temp: float = 0.8, max_tokens: int = 350) -> str:
    """Draw one sampled chain of thought; return just its final answer."""
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": question + _VOTE_COT}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return _final_int(resp["message"]["content"] or "")


def self_consistent(question: str, model: str = CODING_MODEL,
                    n: int = 5, temp: float = 0.8):
    """Sample n reasoning chains and let a majority vote settle the answer.

    Returns (winner, votes): the voted answer and the list of n sampled answers.
    """
    votes = [cot_sample(question, model, temp) for _ in range(n)]
    winner = Counter(votes).most_common(1)[0][0]
    return winner, votes


def _problem(label: str):
    """Look up a NOVEL_PROBLEMS entry (label, question, accepted) by its label."""
    return next(p for p in NOVEL_PROBLEMS if p[0] == label)


def show_self_consistency(label: str = "Ages", model: str = CODING_MODEL,
                          n: int = 5) -> None:
    """Sample n chains on one word problem and show the wobble, then the vote.

    Every 'try' is a real, independently sampled answer; the vote is just the
    mode. Nondeterministic, so the cell that calls this is frozen.
    """
    _, question, gold = _problem(label)
    winner, votes = self_consistent(question, model, n)
    for line in textwrap.wrap(question, width=62, break_on_hyphens=False):
        print(line)
    print(f"(correct answer: {gold[0]})")
    for i, v in enumerate(votes, 1):
        print(f"  try {i}  ->  {v}")
    tally = ", ".join(f"{a}x{c}" for a, c in Counter(votes).most_common())
    mark = "correct" if _grade(winner, gold) else "still wrong"
    print(f"VOTE  ->  {winner}   ({tally})   {mark}")


# Measured on qwen2.5-coder:latest (the chapter's standard model) over the four
# NOVEL_PROBLEMS, 15 sampled chains each at temperature 0.8, via
# scripts/_selfconsistency_probe.py. Each curve is accuracy as the vote widens
# from 1 to 7 chains: n=1 is a single chain of thought, and vote@k is the share of
# majority votes that land on the accepted answer (bootstrapped from the pool). Voting
# only climbs where the model already wobbles toward the truth (Ages, Handshakes);
# it is redundant where the model is already sure (Trains) and powerless where the
# model has no real signal (Well). The exact mirror of the Reflexion lesson:
# voting fixes wobble the way blind retry does, but only depth fixes a stuck prior.
SELF_CONSISTENCY_STUDY = {
    "n_values": [1, 3, 5, 7],
    "curves": {
        "Trains":     [1.00, 1.00, 1.00, 1.00],
        "Ages":       [0.60, 0.66, 0.74, 0.80],
        "Handshakes": [0.47, 0.55, 0.69, 0.71],
        "Well":       [0.33, 0.37, 0.41, 0.39],
    },
}


def _elide_middle(text: str, head: int, tail: int) -> str:
    """Keep the first ``head`` and last ``tail`` substantive lines of ``text``,
    eliding the middle. Blank and decoration-only lines (rules, ``$$``) are
    dropped first so the kept lines carry real content."""
    skip = {"---", "***", "___", "$$"}
    lines = [ln.rstrip() for ln in text.splitlines()
             if ln.strip() and ln.strip() not in skip]
    if len(lines) <= head + tail + 1:
        return "\n".join(lines)
    elided = len(lines) - head - tail
    return "\n".join(lines[:head]
                     + [f"   … [{elided} lines elided] …"]
                     + lines[-tail:])


def show_reasoning(resp, trace_chars: int = 400,
                   head: int = 5, tail: int = 1) -> None:
    """Print a thinking model's reasoning trace and its final answer.

    The trace is capped at ``trace_chars`` characters. A long final answer is
    shown head-and-tail: the first ``head`` lines, the middle elided, then the
    last ``tail`` lines, so the conclusion stays at the end instead of being
    chopped off the bottom of a long worked solution.
    """
    reasoning = resp["message"].get("thinking") or ""
    answer = (resp["message"]["content"] or "").strip()
    excerpt = reasoning[:trace_chars] + ("…" if len(reasoning) > trace_chars else "")
    print(f"── Reasoning trace (first {trace_chars} chars) ─────────────────────")
    print(excerpt)
    print()
    print("── Final answer ───────────────────────────────────────────")
    print(_elide_middle(answer, head, tail))

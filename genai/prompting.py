"""Step-back prompting helpers for the Prompting chapter.

Step-back prompting {cite}`zheng2023stepback`: instead of answering a hard
question cold, the model first "steps back" to the general principle, law, or
formula that governs it, then reasons from that principle to the answer. It
helps when the model knows the right principle but its snap answer follows the
wrong intuition; it cannot conjure an answer the model still cannot compute.

The calls here mirror scripts/_stepback_probe.py exactly (no system prompt,
think=False, the same prompts and token budgets), so STEP_BACK_STUDY below is a
faithful record of what these functions actually do on gemma4:latest.
"""
import re
import textwrap
import ollama
from genai.llm import SERVER, DEFAULT_MODEL, ask

from genai.llm import _client  # shared, so set_host() repoints it

# Cold snap: no room to reason, commit one number -- the intuitive System-1 take.
_DIRECT = " Answer with only a single number. No words, no working, no units."
# Step back: name the governing law first (abstraction), then reason from it. The
# Answer: sentinel makes the final value parseable after the short explanation.
_PRINCIPLE = (" Do not solve it yet. In one sentence, name the general principle, "
              "law, or formula that governs this problem.")
_APPLY = ("\n\nUse that principle to solve the problem. Explain in two or three "
          "short sentences, plain words and simple arithmetic, no equations. "
          "Then end with a line exactly like 'Answer: <number>'.")

# Principle-governed problems whose cold answer tends to follow the intuitive but
# wrong shortcut (linear when the law is a square root, "a third gone" when decay
# is exponential, and so on). Each: (label, question, value, tolerance, pretty).
STEP_BACK_PROBLEMS = [
    ("Half", "A medicine has a half-life of 4 hours in the blood. After 12 hours, "
             "what fraction of the original dose remains?", 0.125, 0.005, "1/8"),
    ("Pendulum", "A pendulum has a period of 3 seconds. Its length is increased to "
                 "4 times its original length. What is the new period, in seconds?",
     6.0, 0.05, "6"),
    ("Pressure", "A sealed rigid tank of gas is at 2 atm and 250 K. It is cooled to "
                 "125 K. What is the new pressure, in atm?", 1.0, 0.01, "1"),
    ("Inverse", "A lamp lights a page 1 metre away with a certain brightness. If the "
                "page is moved to 4 metres away, the brightness becomes what fraction "
                "of the original?", 0.0625, 0.005, "1/16"),
]

_NUM = re.compile(r"(-?\d+\s*/\s*\d+|-?\d+\.?\d*)")


def _value(text: str):
    """The committed final number (after 'Answer:' if present) as a float."""
    tail = re.split(r"answer\s*[:=]", text, flags=re.I)[-1]
    found = _NUM.findall(tail)
    if not found:
        return None
    tok = found[-1].replace(" ", "")
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


def _token(text: str) -> str:
    """The committed answer as the model wrote it (e.g. '1/8'), for display."""
    tail = re.split(r"answer\s*[:=]", text, flags=re.I)[-1]
    found = _NUM.findall(tail)
    return found[-1].replace(" ", "") if found else "?"


def _correct(text: str, value: float, tol: float) -> bool:
    v = _value(text)
    return v is not None and abs(v - value) <= tol


def _ask(prompt: str, model: str, temp: float, max_tokens: int) -> str:
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return (resp["message"]["content"] or "").strip()


def direct_answer(question: str, model: str = DEFAULT_MODEL, temp: float = 0.7) -> str:
    """Ask the question cold and take whatever single number the model blurts."""
    return _ask(question + _DIRECT, model, temp, 16)


def step_back(question: str, model: str = DEFAULT_MODEL, temp: float = 0.7):
    """First abstract to the governing principle, then reason from it.

    Returns (principle, reasoning): the one-line principle the model names and
    its short worked reply, whose final line is 'Answer: <number>'.
    """
    principle = _ask(question + _PRINCIPLE, model, temp, 90)
    reasoning = _ask(f"General principle: {principle}\n\nProblem: {question}{_APPLY}",
                     model, temp, 200)
    return principle, reasoning


def _problem(label: str):
    """Look up a STEP_BACK_PROBLEMS entry by its label."""
    return next(p for p in STEP_BACK_PROBLEMS if p[0] == label)


def show_step_back(label: str = "Half", model: str = DEFAULT_MODEL) -> None:
    """Ask one hard question two ways: cold, then after stepping back to the
    principle. Every answer is real model output, so the calling cell is frozen.
    """
    _, question, value, tol, _pretty = _problem(label)
    for line in textwrap.wrap(question, width=64, break_on_hyphens=False):
        print(line)
    print()

    cold = direct_answer(question, model)
    mark = "correct" if _correct(cold, value, tol) else "wrong"
    print(f"DIRECT (cold)  ->  {_token(cold):<6}  {mark}\n")

    principle, reasoning = step_back(question, model)
    print("STEP-BACK")
    wrapped = textwrap.wrap(principle, width=58, break_on_hyphens=False)
    print(f"  principle ->  {wrapped[0]}")
    for extra in wrapped[1:]:
        print(f"                {extra}")
    mark = "correct" if _correct(reasoning, value, tol) else "wrong"
    print(f"  answer    ->  {_token(reasoning):<6}  {mark}")


# Measured on gemma4:latest (the chapter's standard model), 8 cold answers vs 8
# step-back answers per problem at temperature 0.7, via scripts/_stepback_probe.py.
# `direct` is the share of cold snap answers that land on the accepted value;
# `step_back` is the share of (name-the-principle then apply) answers that land.
# The honest spread: Half and Pendulum are wins (the model knows the law but the
# snap follows the wrong intuition), Pressure is a partial win (the principle
# points the right way but the arithmetic still trips it), and Inverse is an
# already-right ceiling where stepping back is redundant.
STEP_BACK_STUDY = {
    "samples": 8,
    "labels":    ["Half", "Pendulum", "Pressure", "Inverse"],
    "direct":    [0.000, 0.500, 0.000, 1.000],
    "step_back": [1.000, 1.000, 0.500, 1.000],
}


# The student-question task the Few-Shot section runs on, split so a sweep over
# the number of examples has somewhere honest to measure. POOL supplies the
# examples that go in the prompt, interleaved by class so every prefix stays
# roughly balanced; its first four are exactly the FEW_SHOT template the chapter
# shows, which puts that template on the curve at 4 shots. TEST is held out, five
# questions per class, and never appears in a prompt.
FEWSHOT_POOL = [
    ("When is the project due?", "LOGISTICS"),
    ("Why does gradient descent need a learning rate?", "CONCEPTUAL"),
    ("My loop prints nothing, what is wrong?", "DEBUGGING"),
    ("Can I use a late day on the lab?", "LOGISTICS"),
    ("What does big-O notation actually measure?", "CONCEPTUAL"),
    ("My program crashes only when the input file is empty.", "DEBUGGING"),
    ("Are the slides posted before or after class?", "LOGISTICS"),
    ("How is a stack different from a queue?", "CONCEPTUAL"),
]

FEWSHOT_TEST = [
    ("Where do I submit the final report?", "LOGISTICS"),
    ("How many late days do we get for the semester?", "LOGISTICS"),
    ("Is the final exam open book?", "LOGISTICS"),
    ("Do I need to come to lab if I finished early?", "LOGISTICS"),
    ("When do grades get posted?", "LOGISTICS"),
    ("Why does a neural network need a nonlinear activation?", "CONCEPTUAL"),
    ("What is the difference between a list and a tuple?", "CONCEPTUAL"),
    ("How does garbage collection decide what to free?", "CONCEPTUAL"),
    ("Why is quicksort usually faster than mergesort in practice?", "CONCEPTUAL"),
    ("What makes a hash function good?", "CONCEPTUAL"),
    ("My recursion hits maximum depth on a list of ten items.", "DEBUGGING"),
    ("The plot shows up empty even though the data loads fine.", "DEBUGGING"),
    ("I get a KeyError on line 40 but the key is definitely there.", "DEBUGGING"),
    ("My tests pass locally and fail on the server.", "DEBUGGING"),
    ("The loop skips every other element for some reason.", "DEBUGGING"),
]


# The sweep over how many examples a prompt carries, via
# scripts/_fewshot_sweep_probe.py. Same 15 held-out questions, same instruction,
# same model, run twice under two label vocabularies:
#
#   known    LOGISTICS / CONCEPTUAL / DEBUGGING, words that explain themselves
#   taught   Q1 / Q2 / Q3, the same three classes under a department's ticket
#            codes, whose meaning appears nowhere except in the examples
#
# `tokens` is what the prompt costs at each shot count. The pair is the whole
# lesson: examples are worth nothing when the model can already read the label
# names off the instruction, and they are worth everything when they carry the
# only definition of what the labels mean. Even then the curve tops out by the
# fourth example, and every one after that is tokens spent for no gain.
FEW_SHOT_STUDY = {
    "model": "gemma4:latest",
    "n_test": 15,
    "shots":  [0, 1, 2, 3, 4, 6, 8],
    "tokens": [31, 44, 61, 77, 94, 128, 161],
    "known":  [1.00, 1.0000, 1.0000, 1.0000, 1.00, 1.00, 1.00],
    "taught": [0.20, 0.4667, 0.8667, 0.9333, 1.00, 1.00, 1.00],
}


# The pink-elephant ban, measured token by token. Ollama reports top_logprobs for
# every generated token, so one call per model gives both the sentence it wrote
# and how much probability sat on an "elephant" token at each step. Captured on
# the same call path as the transcript above the figure (chat endpoint, BRIEF
# system prompt, think=False, temperature 0) via scripts/_pink_elephant_probe.py,
# trimmed to the first 12 tokens since every later step reads 0.
#
# `pressure` is P(elephant) at that step, `said_at` the step where the model
# actually emitted the word, `free_peak` the highest it reaches with no ban in
# the prompt at all. Read the three rows together and the ban turns out to fail
# in two different ways: it erases the word outright for gemma4 (1.00 -> 0.00),
# it cuts llama3.2:1b by two thirds and still loses because 0.33 stays the top
# pick, and it barely registers on gemma3:1b (0.82 -> 0.81).
PINK_ELEPHANT_STUDY = {
    "models": [
        {
            "model": "gemma4:latest",
            "free_peak": 1.0000,
            "said_at": None,
            "tokens": ["The", " largest", " terrestrial", " mammal",
                       " is", " the", " African", " bush",
                       " behem", "oth", "."],
            "pressure": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        },
        {
            "model": "gemma3:1b",
            "free_peak": 0.8212,
            "said_at": 10,
            "tokens": ["The", " largest", " land", " animal",
                       " on", " Earth", " is", " the",
                       " African", " bush", " elephant", "."],
            "pressure": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                         0.0000, 0.0000, 0.0000, 0.0035, 0.8067, 0.0000],
        },
        {
            "model": "llama3.2:1b",
            "free_peak": 0.9945,
            "said_at": 2,
            "tokens": ["The", " African", " elephant", " is",
                       " the", " largest", " land", " animal",
                       " on", " Earth", ",", " with"],
            "pressure": [0.0000, 0.0000, 0.3318, 0.0000, 0.0000, 0.0000,
                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        },
    ],
}


# ── Structured output: the "output format" lever from the prompt anatomy ────────

# A \small output box on the PDF page holds about 71 monospace characters before a
# line overflows; 69 leaves a cushion. Matches genai.agent._LINE_BUDGET.
_LINE_BUDGET = 69


def _say(label: str, body, gutter: int = 8, width: int = 63) -> None:
    """Print one turn of a transcript: a short label, then the wrapped body, so
    every turn lines up under the same gutter and the body fits the PDF box."""
    head, cont = f"{label:<{gutter}}", " " * gutter
    lines = []
    for para in str(body).split("\n"):
        lines.extend(textwrap.wrap(para, width=width, break_on_hyphens=False) or [""])
    print(head + lines[0])
    for line in lines[1:]:
        print(cont + line)


def show_structured(question: str, keys: list, model: str = "gemma4:latest") -> None:
    """The same question asked three ways, shown as a conversation: each USER turn is
    the real prompt, the model turn is the real reply, and PYTHON is the program
    trying to read one field straight out of it.

    Asked for a sentence, the model writes a correct line of English no parser can
    index. Asked for a loose list, it returns tidy lines but picks its own labels and
    leaves the units in, so the lookup misses. Only the strict ask -- exact key names,
    bare values, no units -- gives lines a one-line parser reads field by field.
    ("numbers only" would be the natural phrasing but it contradicts the `sport`
    key, whose value is a word.)
    Real model output, so the calling cell is frozen.
    """
    speaker, last = model.split(":")[0], keys[-1]
    hint = ", ".join(keys)
    prompts = [
        f"{question} Answer in a single sentence.",
        f"{question} List each fact on its own line.",
        f"{question} Reply as 'key: value' lines, using exactly the keys "
        f"{hint}, bare values with no units.",
    ]
    for i, prompt in enumerate(prompts):
        if i:
            print()
        reply = ask(prompt, model=model, max_tokens=120).strip()
        fields = {k.strip(): v.strip() for k, v in
                  (ln.split(":", 1) for ln in reply.splitlines() if ":" in ln)}
        got = repr(fields[last]) if last in fields else "KeyError"
        _say("USER", prompt)
        _say(speaker, reply)
        _say("PYTHON", f"fields['{last}'] -> {got}")


# --- comparison-demo display helpers ---------------------------------------
# Keep each demo cell to its variant data plus one call; the ask-loop, the
# first-sentence trimming, and the column formatting live here.

def _first_sentence(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def show_snake_walls(asks) -> None:
    """Generate Snake for each prompt and report its length and wall behaviour."""
    from genai.snake import generate_snake, walls_behavior
    for label, prompt in asks:
        src = generate_snake(prompt)
        print(f"{label:18s} {len(src.splitlines()):3d} lines   "
              f"walls: {walls_behavior(src)}")


def show_sampling_dials(prompt, dials, max_tokens: int = 40) -> None:
    """Run one prompt under each (label, sampling-options) pair, full output."""
    for label, opts in dials:
        print(f"{label:11s} {ask(prompt, options=opts, max_tokens=max_tokens)}")


def show_scenarios(prompt, scenarios, max_tokens: int = 45) -> None:
    """Run one prompt under each named sampling preset; trim to a sentence."""
    for name, params in scenarios.items():
        reply = _first_sentence(ask(prompt, options=params, max_tokens=max_tokens))
        print(f"── {name:24s} {reply}")


def show_classifications(template, questions, temperature: float = 0.1) -> None:
    """Classify each question with a few-shot template; print the one-word label."""
    for q in questions:
        raw = ask(template.format(body=q), options={"temperature": temperature})
        print(f"{raw.split()[0]:12s}  {q}")


def show_models(prompt, models, clip: bool = True, options=None,
                max_tokens: int = 40, system: str = None) -> None:
    """Run one prompt across several models; one row per model, model name left,
    its real reply beside it.

    The name column is sized to the longest model name, and a reply too long for
    one row wraps under a hanging indent instead of overflowing the PDF output
    box, where LaTeX would break it back to the far-left margin and lose the
    column the row was making.
    """
    gutter = max(len(m) for m in models) + 5   # "── " + name + 2 spaces
    for model in models:
        out = ask(prompt, model=model, system=system,
                  options=options or {}, max_tokens=max_tokens)
        if clip:
            out = _first_sentence(out)
        _say(f"── {model}", out, gutter=gutter, width=_LINE_BUDGET - gutter)


def show_bans(bans, model: str = DEFAULT_MODEL, max_tokens: int = 40) -> None:
    """Run one negative-instruction prompt per row: the ban on the left, the real
    reply beside it.

    The label column is sized to the longest label, so a short label leaves a
    single space of padding instead of the hole a fixed width would punch, and
    each reply wraps under a hanging indent that keeps every line inside the PDF
    output box. Real model output, so the calling cell is frozen.
    """
    gutter = max(len(label) for label, _ in bans) + 5  # "── " + label + 2 spaces
    for label, prompt in bans:
        reply = ask(prompt, model=model, options={"temperature": 0},
                    max_tokens=max_tokens)
        _say(f"── {label}", reply, gutter=gutter, width=_LINE_BUDGET - gutter)


def show_cot_comparison(problem, models, max_tokens: int = 80,
                        cot_max_tokens: int = 160) -> None:
    """Each model answers cold vs. with 'think step by step'; print both finals."""
    final = lambda t: (re.findall(r"-?\d+", t) or ["?"])[-1]
    for m in models:
        plain = final(ask(problem, model=m, max_tokens=max_tokens))
        cot = final(ask(problem + "\nLet's think step by step.",
                        model=m, max_tokens=cot_max_tokens))
        print(f"── {m:15s} no CoT: {plain:>4}   with CoT: {cot:>4}")


def show_reply(prompt, model: str = "gemma4:latest",
               num_predict: int = 200, limit: int = 400) -> None:
    """Print one model reply, capped to ``limit`` characters."""
    print(ask(prompt, model=model, options={"num_predict": num_predict})[:limit])

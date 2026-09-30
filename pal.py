"""Program-aided language model (PAL) helpers for the Prompting chapter.

PAL {cite}`gao2022pal` (and the closely related Program-of-Thoughts): instead of
reasoning toward a number in prose, the model writes a short Python program whose
execution yields the answer, with its reasoning expressed as code and comments. A
real interpreter runs the program, so the brittle step (multi-step arithmetic) is
handed off from the model to the interpreter. It helps when the problem is easy to
translate to code but awkward to grind out in the head; it cannot save a problem
the model mis-models (a wrong program runs to a confidently wrong number).

The calls here mirror scripts/_pal_probe.py exactly (no system prompt, think=False,
the same prompts and token budgets), so PAL_STUDY below is a faithful record of
what these functions actually do on qwen2.5-coder:latest. Because we now run code
the model wrote, run_program screens and executes it in a locked-down sandbox,
mirroring safe_eval in genai.agent, widened from one expression to a few lines.
"""
import re
import ast
import io
import builtins
import contextlib
import ollama
from genai.llm import SERVER, CODING_MODEL
from genai.agent import show_turn

_client = ollama.Client(host=SERVER)

# Prose chain of thought: reason in words, then commit to a parseable final line.
# This is the baseline PAL is measured against -- the same model, same problem,
# just doing the arithmetic itself instead of writing a program to do it.
_COT = (" Think step by step in words, then end with a line exactly like "
        "'Answer: <number>'.")
# PAL: don't do the arithmetic, write a program that does. Reasoning goes in the
# comments, the program stays import-free so it runs in the sandbox, and the final
# value lands in a known variable so we can read it back after running it.
_PAL = ("\n\nDo not do the arithmetic yourself. Write a short Python program that "
        "computes the answer. Reason in comments, use plain arithmetic and built-in "
        "functions only with no imports, and assign the final numeric answer to a "
        "variable named answer. Reply with only the program in one code block.")

# ── The sandbox ───────────────────────────────────────────────────────────────
# Built-ins a numeric program legitimately needs. Everything that could reach the
# file system, the network, or the interpreter internals (open, eval, __import__,
# ...) is deliberately left out, so the worst a buggy or hostile program can do is
# crash or compute a wrong number -- it can never escape the namespace it runs in.
_SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "min", "max", "round", "sum", "len", "int", "float", "range", "pow",
    "sorted", "enumerate", "zip", "map", "filter", "list", "dict", "set", "tuple",
    "bool", "str", "divmod", "print", "reversed", "all", "any") if hasattr(builtins, n)}
_BANNED_CALLS = {"eval", "exec", "open", "compile", "input", "__import__",
                 "globals", "locals", "vars", "getattr", "setattr", "delattr"}


def run_program(code: str, result: str = "answer"):
    """Run a short, model-written program in a throwaway, locked-down namespace and
    hand back the value it leaves in ``result``.

    The program is screened before it runs: its syntax tree is walked and any
    import, any call to a dangerous builtin, or any dunder attribute trick is
    refused. Then it executes with a restricted ``__builtins__`` and its stdout
    swallowed, so it cannot touch the world outside the sandbox. Raises ValueError
    if the program is rejected or never assigns ``result``.
    """
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in the sandbox")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in _BANNED_CALLS:
            raise ValueError(f"{node.func.id}() is not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed in the sandbox")
    ns = {"__builtins__": _SAFE_BUILTINS}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, "<pal>", "exec"), ns)
    if result not in ns:
        raise ValueError(f"the program never assigned `{result}`")
    return ns[result]


def _extract_code(reply: str) -> str:
    """Pull the program out of a chatty reply: the first ```fenced``` block if there
    is one, otherwise the whole reply."""
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


_NUM = re.compile(r"(-?\d+\s*/\s*\d+|-?\d+\.?\d*)")


def _value(text: str):
    """The committed final number (after 'Answer:' if present) as a float, with
    thousands commas and dollar signs stripped so '1,564' reads as one number."""
    tail = re.split(r"answer\s*[:=]", text, flags=re.I)[-1].replace(",", "").replace("$", "")
    found = _NUM.findall(tail)
    if not found:
        return None
    tok = found[-1].replace(" ", "")
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


def _close(value, target: float, tol: float) -> bool:
    return value is not None and abs(value - target) <= tol


def _ask(prompt: str, model: str, temp: float, max_tokens: int) -> str:
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return (resp["message"]["content"] or "").strip()


def cot_answer(question: str, model: str = CODING_MODEL, temp: float = 0.7):
    """Solve the problem in prose. Returns ``(text, value)``: the model's full
    step-by-step reply and the number parsed from its final 'Answer:' line."""
    text = _ask(question + _COT, model, temp, 320)
    return text, _value(text)


def pal_solve(question: str, model: str = CODING_MODEL, temp: float = 0.7):
    """Solve the problem by program. Ask the model for a short Python program, pull
    the code out, run it in the sandbox, and read back ``answer``. Returns
    ``(code, value)``; value is None if the program is rejected, crashes, or never
    assigns ``answer``."""
    code = _extract_code(_ask(question + _PAL, model, temp, 320))
    try:
        return code, float(run_program(code))
    except Exception:
        return code, None


# Curated multi-step word problems with UNCOMMON numbers. The first three are
# prose-hostile: summing a series or compounding over many steps is exactly what a
# model is bad at in its head but a program is perfect at. The last two have a
# single tricky step (an off-by-one) but light arithmetic, so the model already
# nails them either way and PAL is redundant. "Snail" is also the cautionary tale:
# the naive program (7 // (4 - 3)) would run to a confident 7, but the snail climbs
# out on day 4 before it can slide back, a reminder that PAL is only ever as right
# as the model's model of the problem. Each: (label, question, value, tol, pretty).
PAL_PROBLEMS = [
    ("Pages", "On the first day a student reads 3 pages. Each day after, she reads "
              "4 more pages than the day before. How many pages has she read in "
              "total after 30 days?", 1830, 0.5, "1830"),
    ("Seats", "A theatre has 24 rows. The first row has 18 seats and each row after "
              "has 3 more seats than the one before. How many seats are there in "
              "total?", 1260, 0.5, "1260"),
    ("Compound", "A town of 1200 people grows by 8 percent every year. How many "
                 "people are there after 10 years, rounded to the nearest whole "
                 "person?", 2591, 0.5, "2591"),
    ("Snail", "A snail is at the bottom of a 7 metre well. It climbs 4 metres each "
              "day and slides back 3 metres each night. How many days does it take "
              "to reach the top?", 4, 0.5, "4"),
    ("Cuts", "It takes 4 minutes to saw through a log. A 30 metre log is cut into 6 "
             "equal pieces. How many minutes does the sawing take in total?", 20, 0.5,
     "20"),
]


def _problem(label: str):
    """Look up a PAL_PROBLEMS entry by its label."""
    return next(p for p in PAL_PROBLEMS if p[0] == label)


def _fmt(value) -> str:
    """Format a computed number for the transcript: drop a trailing .0, keep the
    rest, and show a failed run as '(no answer)'."""
    if value is None:
        return "(no answer)"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _compact(code: str) -> list:
    """Drop blank lines so the program prints as a tight block in the transcript."""
    return [ln for ln in code.splitlines() if ln.strip()]


def _elide_reply(text: str, head_chars: int = 120) -> str:
    """Render a prose chain of thought as one transcript turn: its opening words
    and its committed final line, with the long arithmetic middle elided so the
    answer stays visible at the end. Both ends are the model's real words."""
    flat = " ".join(text.split())
    final = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), flat)
    if len(flat) <= head_chars + len(final) + 20:
        return flat
    head = flat[:head_chars].rstrip()
    head = head[:head.rfind(" ")] if " " in head else head  # back off to a whole word
    return f"{head} …\n… [reasoning elided] …\n{final}"


def _show_prose_turn(label: str, speaker: str, text: str, val) -> None:
    """Render one prose attempt as a USER / model / GRADE conversation. Split out
    from the live draw so a cached reply (the clean wrong commit is rare enough to
    be worth freezing) can be replayed into the frozen cell without re-rolling."""
    _, question, value, tol, pretty = _problem(label)
    show_turn("USER", question)
    show_turn(speaker, _elide_reply(text))
    mark = "correct" if _close(val, value, tol) else "wrong"
    show_turn("GRADE", f"lands on {_fmt(val)}, should be {pretty}  ({mark})")


def show_pal_prose(label: str = "Pages", model: str = CODING_MODEL) -> None:
    """Panel one, as a conversation: ask the problem in prose and watch the model
    set the calculation up correctly and then commit a confidently wrong number.
    The reply's middle is elided so its committed answer lands at the end, and a
    GRADE line scores it against the target instead of dressing the target up as
    part of the prompt. Real model output, so the calling cell is frozen."""
    text, val = cot_answer(_problem(label)[1], model)
    _show_prose_turn(label, model.split(":")[0], text, val)


def _show_program_turn(label: str, code: str, val) -> None:
    """Render one program attempt as a USER / code / PYTHON conversation that
    mirrors the prose panel. Split from the live solve so a cached program can be
    replayed into the frozen cell. The program prints flush rather than under a
    speaker gutter: its commented lines are too wide to indent and still fit the
    output box, and narrating it under the model's name would not be its real
    output anyway. The real output is the code itself."""
    _, question, value, tol, pretty = _problem(label)
    show_turn("USER", question)
    print()
    for ln in _compact(code):
        print(ln)
    print()
    mark = "correct" if _close(val, value, tol) else "wrong"
    show_turn("PYTHON", f"the program computes {_fmt(val)}  ({mark})")


def show_pal(label: str = "Pages", model: str = CODING_MODEL) -> None:
    """Panel two: hand the SAME question to the SAME model as a program. The model's
    real program is printed verbatim between a USER turn and the PYTHON turn that
    reports what the sandboxed interpreter computed from it. Real model output and a
    real execution, so the calling cell is frozen."""
    code, val = pal_solve(_problem(label)[1], model)
    _show_program_turn(label, code, val)


# Measured on qwen2.5-coder:latest (one coding model, both modes, so the contrast
# is apples-to-apples), 8 tries each at temperature 0.7, via scripts/_pal_probe.py.
# `cot` is the share of prose chain-of-thought answers that land on the value;
# `pal` is the share of (write-a-program then run it) answers that land. The honest
# spread: Pages/Seats/Compound are prose-hostile wins (summing a series or
# compounding is bookkeeping the model botches in its head but a loop nails every
# time), Cuts is a ceiling where the light arithmetic is easy enough that PAL is
# redundant, and Snail is the limit -- the model keeps mis-modeling it (a naive
# program forgets the snail climbs out before it can slide), so even PAL is wrong
# most of the time: a wrong program runs to a confidently wrong number.
PAL_STUDY = {
    "samples": 8,
    "labels":    ["Pages", "Seats", "Compound", "Cuts", "Snail"],
    "cot":       [0.500, 0.250, 0.250, 1.000, 0.125],
    "pal":       [1.000, 1.000, 1.000, 1.000, 0.375],
}

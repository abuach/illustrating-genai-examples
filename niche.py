"""Engineering around a language the model has never seen.

A generative model is fluent in Python because the web is full of Python. Drop
it into a low-resource language (Verilog, Rust's macro system, R, an in-house
configuration DSL) and the fluency evaporates: it invents keywords, mangles the
syntax, and states a wrong program as confidently as a right one. The cause is
data scarcity, not a flaw in the model, so the fix is engineering around the shortfall
rather than waiting for a bigger model.

To make the shortfall airtight we invent the most niche language possible: CONVEY, a
tiny conveyor-belt language for a list of numbers, with zero examples anywhere in
any training set. A model that can write CONVEY at all is writing it from the
docs we hand it, not from memory. This module is the CONVEY interpreter, its
spec (the docs we retrieve over), and three scarcity workarounds measured against
a no-help baseline: retrieval over the spec, grammar-constrained decoding, and
transfer from a high-resource sibling.

The model is a code model (qwen2.5-coder), not gpt-oss: writing code in an unseen
syntax is exactly the job a code-tuned model is built for, and gpt-oss empties its
content channel to reason. Runs are nondeterministic, so capture NICHE_STUDY once
and freeze the cells. Captured by scripts/_niche_workarounds_probe.py.
"""
import json
import re

import ollama

# A code model fits a code task; see the module docstring. Hardcoded here the way
# reason.py pins EIS_MODEL, so the chapter cell names no model.
NICHE_MODEL = "qwen2.5-coder:latest"


# ── The CONVEY language ───────────────────────────────────────────────────────
# A program is a list of stations, one per line, run top to bottom. The belt
# carries a list of integers. feed loads it, only filters it, each maps it, pack
# collapses it to a single number. The keywords are deliberately not the obvious
# ones (only, each, pack, not filter, map, reduce) so a model can't guess the
# syntax from a sibling language; it has to read the spec.

_PREDS = {"even": lambda n: n % 2 == 0,
          "odd": lambda n: n % 2 != 0,
          "positive": lambda n: n > 0}
_FNS = {"double": lambda n: n * 2,
        "negate": lambda n: -n,
        "plus1": lambda n: n + 1}
_AGGS = {"sum": sum, "max": max, "min": min, "count": len}
KEYWORDS = list(_PREDS) + list(_FNS) + list(_AGGS)


class ConveyError(Exception):
    """A CONVEY program that doesn't parse or doesn't run."""


def run_convey(program: str):
    """Execute a CONVEY program and return the number its ``pack`` station emits.

    Raises ConveyError on any unknown station, malformed line, or a program that
    never packs the belt down to a result. That strictness is the point: it's the
    deterministic oracle that tells a generated program apart from a working one.
    """
    belt, result = [], None
    for raw in program.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split()
        if head == "feed":
            try:
                belt = [int(tok) for tok in rest]
            except ValueError:
                raise ConveyError(f"feed wants integers, got {rest}")
        elif head == "only":
            if not rest or rest[0] not in _PREDS:
                raise ConveyError(f"unknown filter: {' '.join(rest) or '(none)'}")
            belt = [n for n in belt if _PREDS[rest[0]](n)]
        elif head == "each":
            if not rest or rest[0] not in _FNS:
                raise ConveyError(f"unknown map: {' '.join(rest) or '(none)'}")
            belt = [_FNS[rest[0]](n) for n in belt]
        elif head == "pack":
            if not rest or rest[0] not in _AGGS:
                raise ConveyError(f"unknown pack: {' '.join(rest) or '(none)'}")
            result = _AGGS[rest[0]](belt) if belt else 0
        else:
            raise ConveyError(f"unknown station: {head!r}")
    if result is None:
        raise ConveyError("program never packed the belt into a result")
    return result


# ── The spec we retrieve over (the language's own docs) ───────────────────────
# Eight short passages, the kind of reference a real DSL ships. RAG retrieves the
# slice relevant to a task; the overview passage is always pinned so the four
# station names are never missing, the way a real RAG prompt pins a system doc.

SPEC = [
    ("overview",
     "A CONVEY program is a conveyor belt for a list of numbers, one station per "
     "line, run top to bottom. The four stations are feed (load numbers), only "
     "(keep/filter), each (transform/map), and pack (reduce to one number)."),
    ("feed",
     "feed starts the belt. Write 'feed' then the numbers separated by spaces, "
     "for example 'feed 4 7 2 9'. It must be the first station. Negative numbers "
     "are allowed, e.g. 'feed 3 -2 8'."),
    ("only",
     "only keeps the items matching a test and drops the rest (a filter). The "
     "tests are 'only even', 'only odd', and 'only positive'. Example: 'only "
     "even' keeps just the even numbers on the belt."),
    ("each",
     "each transforms every item on the belt (a map). The transforms are 'each "
     "double' (multiply by two), 'each negate' (flip the sign), and 'each plus1' "
     "(add one). Example: 'each double' turns 3 into 6."),
    ("pack",
     "pack collapses the whole belt into a single number and ends the program (a "
     "reduce). The reducers are 'pack sum' (add them up/total), 'pack max' (the "
     "largest), 'pack min' (the smallest), and 'pack count' (how many). A program "
     "must end in pack."),
    ("order",
     "Stations run in the order you write them, so order changes the answer. "
     "'each double' then 'only even' is not the same as 'only even' then 'each "
     "double': doubling first makes every number even."),
    ("example1",
     "Example: load 1 2 3 4, keep the even ones, and total them. The program is "
     "'feed 1 2 3 4', then 'only even', then 'pack sum', which yields 6."),
    ("example2",
     "Example: load 5 9 2, double each number, and report the largest. The "
     "program is 'feed 5 9 2', then 'each double', then 'pack max', which yields 18."),
]


def retrieve_spec(task: str, k: int = 3) -> list:
    """The overview passage plus the ``k`` spec passages most relevant to ``task``.

    Relevance is plain word overlap, the same cheap retriever the MCP chapter's
    search tool used. The overview is always pinned, so the model always sees the
    four station names even when a task's wording matches only a station or two.
    """
    words = set(re.findall(r"[a-z]{3,}", task.lower()))

    def overlap(passage):
        return len(words & set(re.findall(r"[a-z]{3,}", passage[1].lower())))

    detail = [p for p in SPEC if p[0] != "overview"]
    ranked = sorted(detail, key=overlap, reverse=True)[:k]
    pinned = next(p for p in SPEC if p[0] == "overview")
    return [pinned] + ranked


# ── The model writing CONVEY four ways ────────────────────────────────────────

def _extract(text: str) -> str:
    """Pull the CONVEY program out of a free-form reply: a fenced block if there
    is one, else the lines that start with a station keyword."""
    fenced = re.search(r"```[a-z]*\n(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    starts = ("feed", "only", "each", "pack")
    lines = [ln for ln in text.splitlines() if ln.strip().split(" ")[0] in starts]
    return "\n".join(lines).strip() or text.strip()


def _chat(prompt: str, model: str, schema: dict = None) -> str:
    """One chat turn. With a schema, decoding is grammar-constrained to it."""
    kw = {"format": schema} if schema else {}
    return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"num_predict": 400, "temperature": 0},
                       **kw)["message"]["content"]


_ASK = ("Write a CONVEY program for this task. Output only the program.\n"
        "Task: {task}")

_TRANSFER = (
    "CONVEY is a small language like a Unix shell pipeline: a source stage, then "
    "filter stages, then transform stages, then one final reducer, written one "
    "stage per line instead of joined by pipes. Using that analogy, write a "
    "CONVEY program for this task. Output only the program.\nTask: {task}")

# The JSON schema is the grammar: decoding can only emit one of the four station
# names and one of the real keywords, so the syntax is valid by construction.
SCHEMA = {
    "type": "object",
    "properties": {
        "stations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["feed", "only", "each", "pack"]},
                    "nums": {"type": "array", "items": {"type": "integer"}},
                    "arg": {"type": "string", "enum": KEYWORDS},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["stations"],
}

_SCHEMA_ASK = (
    "Express this task as CONVEY stations. feed loads numbers (nums); only filters "
    "with arg even/odd/positive; each maps with arg double/negate/plus1; pack "
    "reduces with arg sum/max/min/count.\nTask: {task}")


def _serialize(obj: dict) -> str:
    """Render a structured program object into CONVEY text. Because every field
    came from the schema's enums, the text is syntactically valid by construction."""
    lines = []
    for st in obj.get("stations", []):
        op = st.get("op", "")
        if op == "feed":
            lines.append("feed " + " ".join(str(n) for n in st.get("nums", [])))
        else:
            lines.append(f"{op} {st.get('arg', '')}".strip())
    return "\n".join(lines)


def write_convey(task: str, mode: str = "baseline", model: str = NICHE_MODEL) -> str:
    """Ask the model for a CONVEY program under one scarcity workaround.

    ``baseline`` hands it nothing but the task; ``transfer`` leans on the
    shell-pipeline sibling; ``rag`` prepends the retrieved spec passages;
    ``grammar`` constrains decoding to the schema; ``assist`` combines rag and
    grammar. Returns the program text (already serialized for the grammar modes).
    """
    if mode in ("grammar", "assist"):
        prompt = _SCHEMA_ASK.format(task=task)
        if mode == "assist":
            docs = "\n".join(f"- {name}: {body}" for name, body in retrieve_spec(task))
            prompt = f"CONVEY reference:\n{docs}\n\n{prompt}"
        raw = _chat(prompt, model, schema=SCHEMA)
        try:
            return _serialize(json.loads(raw))
        except (json.JSONDecodeError, AttributeError):
            return raw
    if mode == "transfer":
        return _extract(_chat(_TRANSFER.format(task=task), model))
    if mode == "rag":
        docs = "\n".join(f"- {name}: {body}" for name, body in retrieve_spec(task))
        prompt = f"CONVEY reference:\n{docs}\n\n{_ASK.format(task=task)}"
        return _extract(_chat(prompt, model))
    return _extract(_chat(_ASK.format(task=task), model))


def assist(task: str, model: str = NICHE_MODEL) -> tuple:
    """The recommended workaround (retrieval + grammar) for one task.

    Returns ``(program, result, ok)``: the CONVEY text, the number it runs to (or
    None if it doesn't run), and whether that matches the task's gold answer.
    """
    program = write_convey(task, mode="assist", model=model)
    return (program, *_check(program, task))


# ── The measured study ────────────────────────────────────────────────────────
# Each task is a sentence plus the reference CONVEY program that solves it. Gold
# answers are computed by running the reference through the interpreter, never
# hand-typed, so a generated program passes only when it lands on the same number.

TASKS = [
    ("Load 4, 7, 2, 9 onto the belt, keep only the even numbers, then add them up.",
     "feed 4 7 2 9\nonly even\npack sum"),
    ("Start with 1, 2, 3, 4, 5; double each number; and report the largest.",
     "feed 1 2 3 4 5\neach double\npack max"),
    ("Take 3, -2, 8, -5; keep the positive ones; and count how many there are.",
     "feed 3 -2 8 -5\nonly positive\npack count"),
    ("Load 10, 15, 20; add one to each; then give the smallest.",
     "feed 10 15 20\neach plus1\npack min"),
    # the tricky tail: order matters, and the natural wording invites the wrong order
    ("Feed 5, 8, 13, 2; double each number; then keep only the even ones; and "
     "add them up.",
     "feed 5 8 13 2\neach double\nonly even\npack sum"),
    ("Take 7, 4, 9, 6; keep the odd numbers; negate each; and report the smallest.",
     "feed 7 4 9 6\nonly odd\neach negate\npack min"),
]

MODES = ["baseline", "transfer", "rag", "grammar", "assist"]


def gold(reference: str) -> int:
    """The gold answer for a task: its reference program's real output."""
    return run_convey(reference)


def _check(program: str, task: str) -> tuple:
    """Run a generated program and compare it to the task's gold answer.

    Returns ``(result, ok)`` where result is the number the program produced (or
    None if it failed to run) and ok is whether it equals gold.
    """
    reference = dict((t, r) for t, r in TASKS)[task]
    want = gold(reference)
    try:
        got = run_convey(program)
    except (ConveyError, Exception):
        return None, False
    return got, got == want


def niche_study(model: str = NICHE_MODEL, trials: int = 3) -> dict:
    """Pass rate for every workaround across the task suite, ``trials`` times each.

    Returns ``{mode: pass_rate}`` in [0, 1]. This is the honest, re-runnable
    harness behind NICHE_STUDY; the probe just prints what it returns.
    """
    out = {mode: 0 for mode in MODES}
    n = len(TASKS) * trials
    for task, _ in TASKS:
        for _ in range(trials):
            for mode in MODES:
                program = write_convey(task, mode=mode, model=model)
                _, ok = _check(program, task)
                out[mode] += int(ok)
    return {mode: round(hits / n, 3) for mode, hits in out.items()}


# ── Baked captures (scripts/_niche_workarounds_probe.py, qwen2.5-coder) ────────
# Real model output, kept verbatim so the cells reproduce without a rerun.

# Pass rate per workaround over six tasks x three trials (temperature 0, so the
# three trials of each task are identical and rates land on sixths). The finding:
# with no help, and even leaning on the pipeline analogy, the model scores zero;
# it can't guess CONVEY's keywords. Retrieval and grammar each take it to a clean
# sweep, because the scarce part was the syntax, not the logic. Stacking both
# (assist) doesn't help and costs one task: on task 0 the constrained decoder
# emitted `feed` with an empty number list, a valid program that loads nothing and
# sums to 0. Grammar guarantees syntax, not sense.
NICHE_STUDY = {
    "baseline": 0.0, "transfer": 0.0, "rag": 1.0, "grammar": 1.0, "assist": 0.833,
}

# The §1 failure: the model handed task 5 with no help. Its real program has the
# right shape (a pipeline) and the wrong words for every station, so the
# interpreter rejects it on the first one. (On the longer task 0 it lurched
# further off and wrote fifty lines of an invented assembly language.)
NICHE_FAIL = {
    "task": TASKS[5][0],
    "program": "input 7, 4, 9, 6;\nfilter odd;\nnegate all;\nreport min;",
    "verdict": "rejected: unknown station: 'input'",
}

# The §4 success: the same task, now with retrieval over the spec. The passages
# RAG surfaced, the valid program the model wrote with them in front of it, and
# the checked verdict (trailing markdown whitespace trimmed from each line).
NICHE_FIX = {
    "task": TASKS[5][0],
    "retrieved": ["overview", "only", "each", "example2"],
    "program": "feed 7 4 9 6\nonly odd\neach negate\npack min",
    "verdict": "ran to -9 -> PASS",
}


def verdict(program: str, task: str) -> str:
    """The interpreter's one-line ruling on a generated program, for display."""
    reference = dict(TASKS)[task]
    want = run_convey(reference)
    try:
        got = run_convey(program)
    except Exception as exc:
        return f"rejected: {exc}"
    if got == want:
        return f"ran to {got} -> PASS"
    return f"ran to {got}, but the task wanted {want} -> wrong"


def _listing(program: str) -> str:
    """Indent each program line for display. The indent keeps the transcript
    reflow pass from mistaking a short, regular program for word-wrapped prose
    and merging its lines; the stored ``program`` stays the model's clean output."""
    return "\n".join("  " + line for line in program.splitlines())


def show_niche_fail(demo=NICHE_FAIL):
    """The cold-start failure: a strong code model, a tiny unseen language, no help."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["task"])
    show_code("qwen-coder", _listing(demo["program"]))
    show_turn("CONVEY", demo["verdict"])


def show_niche_fix(demo=NICHE_FIX):
    """The same task with retrieval over the spec: the model writes from the docs."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["task"])
    show_turn("RAG", "retrieved spec passages: " + ", ".join(demo["retrieved"]))
    show_code("qwen-coder", _listing(demo["program"]))
    show_turn("CONVEY", demo["verdict"])


# ── The version it learned was last year's ────────────────────────────────────
# Retrieval fixed the *volume* axis of scarcity: hand the model the spec and it
# writes CONVEY that runs. But a language has a version, and the spec you retrieve
# over (like the corpus a model trained on) can be a release behind. CONVEY v2
# renamed the final station 'pack' to 'collapse'. A model working from the v1 spec
# faithfully writes 'pack', valid last year and rejected by today's interpreter, and
# a changelog line buried in the old docs isn't enough to move it; the fix is to
# retrieve over the *current* spec. Same RAG workaround, aimed at time instead of
# volume. Model output is baked (qwen2.5-coder, temperature 0); the v2 verdicts
# recompute live from it. Captured by scripts/_niche_version_probe.py.

def run_convey_v2(program: str):
    """CONVEY v2, where the reducer station 'pack' was renamed 'collapse'. Behaviour
    is unchanged, so 'collapse' does exactly what 'pack' did; the old keyword is now
    an unknown station, the way a real rename retires a name."""
    lines = []
    for raw in program.strip().splitlines():
        line = raw.strip()
        head = line.split()[0] if line.split() else ""
        if head == "pack":
            raise ConveyError("unknown station: 'pack' (renamed 'collapse' in v2)")
        if head == "collapse":
            line = line.replace("collapse", "pack", 1)
        lines.append(line)
    return run_convey("\n".join(lines))


# The updated reference: the v1 spec with the reducer renamed 'pack' -> 'collapse'
# everywhere, and nothing else changed, so retrieval behaves exactly as it did on the
# v1 spec. Retrieving over THIS (not the stale spec) is the fix.
SPEC_V2 = [("collapse" if name == "pack" else name,
            re.sub(r"\bpack\b", "collapse", body)) for name, body in SPEC]


def retrieve_spec_v2(task: str, k: int = 3) -> list:
    """``retrieve_spec`` over the updated (v2) spec, so the model reads the current
    keyword names instead of the retired ones."""
    words = set(re.findall(r"[a-z]{3,}", task.lower()))
    detail = [p for p in SPEC_V2 if p[0] != "overview"]
    ranked = sorted(detail, key=lambda p: len(words & set(
        re.findall(r"[a-z]{3,}", p[1].lower()))), reverse=True)[:k]
    return [next(p for p in SPEC_V2 if p[0] == "overview")] + ranked


def write_versioned(task: str, current: bool = False, model: str = NICHE_MODEL) -> str:
    """Write CONVEY with RAG over either the stale v1 spec (``current=False``) or the
    updated v2 spec (``current=True``). The reproducer behind the baked demo. The
    v2 station set replaces 'pack' with 'collapse', so the reducer line survives
    extraction."""
    spec = retrieve_spec_v2(task) if current else retrieve_spec(task)
    docs = "\n".join(f"- {name}: {body}" for name, body in spec)
    text = _chat(f"CONVEY reference:\n{docs}\n\n{_ASK.format(task=task)}", model)
    fenced = re.search(r"```[a-z]*\n(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    stations = ("feed", "only", "each", "collapse" if current else "pack")
    lines = [ln for ln in text.splitlines() if ln.strip().split(" ")[0] in stations]
    return "\n".join(lines).strip() or text.strip()


def verdict_v2(program: str, task: str) -> str:
    """The v2 interpreter's one-line ruling on a program, checked against the task's
    gold answer (computed from its reference), for display."""
    want = run_convey(dict(TASKS)[task])
    try:
        got = run_convey_v2(program)
    except Exception as exc:
        return f"rejected: {exc}"
    return f"ran to {got} -> PASS" if got == want else f"ran to {got}, wanted {want} -> wrong"


# Captured by scripts/_niche_version_probe.py (qwen2.5-coder, temperature 0). The
# stale program is the model's real output from the v1 spec; the current one from
# the v2 spec. Trailing display whitespace trimmed per line; verdicts recompute live.
VERSION_STALE = {"task": TASKS[2][0],
                 "program": "feed 3 -2 8 -5\nonly positive\npack count"}
VERSION_CURRENT = {"task": TASKS[2][0],
                   "program": "feed 3 -2 8 -5\nonly positive\ncollapse count"}


def show_stale_dialect(stale: dict = None, current: dict = None) -> None:
    """RAG over a version-stale spec makes the model write the retired 'pack'
    station, which the v2 interpreter rejects; retrieving the updated spec lands the
    current 'collapse'. The programs are the model's real output; the v2 verdicts
    recompute live from the interpreter."""
    from genai.agent import show_code, show_turn
    stale = stale or VERSION_STALE
    current = current or VERSION_CURRENT
    show_turn("you", stale["task"])
    show_turn("RAG", "retrieved the CONVEY spec it has (a version behind)")
    show_code("qwen-coder", _listing(stale["program"]))
    show_turn("CONVEY v2", verdict_v2(stale["program"], stale["task"]))
    show_turn("FIX", "retrieve the current spec instead (v2 renamed pack -> collapse)")
    show_code("qwen-coder", _listing(current["program"]))
    show_turn("CONVEY v2", verdict_v2(current["program"], current["task"]))


# ── The other end of the spectrum: a real low-resource language (Verilog) ─────
# CONVEY is the floor (zero training data). Verilog is the realistic case: a
# modern code model has seen enough to write it fluently, so scarcity surfaces
# not as broken syntax but as a subtle wrong FACT that only a verifier catches.
# We compile the model's module with Icarus Verilog and run it against a fixed
# testbench (the oracle), the same role CONVEY's interpreter plays. The task is a
# 4-bit binary-to-Gray converter; gold is checked by the testbench, not by hand.

GRAY_DECL = "module gray4(input [3:0] bin, output [3:0] gray);"
GRAY_SPEC = "Convert the 4-bit binary input bin to its 4-bit Gray code on gray."
GRAY_DOC = ("Binary-to-Gray conversion: gray = bin ^ (bin >> 1). Each Gray-code "
            "bit is a binary bit XORed with the next-higher binary bit.")
GRAY_TB = (
    "module tb; reg [3:0] bin; wire [3:0] gray; integer f=0;\n"
    " gray4 d(.bin(bin),.gray(gray));\n"
    " initial begin bin=0;#1; if(gray!==4'b0000)f=f+1; bin=1;#1; if(gray!==4'b0001)f=f+1;\n"
    "  bin=2;#1; if(gray!==4'b0011)f=f+1; bin=7;#1; if(gray!==4'b0100)f=f+1;\n"
    "  if(f==0) $display(\"RESULT:PASS\"); else $display(\"RESULT:FAIL\"); $finish; end endmodule\n")


def simulate(module: str, testbench: str = GRAY_TB) -> str:
    """Compile a Verilog module with its testbench (iverilog) and run it (vvp).

    Returns one of 'pass', 'fail' (compiled and ran but the testbench's checks
    failed), or 'compile error'. This is the Verilog oracle, the analogue of
    run_convey: a real tool, not the model's say-so.
    """
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mp, tp, sp = f"{d}/m.v", f"{d}/tb.v", f"{d}/sim"
        open(mp, "w").write(module)
        open(tp, "w").write(testbench)
        c = subprocess.run(["iverilog", "-g2012", "-o", sp, mp, tp],
                           capture_output=True, text=True)
        if c.returncode != 0:
            return "compile error"
        r = subprocess.run(["vvp", sp], capture_output=True, text=True, timeout=10)
        return "pass" if "RESULT:PASS" in r.stdout else "fail"


def write_verilog(decl: str, spec: str, doc: str = None,
                  model: str = NICHE_MODEL) -> str:
    """Ask the model for a Verilog module for one interface; optionally hand it a
    reference passage first (the retrieval workaround)."""
    base = (f"Write a complete Verilog module with this exact interface:\n{decl}\n"
            f"Behavior: {spec}\nOutput only the Verilog code.")
    prompt = f"Verilog reference:\n- {doc}\n\n{base}" if doc else base
    text = _chat(prompt, model)
    fenced = re.search(r"```(?:verilog|systemverilog|v)?\n(.*?)```", text, re.S)
    return (fenced.group(1) if fenced else text).strip()


# Baked from scripts/_niche_workarounds_probe.py (qwen2.5-coder + Icarus Verilog,
# temperature 0). Blank lines trimmed from the model's reply for display; the code
# is otherwise verbatim. The baseline module COMPILES and RUNS but its Gray
# formula is subtly wrong (gray[i] = bin[i-1]^bin[i] instead of bin[i]^bin[i+1]),
# so the simulator's testbench fails. Hand the model the one missing fact and it
# writes the correct one-liner.
VERILOG_FAIL = {
    "task": "Write a Verilog gray-code converter (4-bit binary to Gray).",
    "program": ("module gray4(\n    input [3:0] bin,\n    output reg [3:0] gray\n);\n"
                "always @(*) begin\n    // Convert binary to Gray code\n"
                "    gray[0] = bin[0];\n    gray[1] = bin[0] ^ bin[1];\n"
                "    gray[2] = bin[1] ^ bin[2];\n    gray[3] = bin[2] ^ bin[3];\n"
                "end\nendmodule"),
    "verdict": "compiles and runs, but the simulator's testbench fails: wrong Gray output",
}

VERILOG_FIX = {
    "task": "Write a Verilog gray-code converter (4-bit binary to Gray).",
    "doc": GRAY_DOC,
    "program": ("module gray4(\n    input [3:0] bin,\n    output reg [3:0] gray\n);\n"
                "always @(*) begin\n    gray = bin ^ (bin >> 1);\nend\nendmodule"),
    "verdict": "simulator's testbench passes on every input",
}


def show_verilog_fail(demo=VERILOG_FAIL):
    """A real low-resource language: fluent Verilog that the simulator says is wrong."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["task"])
    show_code("qwen-coder", demo["program"])
    show_turn("SIM", demo["verdict"])


def show_verilog_fix(demo=VERILOG_FIX):
    """The same task after retrieving the one fact the model was missing."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["task"])
    show_turn("RAG", "retrieved: " + demo["doc"])
    show_code("qwen-coder", demo["program"])
    show_turn("SIM", demo["verdict"])


# ── The model that hallucinates a library ─────────────────────────────────────
# Scarcity shows first as *confabulated symbols*: handed a CONVEY task with no
# docs, the model doesn't just fumble the syntax, it invents station and keyword
# names that sound right and don't exist (load, filter, map, reduce, sum_up). A
# fake name is worse than a syntax slip, because it reads as correct until it hits
# the interpreter, and in a real ecosystem an attacker can register the package a
# model keeps hallucinating. The cure is to ground generation in the real symbol
# table, the way `dir(package)` or a language server lists what actually exists.
# Symbols resolved against the vocabulary is the measure. Captured by
# scripts/_niche_hallucination_probe.py (qwen2.5-coder).

STATIONS = ["feed", "only", "each", "pack"]
ARG_SETS = {"only": set(_PREDS), "each": set(_FNS), "pack": set(_AGGS)}

# The symbol table a language server would expose: every name that exists, and
# nothing that doesn't. This is what grounds the model, not prose docs.
LIBRARY = ("CONVEY symbol table (the only names that exist):\n"
           "  stations: feed, only, each, pack\n"
           f"  only <test>: {', '.join(_PREDS)}\n"
           f"  each <fn>: {', '.join(_FNS)}\n"
           f"  pack <reducer>: {', '.join(_AGGS)}")


def program_symbols(program: str) -> list:
    """Every station name and keyword a program uses, as ``(symbol, exists)``
    pairs. feed's numeric arguments aren't symbols, so they're skipped."""
    out = []
    for raw in program.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split()
        out.append((head, head in STATIONS))
        if head in ARG_SETS and rest:
            out.append((rest[0], rest[0] in ARG_SETS[head]))
    return out


def resolve_rate(program: str) -> float:
    """Fraction of the symbols a program uses that exist in the CONVEY vocabulary.
    A confabulated name drags it below 1.0."""
    syms = program_symbols(program)
    return round(sum(ok for _, ok in syms) / len(syms), 3) if syms else 0.0


def unresolved(program: str) -> list:
    """The invented symbols in a program: names it used that don't exist."""
    return [s for s, ok in program_symbols(program) if not ok]


def write_grounded(task: str, model: str = NICHE_MODEL) -> str:
    """Write a CONVEY program with the real symbol table in the prompt (the
    LIBRARY event), so the model can only reach for names that exist."""
    prompt = f"{LIBRARY}\n\n{_ASK.format(task=task)}"
    return _extract(_chat(prompt, model))


def hallucination_study(tasks: list = None, model: str = NICHE_MODEL) -> dict:
    """Mean symbol-resolution rate over the tasks with no help versus with the
    symbol table, plus every invented symbol the naive arm reached for."""
    tasks = tasks or [t for t, _ in TASKS]
    naive_rates, grounded_rates, invented = [], [], []
    for task in tasks:
        naive = write_convey(task, "baseline", model)
        grounded = write_grounded(task, model)
        naive_rates.append(resolve_rate(naive))
        grounded_rates.append(resolve_rate(grounded))
        invented += unresolved(naive)
    from statistics import mean
    return {"n": len(tasks),
            "naive_resolve": round(mean(naive_rates), 3),
            "grounded_resolve": round(mean(grounded_rates), 3),
            "invented": sorted(set(invented))}


# Captured by scripts/_niche_hallucination_probe.py (qwen2.5-coder, n=6). With no
# docs the model confabulates every symbol (0.0 resolve); with the symbol table it
# draws almost entirely from names that exist (0.873). It doesn't make the model
# fluent, that's what the fuller workarounds below do, but it purges the invented
# names, the failure with a supply-chain tail.
HALLUCINATION_STUDY = {"n": 6, "naive_resolve": 0.0, "grounded_resolve": 0.873}

# One real task: with no docs the model writes Python and invents a whole language;
# with the symbol table every name it emits exists.
HALLUCINATION_DEMO = {
    "task": "Take 3, -2, 8, -5; keep the positive ones; and count how many there are.",
    "naive": "input numbers = [3, -2, 8, -5]\noutput count = 0\nfor each number in "
             "numbers:\n    if number > 0:\n        count = count + 1\nprint count",
    "grounded": "feed 3 -2 8 -5\nonly even\neach negate\npack count",
}


def show_hallucinate(demo: dict = None, study: dict = None) -> None:
    """No docs: the model invents a language whole. With the symbol table in the
    prompt, every station and keyword it reaches for actually exists."""
    from genai.agent import show_code, show_turn
    demo = demo or HALLUCINATION_DEMO
    study = study or HALLUCINATION_STUDY
    show_turn("task", demo["task"])
    show_code("qwen, no docs", demo["naive"])
    inv = unresolved(demo["naive"])
    show_turn("INVENTED", "none of these exist in CONVEY: " + ", ".join(inv))
    show_turn("resolve", f"{resolve_rate(demo['naive'])} of its symbols are real")
    show_turn("LIBRARY", "hand the model the symbol table: feed, only, each, pack; "
              "even/odd/positive; double/negate/plus1; sum/max/min/count")
    show_code("qwen, with the table", demo["grounded"])
    show_turn("resolve", f"{resolve_rate(demo['grounded'])}: the invented names are "
              "gone (still not fluent, but nothing confabulated)")
    show_turn("across 6 tasks", f"symbol resolution {study['naive_resolve']} -> "
              f"{study['grounded_resolve']} with the table")

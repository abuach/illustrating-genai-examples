"""Chain-of-Verification (CoVe) helpers for the Augmentation chapter.

CoVe {cite}`dhuliawala2023cove` is a retrieval-free way to trim hallucinations: the
model checks its own work in four steps. (1) DRAFT a quick answer, which for a list
question often slips in a plausible-but-wrong entry. (2) PLAN a focused yes/no
question for each item in the draft. (3) EXECUTE each question INDEPENDENTLY, with
the draft NOT in context, so the model answers the small fact on its own rather than
rereading and rubber-stamping its first guess. This "factored" check is the crux of
why CoVe works. (4) REVISE by keeping only the items the model's own check confirmed.

The insight is that a narrow, isolated question ("does Portugal border Morocco?") is
easier to get right than the sprawling original, so the model catches its own slips.
The catch, which the numbers below make honest, is that CoVe can only ever be as good
as that factored check: when the model is confidently wrong about the small fact too,
the check repeats the mistake and CoVe drops a correct answer. That is the boundary
where self-verification stops and real retrieval has to take over.

The calls here mirror scripts/_cove_probe.py exactly (no system prompt, think=False,
the same prompts and token budgets), so COVE_STUDY at the bottom is a faithful record
of what these functions actually do on gemma4:latest.
"""
import re
import textwrap
import ollama
from genai.llm import SERVER, DEFAULT_MODEL

_client = ollama.Client(host=SERVER)

# The draft asks for a bare comma-separated list, which is where an over-eager extra
# item shows up most cleanly. The verify gives the model ROOM to answer in a sentence:
# pinning it to a single yes/no token collapses gemma4 to "no" on everything (even
# "is Lake Superior a Great Lake?"), so we let it reason briefly and read the verdict.
_DRAFT = " Reply with only the names, separated by commas, and nothing else."
_VERIFY = " Answer yes or no, then give a one sentence reason."


def _ask(prompt: str, model: str, temp: float, max_tokens: int) -> str:
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return (resp["message"]["content"] or "").strip()


def _canon(name: str) -> str:
    """Fold a name to a comparable key so 'the Netherlands' and 'Netherlands.' both
    reduce to 'netherlands': lowercase, drop a leading article, strip punctuation."""
    s = name.lower().strip().strip(".")
    s = re.sub(r"^(the|a)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _items(reply: str) -> list:
    """Split a comma or newline list into clean, de-duplicated item names, dropping
    any list bullets or numbering the model added."""
    out = []
    for tok in re.split(r"[,\n]", reply):
        c = _canon(re.sub(r"^[\-\*\d\.\)\s]+", "", tok))
        if c and c not in out:
            out.append(c)
    return out


def draft_answer(question: str, model: str = DEFAULT_MODEL, temp: float = 0.7) -> list:
    """Step 1: the quick baseline list answer, parsed into canonical item names."""
    return _items(_ask(question + _DRAFT, model, temp, 120))


def verify_claim(item: str, template: str,
                 model: str = DEFAULT_MODEL, temp: float = 0.7):
    """Steps 2-3: form a focused yes/no question for one item and answer it on its
    OWN (factored: the draft is not in context). Returns ``(confirmed, reason)``: the
    boolean verdict and the model's one-sentence justification, both real output."""
    reply = _ask(template.format(item=item) + _VERIFY, model, temp, 45)
    m = re.search(r"\b(yes|no)\b", reply.lower())
    confirmed = m is not None and m.group(1) == "yes"
    reason = re.sub(r"^\W*(yes|no)\b[\.,:\s]*", "", reply, flags=re.I).strip()
    return confirmed, reason


def list_precision(items: list, gold: set) -> float:
    """Of the items listed, the fraction that are actually in the gold set. This is
    the deterministic grader: a fixed correct set, no model judge, no eyeballing."""
    return sum(i in gold for i in items) / len(items) if items else 0.0


def list_recall(items: list, gold: set) -> float:
    """Of the truly correct items, the fraction that survived in the list. Recall is
    what CoVe quietly spends to buy precision: every wrongly-dropped item costs here."""
    hit = len({i for i in items if i in gold})
    return min(1.0, hit / _gold_size(gold))


# Surface forms that name the same place, so a recall denominator is not double-counted.
_SYNONYMS = [{"czech republic", "czechia"}, {"myanmar", "burma"}]


def _gold_size(gold: set) -> int:
    n = len(gold)
    for pair in _SYNONYMS:
        if pair <= gold:
            n -= 1
    return max(1, n)


# Curated list questions over STABLE, uncontroversial facts so the gold set is not up
# for debate and a factored "does X border Y?" check is a crisp fact the model knows
# in isolation. Each: (label, subject, draft question, verify template, gold). Subject
# drives the compact transcript phrasing. Chosen from scripts/_cove_probe.py to span
# the three honest regimes gemma4 actually shows (see COVE_STUDY).
COVE_QUESTIONS = [
    ("Portugal", "Portugal",
     "Name the countries that share a land border with Portugal.",
     "Does Portugal share a land border with {item}?",
     {"spain"}),
    ("Mexico", "Mexico",
     "Name the U.S. states that share a border with Mexico.",
     "Does the U.S. state of {item} share a border with Mexico?",
     {"california", "arizona", "new mexico", "texas"}),
    ("Brazil", "Brazil",
     "Name the South American countries that share a land border with Brazil.",
     "Does Brazil share a land border with {item}?",
     {"argentina", "bolivia", "colombia", "guyana", "paraguay", "peru",
      "suriname", "uruguay", "venezuela", "french guiana"}),
]


def _entry(label: str):
    """Look up a COVE_QUESTIONS entry by its label."""
    return next(e for e in COVE_QUESTIONS if e[0] == label)


def cove(label: str = "Portugal", model: str = DEFAULT_MODEL, temp: float = 0.7) -> dict:
    """Run the four factored CoVe steps on a curated list question.

    Draft a list, turn each listed item into its own focused yes/no check, answer
    each check independently, then keep only the confirmed items. Returns a dict with
    the draft, the per-item checks ``(item, confirmed, reason)``, and the revised list.
    Every model call is real, so any cell that calls this is frozen.
    """
    _, subject, question, template, gold = _entry(label)
    drafted = draft_answer(question, model, temp)
    checks = [(item, *verify_claim(item, template, model, temp)) for item in drafted]
    revised = [item for item, ok, _ in checks if ok]
    return {"subject": subject, "question": question, "template": template,
            "gold": gold, "draft": drafted, "checks": checks, "revised": revised}


def _join(items: list) -> str:
    return ", ".join(i.title() for i in items) if items else "(none)"


def _turn(label: str, body: str, tag: str = "", gutter: int = 9,
          width: int = 62, col: int = 66) -> None:
    """Print one transcript turn: a label in a fixed gutter, the body wrapped to fit
    the output box, and an optional tag (a score or a keep/drop call) right-aligned to
    a shared column so the annotations stack neatly. The tag rides the body's last
    line, so it stays put even when a long answer wraps. Continuations hang under the
    gutter."""
    lines = textwrap.wrap(body, width=width, break_on_hyphens=False) or [""]
    rows = [f"{label:<{gutter}}{lines[0]}"] + [" " * gutter + ln for ln in lines[1:]]
    if tag:
        rows[-1] += " " * max(1, col - len(rows[-1]) - len(tag)) + tag
    print("\n".join(rows))


def _reply(ok: bool, reason: str, width: int = 53) -> str:
    """Rebuild the model's spoken answer from its yes/no verdict and reason, trimmed to
    its first sentence so the turn stays one line. Faithful to what the verify step read
    back: the model led with the verdict, then gave the reason this restores in front."""
    sentence = " ".join(reason.split()).split(". ")[0].rstrip(".")
    line = f"{'Yes' if ok else 'No'}. {sentence}." if sentence else f"{'Yes' if ok else 'No'}."
    return line if len(line) <= width else line[:width].rsplit(" ", 1)[0] + "..."


def show_cove(label: str = "Portugal", model: str = DEFAULT_MODEL,
              result: dict = None) -> None:
    """Run CoVe on one list question and show it as a conversation: the user's draft
    question and the model's first answer, then the verification round where each name
    is put back to the model as its own yes/no question (the draft out of context) and
    the model's real reply drives a keep/drop call, then the revised answer the
    keep-only rule leaves behind. Nondeterministic, so the calling cell is frozen.

    Pass a precomputed ``result`` (from ``cove``) to render that exact run instead of
    rolling a fresh one; the capture script uses this to freeze a representative roll.
    """
    r = result if result is not None else cove(label, model)
    gold, draft, checks, revised = r["gold"], r["draft"], r["checks"], r["revised"]
    speaker = model.split(":")[0]
    _turn("USER", r["question"])
    _turn(speaker, _join(draft),
          tag=f"({len(draft)} listed, {list_precision(draft, gold):.0%} right)")
    _turn("VERIFY", "each name re-checked alone, the draft hidden:")
    for item, ok, reason in checks:
        _turn("USER", r["template"].format(item=item.title()))
        _turn(speaker, _reply(ok, reason), tag="keep" if ok else "drop")
    _turn("REVISE", _join(revised),
          tag=f"({len(revised)} listed, {list_precision(revised, gold):.0%} right)")


# Measured on gemma4:latest (the chapter's standard model), 12 samples per question at
# temperature 0.7, via scripts/_cove_probe.py. draft/cove are precision (fraction of
# listed items that are correct) and recall (fraction of correct items kept) before and
# after the factored check. The three questions span the honest range. Portugal is the
# WIN: the draft keeps slipping in Morocco (it sits one strait away, not one border), the
# factored check rejects it, and precision climbs with recall untouched. Mexico is the
# near-CEILING: the draft is almost always clean, so CoVe has little to fix and does no
# harm. Brazil is the LIMIT: gemma4 is sure it does not border Colombia, Bolivia, or Peru
# (it does), so the factored check repeats that wrong belief and drops correct countries
# -- precision cannot rise (the draft was already precise) and recall collapses. CoVe is
# only ever as reliable as its own check; where the model is confidently wrong, you need
# retrieval, not self-questioning.
COVE_STUDY = {
    "samples": 12,
    "labels":          ["Portugal", "Mexico", "Brazil"],
    "draft_precision": [0.75, 0.90, 1.00],
    "cove_precision":  [1.00, 1.00, 1.00],
    "draft_recall":    [1.00, 0.90, 0.62],
    "cove_recall":     [1.00, 0.90, 0.20],
}

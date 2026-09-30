"""Corrective RAG (CRAG) helpers for the Augmentation chapter.

CRAG {cite}`yan2024crag` makes ordinary RAG robust to bad retrieval. Plain RAG trusts
whatever the retriever hands back and answers from it, so when the corpus does not
hold the answer the model confidently makes something up. CRAG adds a cheap step in
between: a RETRIEVAL EVALUATOR grades the retrieved chunks for how well they actually
match the question and routes to one of three corrective actions. (1) CORRECT, the
docs look relevant, so we REFINE them, keeping the on-topic strips and dropping any
filler, and answer locally. (2) INCORRECT, the docs miss the mark, so we DISCARD them
and FALL BACK TO A WEB SEARCH for fresh knowledge. (3) AMBIGUOUS, when the grade is in
between, so we hedge and do BOTH, combining the refined local docs with web results.
Then the model answers from the corrected knowledge instead of from whatever happened
to be nearest in the index.

Two honest simplifications from the paper. The paper trains a small T5 evaluator; we
reuse the chapter's own embeddings and score relevance by cosine similarity of the
question to its best retrieved chunk, which is deterministic and needs no extra model.
And the paper's decompose-recompose refinement is a small pipeline of its own; since
our chunks are already short, we approximate it by keeping the strips that are
individually on-topic. The routing, the web fallback, and the lesson are faithful.

The web fallback calls the live internet through the `ddgs` package (DuckDuckGo, no API
key), so it is slow, rate-limited, nondeterministic, and sends the query off the
machine. Every cell that calls the model or the web is frozen with a saved snapshot;
re-running hits the live web and the result will drift. CRAG_STUDY at the bottom is a
faithful record of what these functions do on gemma4:latest, reproduced by
scripts/_crag_probe.py.
"""
import textwrap
import ollama
from genai.llm import SERVER, DEFAULT_MODEL
from genai.embed import embed, similarity
from genai.rag import build_wiki_store, wiki_retrieve_dense, rag

_client = ollama.Client(host=SERVER)


# ── The web-search fallback (the INCORRECT branch) ──────────────────────────────
def web_search(query: str, k: int = 3, timeout: int = 8) -> list:
    """Run a REAL web search and return up to ``k`` results as ``{title, snippet,
    url}`` dicts.

    This is the live-internet step. It calls DuckDuckGo through the ``ddgs`` package,
    which needs no API key, so it is the cheapest thing to hand a student, but it is
    also slow, rate-limited, and nondeterministic, and it sends the query off this
    machine. On any network or library error it returns an empty list rather than
    raising, so a notebook rebuild degrades gracefully instead of crashing.
    """
    try:
        from ddgs import DDGS
        hits = DDGS(timeout=timeout).text(query, max_results=k)
        return [{"title": h.get("title", ""), "snippet": h.get("body", ""),
                 "url": h.get("href", "")} for h in hits]
    except Exception:
        return []


# ── The retrieval evaluator (an embedding-similarity threshold) ─────────────────
# Tuned on the chapter's wiki_store (scripts/_crag_probe.py): clearly on-topic
# questions land at or above CORRECT_T, clearly off-topic ones at or below
# INCORRECT_T, and the band between is where the evaluator is unsure and hedges.
CORRECT_T, INCORRECT_T = 0.70, 0.60


def grade_retrieval(query: str, chunks: list):
    """Grade how well the retrieved chunks answer the query and return ``(score,
    verdict)``.

    The score is the cosine similarity of the query to its single best chunk,
    computed with the chapter's own embeddings, so it is deterministic and needs no
    trained grader. The verdict is CORRECT (>= CORRECT_T), INCORRECT (<= INCORRECT_T),
    or AMBIGUOUS in between. This is a proxy for the paper's learned evaluator and, as
    a proxy, it can misjudge: a topically-similar chunk that does not actually contain
    the answer can still score high.
    """
    if not chunks:
        return 0.0, "INCORRECT"
    qv = embed(query)
    score = max(similarity(qv, c) for c in chunks)
    verdict = ("CORRECT" if score >= CORRECT_T else
               "INCORRECT" if score <= INCORRECT_T else "AMBIGUOUS")
    return score, verdict


def refine(query: str, chunks: list, keep_t: float = 0.60) -> list:
    """The CORRECT branch's decompose-recompose, simplified: keep only the chunks
    that are individually on-topic for the query and drop the filler. Falls back to
    the full set if filtering would discard everything."""
    qv = embed(query)
    kept = [c for c in chunks if similarity(qv, c) >= keep_t]
    return kept or chunks


def _web_knowledge(query: str, k: int, web_fn) -> list:
    """Turn web hits into short context strings the generator can cite."""
    return [f"{h['title']}: {h['snippet']}" for h in web_fn(query, k)]


def crag(question: str, store=None, k: int = 3, model: str = DEFAULT_MODEL,
         web_fn=web_search) -> dict:
    """Run the full corrective pipeline: retrieve, grade, correct, then generate.

    Retrieve the top-k local chunks, grade them, and route. On CORRECT we refine the
    local chunks; on INCORRECT we drop them and search the web; on AMBIGUOUS we use
    both. The model then answers from that corrected knowledge. Returns a dict with
    every stage (chunks, score, verdict, web hits, final knowledge, answer) so the
    flow can be narrated. ``web_fn`` is injectable so a bake can cache the live call.
    Calls the model (and sometimes the web), so any cell that runs this is frozen.
    """
    store = store if store is not None else _wiki_store()
    chunks = [h["text"] for h in wiki_retrieve_dense(question, store, k)]
    score, verdict = grade_retrieval(question, chunks)
    web = []
    if verdict == "CORRECT":
        knowledge = refine(question, chunks)
    elif verdict == "INCORRECT":
        web = web_fn(question, k)
        knowledge = _web_knowledge(question, k, lambda q, kk: web)
    else:  # AMBIGUOUS: hedge and combine both sources
        web = web_fn(question, k)
        knowledge = refine(question, chunks) + _web_knowledge(question, k, lambda q, kk: web)
    answer = rag(question, knowledge, model=model)
    return {"question": question, "chunks": chunks, "score": score,
            "verdict": verdict, "web": web, "knowledge": knowledge, "answer": answer}


def standard_rag(question: str, store=None, k: int = 3,
                 model: str = DEFAULT_MODEL) -> str:
    """Plain RAG with no correction: retrieve the top-k local chunks and answer from
    them, no matter how relevant they are. This is the baseline CRAG is measured
    against, and the one that confidently makes things up when retrieval fails."""
    store = store if store is not None else _wiki_store()
    chunks = [h["text"] for h in wiki_retrieve_dense(question, store, k)]
    return rag(question, chunks, model=model)


def answer_hit(answer: str, gold: set) -> bool:
    """Deterministic grade: does the answer contain a known-correct keyword? ``gold``
    holds accepted lowercase surface forms; a hit means at least one appears. No model
    judge, no eyeballing, the same accepted-answer matching used elsewhere."""
    a = (answer or "").lower()
    return any(g in a for g in gold)


# ── Curated queries, picked so the three regimes are clean (scripts/_crag_probe.py) ──
# The first three are answerable from the chapter's wiki_store, so they grade CORRECT
# and standard RAG already gets them right; CRAG must match, not regress, and it must
# NOT waste a web call. The last two are traps: the corpus simply does not hold the
# answer, so they grade INCORRECT, standard RAG makes something up or begs off, and the
# web fallback recovers them. Each: (label, question, gold keywords).
CRAG_QUERIES = [
    ("CNN", "What kind of data are convolutional neural networks designed for?",
     {"image", "grid", "spatial", "pixel", "vision"}),
    ("BERT", "Is BERT an encoder or a decoder language model?", {"encoder"}),
    ("Embedding", "What does a word embedding represent in machine learning?",
     {"vector", "represent", "meaning", "semantic", "similar"}),
    ("Nobel '24", "Who won the 2024 Nobel Prize in Physics?", {"hinton", "hopfield"}),
    ("Olympics", "Which city hosted the 2024 Summer Olympics?", {"paris"}),
]


def _query(label: str):
    """Look up a CRAG_QUERIES entry by its label."""
    return next(q for q in CRAG_QUERIES if q[0] == label)


def _wrap(text: str, width: int, n: int = 2) -> list:
    """First ``n`` wrapped lines of a single-spaced string."""
    return textwrap.wrap(" ".join(text.split()), width=width,
                         break_on_hyphens=False)[:n]


# ── Line budget for the transcript ──────────────────────────────────────────────
# The PDF output box holds 68 monospace chars at the book's scale before a line
# overflows (69 is right on the edge and wraps unpredictably — measured in the
# rendered PDF). The label gutter ("RETRIEVE  ->  ") and its matching continuation
# indent are 14 wide, so every body must wrap to <= _BODY; a longer line overflows
# the box and LaTeX wraps it a SECOND time into a `↪` staircase with a word or two
# left dangling. Deriving every width below from _BODY keeps that from coming back
# the next time this demo is re-baked. See [[transcript-wrap-indent]].
_BOX, _GUT = 68, 14
_BODY = _BOX - _GUT          # 54: a wrapped body line
_IND = " " * _GUT            # continuation indent, aligned under the body


def show_crag(label: str = "Nobel '24", store=None, result: dict = None) -> None:
    """Run CRAG on one curated query and show the corrective flow end to end: the
    question, the text of the best local chunk with its relevance score, the grade and the branch
    it triggers, the live web query and the snippet actually used (only when the grade
    sends us there), the corrected answer, and a numbered SOURCES list of the corrected
    knowledge so the answer's ``[n]`` citations resolve instead of dangling.
    Nondeterministic, so the calling cell is frozen. Pass a precomputed ``result``
    (from ``crag``) to render that exact run."""
    store = store if store is not None else _wiki_store()
    r = result if result is not None else crag(label_to_q(label), store)
    print(r["question"])
    best = r["chunks"][0] if r["chunks"] else ""
    print(f"RETRIEVE  ->  best of {len(r['chunks'])} local chunks, "
          f"similarity {r['score']:.2f}")
    if best:
        print(f'{_IND}"{_clip(best, _BODY - 2)}"')          # 2 for the "" marks
    if r["verdict"] == "CORRECT":
        kept, n = len(refine(r["question"], r["chunks"])), len(r["chunks"])
        detail = "all on topic" if kept == n else f"dropped {n - kept} off-topic"
        print(f"GRADE     ->  CORRECT: local docs look relevant, no web call needed")
        print(f"REFINE    ->  kept {kept} of {n} chunks ({detail})")
    else:
        tail = "search the web too" if r["verdict"] == "AMBIGUOUS" else "fall back to the web"
        print(f"GRADE     ->  {r['verdict']}: local docs miss it, {tail}")
        print(f'WEB       ->  sent "{r["question"]}"')
        if r["web"]:
            h = _best_hit(r["answer"], r["web"])
            for line in _wrap(f'{_domain(h["url"])}: "{h["snippet"]}"', _BODY, 2):
                print(f"{_IND}{line}")
    ans = _wrap(r["answer"], _BODY, 2)
    print(f"ANSWER    ->  {ans[0] if ans else '(none)'}")
    for line in ans[1:]:
        print(f"{_IND}{line}")
    if r["knowledge"]:
        print(f"SOURCES   ->  [1] {_clip(r['knowledge'][0], _BODY - 4)}")  # 4 for "[n] "
        for i, k in enumerate(r["knowledge"][1:], 2):
            print(f"{_IND}[{i}] {_clip(k, _BODY - 4)}")


def label_to_q(label: str) -> str:
    """The question text for a curated label (so the demo cell stays a one-liner)."""
    return _query(label)[1]


def _clip(text: str, width: int) -> str:
    """Collapse whitespace and cap the text at ``width`` columns, ending in an
    ellipsis when it had to cut. A true hard cap: the result, ellipsis and all,
    is never longer than ``width``, so a caller can budget exact box room. The
    cut lands on a word boundary so it never ends mid-word."""
    one = " ".join((text or "").split())
    if len(one) <= width:
        return one
    return one[:width - 3].rsplit(" ", 1)[0] + "..."


def _best_hit(answer: str, web: list) -> dict:
    """The single web result the answer leaned on most, by embedding similarity of
    each result to the generated answer, so the snippet we display is the one that
    actually fed the correction rather than just the top-ranked link."""
    av = embed(answer)
    return max(web, key=lambda h: similarity(av, h["title"] + " " + h["snippet"]))


def _domain(url: str) -> str:
    """The bare host of a result URL (nobelprize.org), so the transcript names its
    source without printing a full link."""
    from urllib.parse import urlparse
    host = urlparse(url).netloc or url
    return host[4:] if host.startswith("www.") else host


# ── Lazily-built corpus, shared with the rest of the chapter ────────────────────
_STORE = None


def _wiki_store():
    """Build (once) the same wiki_store the chapter uses, so CRAG runs on the very
    corpus the reader already met in Retrieval at Scale."""
    global _STORE
    if _STORE is None:
        _STORE, _ = build_wiki_store()
    return _STORE


# Measured on gemma4:latest (the chapter's standard model), 5 tries per query, via
# scripts/_crag_probe.py; grading is the deterministic keyword check in answer_hit. For
# each query, `standard` is the share of plain-RAG answers that contain the correct
# fact and `crag` is the share of corrected answers that do. The first three questions
# are answerable from the wiki_store: they grade CORRECT, so CRAG keeps the local docs,
# skips the web, and matches the baseline -- the no-regression case. The last two are
# traps the corpus cannot answer: they grade INCORRECT, standard RAG either begs off or
# invents, and the web fallback recovers the fact. The difference is exactly the trap queries,
# which is the whole point: CRAG buys robustness to retrieval failure, not omniscience.
# (The reproducer also measures a fourth regime not charted here, a FALSE CORRECT: the
# "how does a neural network learn from its errors" question grades CORRECT at 0.75 yet
# the corpus does not hold the mechanism, so CRAG trusts bad docs and misses just as
# standard RAG does -- the evaluator's blind spot, discussed in the read-out.)
CRAG_STUDY = {
    "samples": 5,
    "labels":   ["CNN", "BERT", "Embedding", "Nobel '24", "Olympics"],
    "standard": [1.0, 1.0, 1.0, 0.0, 0.0],
    "crag":     [1.0, 1.0, 1.0, 1.0, 1.0],
}

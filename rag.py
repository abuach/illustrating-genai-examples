"""Retrieval-Augmented Generation utilities."""
import textwrap
import numpy as np
from genai.embed import embed, similarity
from genai.llm import chat, DEFAULT_MODEL


def chunk(text: str, size: int = 200, overlap: int = 50) -> list:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i: i + size]) for i in range(0, len(words), step) if words[i: i + size]]


class DocumentStore:
    """Minimal vector store: add documents, search by similarity, or ask questions."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._docs = []
        self._vecs = []
        self._model = model

    def add(self, text: str, metadata: dict = None):
        self._docs.append({"text": text, "meta": metadata or {}})
        self._vecs.append(embed(text))

    def texts(self) -> list:
        """Return every stored document's text, in insertion order."""
        return [d["text"] for d in self._docs]

    def search(self, query: str, k: int = 3) -> list:
        """Return the k most relevant document texts."""
        q = embed(query)
        scored = sorted(
            zip(self._docs, self._vecs),
            key=lambda x: similarity(q, x[1]),
            reverse=True,
        )
        return [d["text"] for d, _ in scored[:k]]

    def ask(self, question: str, k: int = 3) -> str:
        """Full RAG: retrieve then generate a cited answer."""
        return rag(question, self.search(question, k), model=self._model)


import re as _re


def _clean(text: str) -> str:
    text = _re.sub(r'<[^>]+>', ' ', text)
    text = _re.sub(r'\s+', ' ', text)
    return _re.sub(r'[^\x00-\x7F]+', ' ', text).strip()


def _contextualize(message: str, history: list) -> str:
    from genai.llm import ask as _ask
    if not history:
        return message
    ctx = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-4:])
    return _ask(
        f"Rewrite as a standalone search query.\n"
        f"Conversation:\n{ctx}\nFollow-up: {message}\nStandalone query:",
        max_tokens=40)


WIKI_TOPICS = [
    "Transformer_(machine_learning_model)", "BERT_(language_model)", "GPT-3",
    "Attention_(machine_learning)", "Backpropagation", "Gradient_descent",
    "Neural_network_(machine_learning)", "Convolutional_neural_network",
    "Recurrent_neural_network", "Long_short-term_memory",
    "Word2vec", "GloVe_(machine_learning)", "Tokenization_(lexical_analysis)",
    "Prompt_engineering", "Fine-tuning_(deep_learning)",
    "Reinforcement_learning", "Transfer_learning",
    "Natural_language_processing", "Named-entity_recognition", "Sentiment_analysis",
    "Diffusion_model", "Generative_adversarial_network", "Variational_autoencoder",
    "Stable_Diffusion", "DALL-E",
    "Hallucination_(artificial_intelligence)", "Retrieval-augmented_generation",
    "Vector_database", "Cosine_similarity", "Semantic_search",
    "LangChain", "Hugging_Face", "OpenAI", "Anthropic_(company)",
    "TensorFlow", "PyTorch", "Machine_learning",
    "Optical_character_recognition", "Deepfake", "Federated_learning",
]


def build_wiki_store(cache: str = "_wiki_cache.json",
                     chunk_sz: int = 25, chunk_ov: int = 5,
                     min_words: int = 8,
                     topics: list = None,
                     clean_fn=None):
    """Fetch or load cached Wikipedia summaries, chunk them, and return a DocumentStore."""
    import json as _json2, time as _time2
    try:
        import requests as _req
    except ImportError:
        _req = None

    if topics is None:
        topics = WIKI_TOPICS
    if clean_fn is None:
        clean_fn = _clean

    # Load or fetch
    try:
        with open(cache) as f:
            wiki_docs = _json2.load(f)
    except FileNotFoundError:
        wiki_docs = []
        hdr = {"User-Agent": "ProgrammingGenAI-Textbook/1.0 (educational)"}
        for title in topics:
            try:
                r = _req.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    headers=hdr, timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    text = d.get("extract", "")
                    if len(text) > 80:
                        wiki_docs.append({"title": d.get("title", title),
                                          "text": text, "topic": title})
            except Exception:
                pass
            _time2.sleep(1.0)
        with open(cache, "w") as f:
            _json2.dump(wiki_docs, f)

    store = DocumentStore()
    for doc in wiki_docs:
        for c in chunk(doc["text"], size=chunk_sz, overlap=chunk_ov):
            if len(c.split()) >= min_words:
                store.add(clean_fn(c), {"topic": doc["topic"]})
    return store, wiki_docs


def wiki_retrieve_dense(query: str, store: "DocumentStore", k: int) -> list:
    """Return top-k chunks by dense (embedding) similarity."""
    q_vec = embed(query)
    ranked = sorted(range(len(store._docs)),
                    key=lambda i: similarity(q_vec, store._vecs[i]),
                    reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def wiki_retrieve_sparse(query: str, kw_scorer, store: "DocumentStore",
                         k: int) -> list:
    """Return top-k chunks by keyword (BM25-style) scoring."""
    ranked = sorted(range(len(store._docs)),
                    key=lambda i: kw_scorer(query, store._docs[i]["text"]),
                    reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def wiki_retrieve_hybrid(query: str, kw_scorer, store: "DocumentStore",
                         alpha: float, k: int) -> list:
    """Return top-k chunks by linear combination of dense and sparse scores."""
    q_vec = embed(query)
    n = len(store._docs)
    ds = {i: similarity(q_vec, store._vecs[i]) for i in range(n)}
    ks = {i: kw_scorer(query, store._docs[i]["text"]) for i in range(n)}
    norm = lambda d: {i: v / (max(d.values()) or 1) for i, v in d.items()}
    dn, kn = norm(ds), norm(ks)
    comb = {i: alpha * dn[i] + (1 - alpha) * kn[i] for i in range(n)}
    ranked = sorted(comb, key=lambda i: comb[i], reverse=True)[:k]
    return [{"text": store._docs[i]["text"],
             "meta": store._docs[i]["meta"]} for i in ranked]


def topic_p_at_k(results: list, target_topic: str, k: int) -> float:
    """Precision@k: fraction of top-k results from the target topic."""
    hits = sum(1 for r in results[:k] if r["meta"]["topic"] == target_topic)
    return hits / k if k else 0.0


def topic_recall_at_k(results: list, target_topic: str, k: int,
                      store: "DocumentStore") -> float:
    """Recall@k: fraction of topic chunks retrieved in top k."""
    n_rel = sum(1 for d in store._docs if d["meta"]["topic"] == target_topic)
    hits  = sum(1 for r in results[:k] if r["meta"]["topic"] == target_topic)
    return hits / n_rel if n_rel else 0.0


def rag(question: str, docs: list, model: str = DEFAULT_MODEL) -> str:
    """Generate an answer grounded in the provided docs, with [n] citations."""
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    messages = [
        {"role": "system", "content": "Answer using only the context. Cite sources [1],[2]… If unsure, say so."},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat(messages, model=model)


# ────────────────────────────────────────────────────────────────────
# Seed data for augmentation chapter demos.
# ────────────────────────────────────────────────────────────────────

# Two verbatim excerpts from this book run together as one block: the
# Prompting chapter on the sampler, then the Multimodal chapter on how a
# vision model reads an image. Real text with a real topic seam, which is
# exactly what the chunking demos need.
LONG_DOC = (
    "At every step, the model hands us a ranked list of possible next tokens "
    "with scores. Then a sampler rolls a weighted die to pick one. Temperature "
    "controls how loaded the die is. The randomness is not coming from the "
    "model. It is coming from the sampler we strapped onto the front of it. "
    'How does a model "see" an image? A language model processes words one '
    "token at a time. But how does it process an image, which is just a grid "
    "of millions of colored pixels? The answer is in patches. A vision model "
    "slices the image into a regular grid of small squares, and treats each "
    "square as a single token."
)

WIKI_EVAL_QUERIES = [
    # Each query is chosen to expose a different retrieval strength, so the
    # contrast between strategies has somewhere to show up.
    ("Transformers",                              # conceptual paraphrase: dense wins
     "how do transformers use self-attention",
     "Transformer_(machine_learning_model)"),
    ("BERT",                                      # bare exact term: sparse wins, dense whiffs
     "BERT",
     "BERT_(language_model)"),
    ("Semantic",                                  # weak on either alone: hybrid beats both
     "find documents by meaning instead of matching exact keywords",
     "Semantic_search"),
    ("Diffusion",                                 # words match literally and by meaning: all tie
     "how diffusion models generate images from noise",
     "Diffusion_model"),
]

WIKI_GOLDEN_QUERIES = [
    ("how do transformers use self-attention in sequence modeling",
     "Transformer_(machine_learning_model)"),
    ("how recurrent networks handle sequential data",
     "Recurrent_neural_network"),
    ("how word2vec learns word similarity using neural networks",
     "Word2vec"),
    ("GAN generator and discriminator adversarial training",
     "Generative_adversarial_network"),
    ("BERT encoder bidirectional training",
     "BERT_(language_model)"),
]

INJECTION_SIGNALS = [
    "ignore previous", "disregard", "system prompt",
    "you are now", "from now on", "override",
    "new instructions", "act as",
]


# ────────────────────────────────────────────────────────────────────
# Lost in the middle: why retrieval beats a giant context window.
# ────────────────────────────────────────────────────────────────────
# Even when a long document fits the window, a model recalls a fact buried
# in the middle far worse than one at the edges. Run live in the Augmentation
# chapter as the payoff to the "why chunk at all?" FAQ.
# A plant has to clear two bars at once, and they pull against each other.
#
#   answerable  the prompt below asks for one word, so the accepted string has
#               to be what that one word would actually be. "What date is the
#               autumn festival?" graded against "22" failed this: a one-word
#               date is a month, so the plant was a guaranteed miss for every
#               model and quietly depressed the edge bars. Asking for the day of
#               the month is no better, since models answer "Twenty-second".
#   unguessable a model that never finds the sentence must not be able to answer
#               anyway. Asking "in which month is the autumn festival?" clears
#               the first bar and fails this one badly: with the fact deleted
#               from the context entirely, gemma4 and qwen3 still answer
#               "October", because that is simply a good guess about autumn.
#
# So every answer here is an arbitrary token the filler never mentions and no
# amount of world knowledge supplies. Re-run the control in
# scripts/_augmentation_context_probe.py --control before changing any of them.
_PLANTS = [
    ("The vault passphrase is BLUEHERON.", "What is the vault passphrase?", "blueheron"),
    ("Dr. Okafor's office is in room 417.", "Which room is Dr. Okafor's office?", "417"),
    ("The rover touched down in the Gale crater.", "Where did the rover land?", "gale"),
    ("This year's autumn festival theme is LANTERNS.",
     "What is the autumn festival's theme?", "lanterns"),
]
_FILLER = [f"Log entry {i}: on day {i} the maintenance crew inspected unit {i} and "
           f"confirmed that every reading on gauge {i} stayed within the normal "
           f"operating range for the quarter." for i in range(200)]   # ~7k tokens


def _haystack(fact: str, position: str) -> str:
    if position == "start":
        items = [fact] + _FILLER
    elif position == "end":
        items = _FILLER + [fact]
    else:
        mid = len(_FILLER) // 2
        items = _FILLER[:mid] + [fact] + _FILLER[mid:]
    return "\n".join(items)


POSITIONS = ("start", "middle", "end")


def score_context(models: list = None, runs: int = 3) -> dict:
    """For each model, recall of a fact planted at the START, MIDDLE and END of a
    long context, averaged over several facts. Behind plot_context / CONTEXT_STUDY.

    Scored per position rather than averaging start and end into one "edges"
    number. That average was hiding the actual result: these models recall the
    end almost perfectly and the start no better than the middle, so folding the
    two together turned a 100/0 split into a flat 50% and made the start look
    like a place a fact could survive.

    Each score is the median of ``runs`` passes. Four plants means a single pass
    can only land on 0, 25, 50, 75 or 100, and generation is sampled, so one
    lucky pass is enough to crown a model that cannot repeat it (llama3.1 once
    scored 50% in the middle and earned a context-window explanation it did not
    deserve).
    """
    from genai.llm import ask
    from genai.agent import AGENT_BENCH_MODELS
    models = models or AGENT_BENCH_MODELS
    recall = lambda fact, q, a, pos, m: a in ask(
        f"Context:\n{_haystack(fact, pos)}\n\n{q} Answer in one word.",
        model=m, think=False, max_tokens=15).lower()
    study = {}
    for m in models:
        row = {}
        for pos in POSITIONS:
            hits = sorted(sum(recall(f, q, a, pos, m) for f, q, a in _PLANTS)
                          for _ in range(runs))
            row[pos] = round(100 * hits[len(hits) // 2] / len(_PLANTS), 1)
        study[m] = row
    return study


# RETIRED 2026-09-26: this table measured Ollama's default 4,096-token num_ctx,
# not lost-in-the-middle. score_context never raised num_ctx, so the ~7k-token
# prompt was cut from the front and only the END survived. Re-probed with
# num_ctx=16384, gemma4/qwen3/qwen3.5 found every fact at every position and
# mistral only dipped at the START. The Augmentation demo was removed and
# the FAQ cites RULER and NoLiMa instead. Kept for the record only.
#
# Measured by scripts/_augmentation_context_probe.py with a ~200-line, ~7k-token
# filler, each score the median of three passes. At THIS length the result is
# unanimous and much sharper than the lost-in-the-middle effect it was built to
# show: every model recalls a fact planted at the END perfectly and one planted
# at the START not once, with the middle no worse than the start. All six score
# 100/0/0. (At the earlier ~40-line length the effect vanished, all positions
# alike -- the null is in this file's git history; the effect is a function of
# context length.)
#
# functiongemma is left out: it reads nothing from any position, so it has no
# positional shape to show and only adds a column of zeroes. It stays in
# AGENT_BENCH_MODELS, so pass an explicit model list to reproduce this table.
#
# This used to average start and end into one "edges" number, which reported a
# flat 50% for nearly every model and read as "sometimes finds it near an edge".
# It was really 100% at the end and 0% at the start, so the average was hiding
# the finding rather than summarising it. Keep the three positions apart.
#
# Two earlier versions of this study were wrong, both fixed:
#   - a plant asked for a date but was graded on "22" while the prompt asks for
#     one word, so it could never hit and dragged every edge score down
#   - the replacement asked which month, which gemma4 and qwen3 answer "October"
#     with the fact deleted from the context entirely. Run the probe with
#     --control before touching _PLANTS: it deletes each fact and asks anyway,
#     and every answer has to miss.
CONTEXT_STUDY = {
    "gemma4:latest":        {"start": 0.0, "middle": 0.0, "end": 100.0},
    "qwen3:latest":         {"start": 0.0, "middle": 0.0, "end": 100.0},
    "qwen3.5:latest":       {"start": 0.0, "middle": 0.0, "end": 100.0},
    "mistral:latest":       {"start": 0.0, "middle": 0.0, "end": 100.0},
    "ministral-3:latest":   {"start": 0.0, "middle": 0.0, "end": 100.0},
    "llama3.1:latest":      {"start": 0.0, "middle": 0.0, "end": 100.0},
}


# --- retrieval-demo display helpers ----------------------------------------
# Each keeps a demo cell to its inputs plus one call; the ranking, slicing,
# and column formatting live here instead of on the page.

def preview_store(store, lo: int, hi: int, box: int = 68) -> None:
    """Print a few stored passages: chapter / heading -> opening text. The text
    preview is budgeted against the fixed label so the whole line fits the output
    box (68 mono chars) and never overflows into a `↪` wrap. See
    [[transcript-wrap-indent]]."""
    for d in store._docs[lo:hi]:
        label = f"{d['meta']['chapter']} / {d['meta']['heading'][:20]} → "
        print(f"{label}{d['text'][:max(0, box - len(label))]}")


def show_chunks(text: str, size: int = 200, overlap: int = 0,
                width: int = 64) -> None:
    """Split ``text`` into fixed-size chunks and print the start of each."""
    for i, c in enumerate(chunk(text, size=size, overlap=overlap)):
        print(f"Chunk {i}: {c[:width]}")


def show_semantic_seam(text: str) -> None:
    """Score neighbouring-sentence similarity and flag the weakest seam."""
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.?]) ', text) if s.strip()]
    sims = [similarity(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]
    seam = sims.index(min(sims))
    print("Sentence-to-sentence similarity:")
    for i, s in enumerate(sims):
        tag = "◀ SPLIT" if i == seam else ""
        print(f"  {s:.3f}  {sents[i][:50]!r} {tag}")


def show_dense_retrieval(store, query: str, k: int = 3) -> None:
    """Rank stored passages by embedding cosine similarity to the query."""
    q_vec = embed(query)
    scored = sorted(enumerate(store._docs),
                    key=lambda t: similarity(q_vec, store._vecs[t[0]]),
                    reverse=True)
    print("Dense retrieval results:")
    for i, d in scored[:k]:
        score = similarity(q_vec, store._vecs[i])
        print(f"  {score:.3f}  [{d['meta']['chapter']}] {d['text'][:48]}")


def show_sparse_retrieval(store, query: str, kw_score, k: int = 3) -> None:
    """Rank stored passages by keyword overlap (the kw_score from the cell above)."""
    ranked = sorted(store._docs, key=lambda d: kw_score(query, d['text']),
                    reverse=True)
    print(f"Sparse retrieval for exact term '{query}':")
    for d in ranked[:k]:
        print(f"  {kw_score(query, d['text']):.4f}  "
              f"[{d['meta']['chapter']}] {d['text'][:48]}")


def show_hits(hits) -> None:
    """Print ranked search hits: score, chapter, opening text."""
    for r in hits:
        print(f"  {r['score']:.3f}  [{r['chapter']}] {r['text'][:48]}")


def show_grounded(answer: str, sources: list, width: int = 61,
                  box: int = 68, check: str = None) -> None:
    """Print a grounded answer, then the numbered source list its ``[n]``
    citations point to, so no bracket is left dangling on the page. The
    sources are the exact passages the model was handed, in citation order,
    each collapsed to one trimmed line. Pass ``check`` (a hallucination-check
    verdict) to print it under the sources.

    The answer, the source lines, and the check all wrap to fit the ``box`` (the
    PDF output cell holds 68 monospace chars before a line overflows). The source
    width leaves room for its ``"  [n] "`` prefix and the trailing ellipsis;
    wrapping the answer keeps a long paragraph from overflowing the box and being
    re-wrapped into a ``↪`` staircase. See [[transcript-wrap-indent]]."""
    def wrapped(text):
        return textwrap.wrap(" ".join(text.split()), width=box,
                             break_on_hyphens=False) or [""]
    for line in wrapped(answer):
        print(line)
    print("\nSOURCES")
    for i, s in enumerate(sources, 1):
        line = " ".join(s.split())
        print(f"  [{i}] {line[:width]}{'…' if len(line) > width else ''}")
    if check is not None:
        print()
        for line in wrapped(f"Check: {check}"):
            print(line)

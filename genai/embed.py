"""Text embeddings and semantic similarity."""
import numpy as np
import ollama
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)
EMBED_MODEL = "nomic-embed-text"


def embed(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    """Return embedding vector for a piece of text."""
    resp = _client.embeddings(model=model, prompt=text)
    return np.array(resp["embedding"])


def similarity(a, b) -> float:
    """Cosine similarity between two vectors (or two strings)."""
    if isinstance(a, str): a = embed(a)
    if isinstance(b, str): b = embed(b)
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search(query: str, docs: list, top_k: int = 3) -> list:
    """Return the top_k most semantically similar docs to the query."""
    q = embed(query)
    scored = sorted(docs, key=lambda d: similarity(q, embed(d)), reverse=True)
    return scored[:top_k]


def word_analogy(a: str, b: str, c: str, candidates: list) -> tuple:
    """Solve: a is to b as c is to ?  Returns (best_word, score)."""
    target = embed(a) - embed(b) + embed(c)
    exclude = {x.lower() for x in [a, b, c]}
    best, best_score = None, -1
    for w in candidates:
        if w.lower() in exclude:
            continue
        s = similarity(target, embed(w))
        if s > best_score:
            best, best_score = w, s
    return best, round(best_score, 4)


# --- display helpers (keep demo cells to a single readable call) ------------

def show_word_overlap(s1: str, s2: str, note: str = "") -> None:
    """Jaccard word overlap: the naive baseline that can't see synonyms."""
    words = lambda s: set(s.lower().split())
    overlap = len(words(s1) & words(s2)) / len(words(s1) | words(s2))
    tail = f"  ({note})" if note else ""
    print(f"Word overlap: {overlap:.2f}{tail}")


def show_skipgram_pairs(text: str, window: int = 2, n: int = 8) -> None:
    """Skip-gram training pairs: each word paired with neighbours in the window."""
    w = text.split()
    pairs = [(w[i], w[j])
             for i in range(len(w))
             for j in range(max(0, i - window), min(len(w), i + window + 1))
             if i != j]
    print(pairs[:n])


def averaged_similarity(a: str, b: str) -> float:
    """Cosine of two sentences' averaged word vectors (the bag-of-words baseline)."""
    avg = lambda s: np.mean([embed(w) for w in s.split()], axis=0)
    return similarity(avg(a), avg(b))


def show_averaged_word_similarity(pairs: list) -> None:
    """Average each sentence's word vectors, then compare (a weak bag-of-words trick).

    ``pairs`` is a list of ``(label, sentence_a, sentence_b)``.
    """
    cells = [f"{label}={averaged_similarity(a, b):.3f}" for label, a, b in pairs]
    print("Averaged words  " + "  ".join(cells))


# (label, anchor, related-paraphrase, unrelated) for the related/unrelated spread demo.
CONTRAST_PAIRS = [
    ("cat / feline",       "The cat sat on the mat",          "A feline rested on the rug",           "Python is a programming language"),
    ("car / automobile",   "He parked the car outside",       "She left the automobile by the curb",  "The orchestra tuned their instruments"),
    ("rain / downpour",    "It rained hard all afternoon",    "There was a downpour all day",         "She solved the equation quickly"),
    ("ocean / sea",        "They sailed across the ocean",    "They crossed the open sea",            "He debugged the software overnight"),
    ("doctor / physician", "The doctor examined the patient", "The physician checked the patient",    "A rocket launched toward the moon"),
]


# Measured by scripts/_embeddinggemma_probe.py: cosine of "The cat sat on the mat."
# to its translation in each language, under two multilingual embedders --
# nomic-embed-text-v2-moe (a mixture-of-experts model) and embeddinggemma:300m.
# Embeddings are deterministic, so these are exact. Both lift the well-resourced
# languages far above the unrelated-control floor (embeddinggemma highest), and both
# sink toward it on Swahili (v2-moe 0.22, onto its own floor; embeddinggemma 0.38), the
# low-resource edge of their coverage. (nomic v2's search_query: prefix changed nothing,
# so these are the book's raw-call numbers.)
# Ordered by descending similarity (then the unrelated floor last) so the line
# chart reads as a clean descent into the low-resource cliff; both models rank the
# languages the same way, so the order is the same under either.
CROSSLINGUAL_STUDY = {
    "labels":   ["German", "French", "Chinese", "Spanish", "Swahili", "unrelated"],
    "nomic_v2": [0.808, 0.721, 0.612, 0.581, 0.216, 0.210],
    "gemma":    [0.921, 0.858, 0.818, 0.816, 0.375, 0.197],
}


def show_opposite_cosines() -> None:
    """Compare antonym pairs against reference pairs.

    The point of the table: real opposites score high and positive, nowhere
    near the -1 the cosine scale allows, and higher than two unrelated words.
    """
    opposites = [("happy", "sad"), ("rich", "poor"), ("hot", "cold"),
                 ("good", "bad"), ("true", "false")]
    refs = [("happy", "economics"), ("happy", "stapler"),
            ("cat", "car"), ("happy", "happy")]
    block = lambda a, b: f"{a + ' / ' + b:<17}{similarity(a, b):>+7.2f}"
    head = lambda name: f"{name:<17}{'cosine':>7}"
    print(head("opposites") + "   " + head("reference"))
    for i, opp in enumerate(opposites):
        right = "   " + block(*refs[i]) if i < len(refs) else ""
        print(block(*opp) + right)


def show_offset_consistency() -> None:
    """Cosine between two displacement vectors, the step ``a -> b`` in meaning-space.

    A high score means both pairs move along the *same* arrow, which is what
    lets an analogy carry from one pair to another. Gender and comparison share
    an arrow; opposites do not.
    """
    step = lambda a, b: embed(b) - embed(a)
    rows = [
        ("gender",      ("man", "woman"),  ("king", "queen")),
        ("gender",      ("man", "woman"),  ("actor", "actress")),
        ("comparative", ("big", "bigger"), ("tall", "taller")),
        ("comparative", ("big", "bigger"), ("small", "smaller")),
        ("opposite",    ("happy", "sad"),  ("rich", "poor")),
        ("opposite",    ("happy", "sad"),  ("hot", "cold")),
        ("opposite",    ("rich", "poor"),  ("hot", "cold")),
        ("opposite",    ("big", "small"),  ("hot", "cold")),
    ]
    name = lambda p: f"{p[0]}->{p[1]}"
    print(f"{'relationship':<14}{'step A':<14}{'step B':<16}{'cosine'}")
    for rel, a, b in rows:
        print(f"{rel:<14}{name(a):<14}{name(b):<16}{similarity(step(*a), step(*b)):+.2f}")

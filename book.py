"""The book as a corpus: this module turns the chapters of *Programming
Generative AI* into clean retrieval passages and indexes them in a real
vector database (Chroma).

The corpus is built from the same notebooks the reader is holding: every
markdown cell is cleaned of MyST markup (directives, citation keys, code
fences), tagged with the chapter and section it came from, and saved to a
committed snapshot, ``book_corpus.json``. Demos run against the snapshot, not
the live chapters, so frozen outputs stay reproducible while the book is
edited; the snapshot is regenerated once near print.

The same notebooks yield two companion snapshots, ``book_code_corpus.json``
and ``book_output_corpus.json``, holding the code cells and their saved
outputs. The book's executable half is content too, so it gets its own
corpora alongside the prose.
"""
import json
import re
from pathlib import Path

from genai.embed import embed
from genai.rag import rag

_PKG = Path(__file__).resolve().parent


def _find_chapters() -> Path:
    """The book's ``chapters/`` directory, holding the notebooks the corpus is
    built from.

    genai is a shared package at the monorepo root, so the notebooks aren't
    next to this file anymore. They're book-specific: notebooks execute with
    the book's chapters/ as the cwd, so resolve there first, then fall back to
    the igai book relative to the repo root so a corpus build also works when
    run from the repo root.
    """
    for cand in (Path.cwd(), _PKG.parent / "books" / "igai" / "chapters"):
        if (cand / f"{SPINE[0]}.ipynb").exists():
            return cand
    return Path.cwd()


SNAPSHOT = _PKG / "book_corpus.json"
CODE_SNAPSHOT = _PKG / "book_code_corpus.json"
OUTPUT_SNAPSHOT = _PKG / "book_output_corpus.json"

# The book's spine, in reading order (front = the preface).
SPINE = ["front", "intro", "prompting", "tokens", "semantics",
         "metacoding", "augmentation", "agentic", "thinking", "mm",
         "responsible", "efficiency"]

_DIRECTIVE = re.compile(r"^```\{(\w+)\}\s*(.*)$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")          # terminal color codes in tracebacks


def _clean_markdown(text: str) -> str:
    """Strip MyST and markdown markup from one cell, keeping only the prose.

    Directive fences keep their body (an admonition's story is book content);
    code fences, raw blocks, tables, target labels, and image lines are
    dropped; inline roles, citation keys, links, and emphasis markers are
    unwrapped to plain text.
    """
    lines, out, in_code, in_raw = text.split("\n"), [], False, False
    for line in lines:
        s = line.strip()
        if in_raw:
            in_raw = not s.startswith(":::")
            continue
        if s.startswith(":::"):
            in_raw = True
            continue
        if m := _DIRECTIVE.match(s):          # keep an admonition's title...
            out.append(m.group(2))
            continue
        if s.startswith("```"):               # ...but drop code fence bodies
            in_code = not in_code
            continue
        if in_code or s.startswith(("#", "![", "|", ":")):
            continue
        if re.match(r"^\(.*\)=$", s):         # (chap:tokens)= target labels
            continue
        out.append(line)
    prose = "\n".join(out)
    prose = re.sub(r"\{cite\}`[^`]*`", "", prose)
    prose = re.sub(r"\{\w+\}`([^`]*)`", r"\1", prose)   # other inline roles
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
    prose = re.sub(r"[`*_]", "", prose)
    return re.sub(r"\s+", " ", prose).strip()


def _split(text: str, limit: int = 220, min_words: int = 40) -> list:
    """Split an over-long passage at sentence boundaries into <= limit words."""
    if len(text.split()) <= limit:
        return [text]
    parts, cur = [], []
    for sent in re.split(r"(?<=[.!?]) ", text):
        if cur and len(" ".join(cur + [sent]).split()) > limit:
            parts.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        parts.append(" ".join(cur))
    if len(parts) > 1 and len(parts[-1].split()) < min_words:
        parts[-2:] = [" ".join(parts[-2:])]     # fold a stub tail back in
    return parts


def _outside_fences(src: str) -> str:
    """Drop fenced blocks so a ``# comment`` in demo code is not a heading."""
    keep, fence = [], False
    for line in src.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
        elif not fence:
            keep.append(line)
    return "\n".join(keep)


def _as_str(val) -> str:
    """A notebook source/output field, as one string whether list or str."""
    return "".join(val) if isinstance(val, list) else (val or "")


def _walk_cells(kind: str):
    """Yield ``(chapter, heading, cell)`` for every cell of one ``cell_type``.

    Cells come back in reading order, with the chapter title and nearest
    section heading tracked from the markdown cells along the way, so a code or
    output passage carries the same metadata a prose passage from the same spot
    would.
    """
    chapters = _find_chapters()
    for stem in SPINE:
        nb = json.loads((chapters / f"{stem}.ipynb").read_text())
        chapter, heading = None, ""
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                prose_only = _outside_fences(_as_str(cell["source"]))
                if chapter is None and (m := re.search(r"^# (.+)$", prose_only, re.M)):
                    chapter = m.group(1).split(":")[0].strip()
                if m := re.search(r"^## (.+)$", prose_only, re.M):
                    heading = m.group(1).strip()
            if cell["cell_type"] == kind:
                yield chapter or stem.capitalize(), heading, cell


def _cell_output_text(cell: dict) -> str:
    """Concatenate one code cell's textual outputs into a single passage.

    Stream text, the ``text/plain`` rendering of execute and display results,
    and error tracebacks are kept (terminal color codes stripped). Outputs that
    carry an image are skipped: their ``text/plain`` is just the figure or
    image object's repr (``<Figure size ...>``), and the figure itself already
    ships as its own captioned content.
    """
    parts = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            parts.append(_as_str(out.get("text")))
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if not any(k.startswith("image/") for k in data):
                parts.append(_as_str(data.get("text/plain")))
        elif otype == "error":
            parts.append("\n".join(out.get("traceback", [])))
    return _ANSI.sub("", "\n".join(p for p in parts if p)).strip()


def build_book_corpus(min_words: int = 40, write: bool = False) -> list:
    """Read every chapter notebook and return the book as retrieval passages.

    Each passage is one cleaned markdown cell (long cells are split at
    sentence boundaries), carrying the chapter and nearest section heading as
    metadata. Figure-attribution captions and cells shorter than ``min_words``
    are skipped. With ``write=True`` the result also replaces the committed
    snapshot.
    """
    corpus = []
    for chapter, heading, cell in _walk_cells("markdown"):
        src = _as_str(cell["source"])
        if src.lstrip().startswith("*Image generated"):
            continue
        prose = _clean_markdown(src)
        if len(prose.split()) < min_words:
            continue
        for part in _split(prose):
            corpus.append({"text": part, "chapter": chapter, "heading": heading})
    if write:
        SNAPSHOT.write_text(json.dumps(corpus, indent=1, ensure_ascii=False)
                            + "\n")
    return corpus


def build_code_corpus(write: bool = False) -> list:
    """Read every chapter notebook and return its code cells as passages.

    Each passage is one code cell's source, verbatim and uncleaned so it stays
    runnable, carrying the chapter and nearest section heading. Empty cells are
    skipped. With ``write=True`` the result replaces ``book_code_corpus.json``.
    """
    corpus = []
    for chapter, heading, cell in _walk_cells("code"):
        src = _as_str(cell["source"]).strip()
        if not src:
            continue
        corpus.append({"text": src, "chapter": chapter, "heading": heading})
    if write:
        CODE_SNAPSHOT.write_text(json.dumps(corpus, indent=1, ensure_ascii=False)
                                 + "\n")
    return corpus


def build_output_corpus(write: bool = False) -> list:
    """Read every chapter notebook and return its cell outputs as passages.

    Each passage is the concatenated text output of one code cell (stream text,
    text/plain results, error tracebacks); cells that print nothing or emit
    only figures are skipped. With ``write=True`` the result replaces
    ``book_output_corpus.json``.
    """
    corpus = []
    for chapter, heading, cell in _walk_cells("code"):
        text = _cell_output_text(cell)
        if not text:
            continue
        corpus.append({"text": text, "chapter": chapter, "heading": heading})
    if write:
        OUTPUT_SNAPSHOT.write_text(json.dumps(corpus, indent=1, ensure_ascii=False)
                                   + "\n")
    return corpus


def load_book_corpus() -> list:
    """Load the committed corpus snapshot the demos run against."""
    return json.loads(SNAPSHOT.read_text())


def load_code_corpus() -> list:
    """Load the committed code-corpus snapshot."""
    return json.loads(CODE_SNAPSHOT.read_text())


def load_output_corpus() -> list:
    """Load the committed output-corpus snapshot."""
    return json.loads(OUTPUT_SNAPSHOT.read_text())


def chapter_sample(corpus: list, chapter: str, n: int = 4) -> list:
    """The first passage from each of n distinct sections of one chapter."""
    seen = {}
    for p in corpus:
        if p["chapter"] == chapter and p["heading"] and p["heading"] not in seen:
            seen[p["heading"]] = p
            if len(seen) == n:
                break
    return list(seen.values())


def _embed_text(passage: dict) -> str:
    """The text actually embedded for a passage: its chapter and section
    heading prepended to the body.

    Headings are a strong signal of what a passage is about, so folding them
    into the vector lets a navigational query ("byte pair encoding") match the
    section that names it even when the body never spells the phrase out. The
    stored document stays the clean body, which is what the model later cites.
    """
    head = passage["heading"]
    prefix = f"{passage['chapter']}: {head}" if head else passage["chapter"]
    return f"{prefix}\n\n{passage['text']}"


class BookIndex:
    """The whole book in a persistent Chroma collection.

    Embeddings come from the same nomic model as every other retrieval demo
    in this book; Chroma stores them on disk, so the book is embedded once
    and every later session reopens the index instantly.
    """

    def __init__(self, path: str = None, name: str = "illustrating-genai"):
        import chromadb
        client = chromadb.PersistentClient(path=path or str(_PKG / "book_index"))
        self._col = client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"})

    def __len__(self) -> int:
        return self._col.count()

    def build(self, corpus: list = None, batch: int = 64) -> "BookIndex":
        """Embed and store every passage not already in the collection.

        Each passage is embedded with its section heading in front (see
        ``_embed_text``); the stored document stays the clean body text.
        """
        corpus = corpus or load_book_corpus()
        ids = [f"p{i:04d}" for i in range(len(corpus))]
        have = set(self._col.get(ids=ids)["ids"])
        todo = [(i, p) for i, p in zip(ids, corpus) if i not in have]
        for n in range(0, len(todo), batch):
            part = todo[n: n + batch]
            self._col.add(
                ids=[i for i, _ in part],
                embeddings=[embed(_embed_text(p)) for _, p in part],
                documents=[p["text"] for _, p in part],
                metadatas=[{"chapter": p["chapter"], "heading": p["heading"]}
                           for _, p in part])
        return self

    def search(self, query: str, k: int = 3, chapter: str = None) -> list:
        """Top-k passages, optionally filtered to one chapter's metadata."""
        res = self._col.query(
            query_embeddings=[embed(query)], n_results=k,
            where={"chapter": chapter} if chapter else None)
        return [{"text": d, **m, "score": round(1 - dist, 3)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                      res["distances"][0])]

    def ask(self, question: str, k: int = 4) -> str:
        """Full RAG over the book: retrieve passages, generate a cited answer."""
        return rag(question, [r["text"] for r in self.search(question, k)])


# ── How much a vector database actually buys you ─────────────────────────────
# The DocumentStore compares a query against every stored vector in a Python
# loop; Chroma reaches the same neighbours through an HNSW index. The difference is
# invisible at the book's own 421 passages and decisive at scale, so we time the
# search alone (the query is embedded once up front) as the corpus grows.

def benchmark_index(sizes=(500, 5_000, 50_000, 100_000), dim: int = 768,
                    k: int = 3, repeats: int = 7, seed: int = 0) -> list:
    """Time one nearest-neighbour search, naive linear scan against Chroma's
    HNSW index, over random corpora of growing size.

    Random vectors stand in for real passages so the corpus can grow past the
    book's own 421, and the query is embedded once, so what we clock is the
    search structure itself, not the embedding call both approaches share.
    Returns one row per size with both medians in milliseconds and their ratio.
    Behind INDEX_SPEED_STUDY / plot_index_speed; rerun by _index_speed_probe.py.
    """
    import time
    import numpy as np
    import chromadb
    from genai.embed import similarity
    rng = np.random.default_rng(seed)

    def median_ms(fn) -> float:
        ts = []
        for _ in range(repeats):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        return sorted(ts)[len(ts) // 2]

    rows = []
    for n in sizes:
        vecs = rng.standard_normal((n, dim))
        q = rng.standard_normal(dim)
        scan = lambda: sorted(range(n), key=lambda i: similarity(q, vecs[i]),
                              reverse=True)[:k]
        col = chromadb.EphemeralClient().create_collection(
            f"bench-{n}", metadata={"hnsw:space": "cosine"})
        ids, emb = [f"p{i}" for i in range(n)], vecs.tolist()
        for s in range(0, n, 5_000):           # Chroma caps one add() at ~5k rows
            col.add(ids=ids[s:s + 5_000], embeddings=emb[s:s + 5_000])
        hnsw = lambda: col.query(query_embeddings=[q.tolist()], n_results=k)
        rows.append({"n": n,
                     "naive_ms": round(median_ms(scan), 2),
                     "chroma_ms": round(median_ms(hnsw), 2)})
    for r in rows:
        r["speedup"] = round(r["naive_ms"] / r["chroma_ms"], 1)
    return rows


# Measured by scripts/_index_speed_probe.py (Apple Silicon, 768-dim random
# vectors, median of 7 searches). The naive scan grows with the corpus while
# Chroma's HNSW index barely moves: a wash at the book's scale, ~119x by 100k
# passages. Wall-clock numbers drift run to run; the shape is what reproduces.
INDEX_SPEED_STUDY = [
    {"n": 500,    "naive_ms": 1.23,   "chroma_ms": 0.61, "speedup": 2.0},
    {"n": 5000,   "naive_ms": 14.43,  "chroma_ms": 1.66, "speedup": 8.7},
    {"n": 50000,  "naive_ms": 146.5,  "chroma_ms": 2.25, "speedup": 65.1},
    {"n": 100000, "naive_ms": 282.46, "chroma_ms": 2.38, "speedup": 118.7},
]


# Three snippets of the book's raw markdown source, exactly the kind of markup
# the ingestion pipeline has to scrub before anything is embedded.
RAW_BOOK_SAMPLES = [
    "```{admonition} Abstract\n\nThis chapter is about giving a language "
    "model access to knowledge it was never trained on. \n```",
    "the same closeness measure we took apart in "
    "[the Semantics chapter](#chap:semantics), where **cosine similarity** "
    "compares two `embed()` vectors",
    "*The limits of my language mean the limits of my world.*\n\n"
    "— Ludwig Wittgenstein {cite}`wittgenstein1922tractatus`",
]


SOPHIA_SYSTEM = (
    "You are Sophia, the reader's companion for the book Illustrating "
    "Generative AI by Chiké Abuah. Answer from the numbered book passages "
    "only, and cite them like [1]. Each passage is tagged with the section "
    "and chapter it comes from; when the reader asks where a topic lives, "
    "point them to that section by name. Say plainly when the book does not "
    "cover something.")


class Sophia:
    """The book's companion assistant: this chapter's layers in one object.

    Each question is screened for injection phrasing, rewritten into a
    standalone query when conversation history makes it elliptical, answered
    from passages retrieved out of the persistent Chroma index (optionally
    pinned to one chapter), and grounded with numbered citations.
    """

    def __init__(self, index: BookIndex = None):
        self.index = index or BookIndex()
        self.history = []
        self.sources = []          # passages retrieved for the last answer

    def ask(self, message: str, k: int = 4, chapter: str = None) -> str:
        from genai.llm import ask as _ask
        from genai.rag import _contextualize, INJECTION_SIGNALS
        if any(s in message.lower() for s in INJECTION_SIGNALS):
            return ("That looks like an attempt to override my instructions, "
                    "so I will skip it.")
        standalone = _contextualize(message, self.history)
        hits = self.index.search(standalone, k=k, chapter=chapter)
        self.sources = hits        # so callers can resolve the [n] citations
        context = "\n\n".join(
            f'[{i+1}] (from "{h["heading"]}" in the {h["chapter"]} chapter) '
            f'{h["text"]}' for i, h in enumerate(hits))
        answer = _ask(f"{context}\n\nQuestion: {standalone}",
                      system=SOPHIA_SYSTEM, max_tokens=150)
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
        return answer


# --- display helpers -------------------------------------------------------

def show_index_speed(rows) -> None:
    """Print the search-latency ladder: corpus size, naive scan, Chroma, speedup."""
    print(f"{'passages':>9}  {'naive scan':>11}  {'Chroma':>9}  speedup")
    for r in rows:
        print(f"{r['n']:>9,}  {r['naive_ms']:>8.1f} ms  {r['chroma_ms']:>6.2f} ms  "
              f"{r['speedup']:>5.0f}x")


def show_window_budget(counts: dict, window: int, default: int) -> None:
    """Show how each model's token count fills its context window and budget."""
    for name, n in counts.items():
        print(f"{name:24} {n:>8,} tokens   {n/window:>4.0%} of the window   "
              f"{n/default:>3.0f}x the default budget")


def show_conversation(agent, questions, limit: int = 170) -> None:
    """Ask each question in turn, print a length-capped answer, then list the book
    sections its ``[n]`` citations resolve to so none is left dangling. Citations
    are read from the shown text, so the reference list matches exactly what the
    reader sees."""
    for q in questions:
        shown = agent.ask(q)[:limit]
        print(f"Q: {q}\nA: {shown}")
        _show_citations(shown, agent.sources)
        print()


def cited_sources(answer: str, sources: list) -> list:
    """The (number, passage) pairs an answer actually cites, in order.

    Reads the ``[n]`` markers out of the answer (``[1]`` and ``[1, 2]`` both
    count) and pairs each with its retrieved passage, skipping any number the
    model invented past the passages it was given.
    """
    used = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer)
            for n in group.split(",") if n.strip().isdigit()}
    return [(n, sources[n - 1]) for n in sorted(used) if 1 <= n <= len(sources)]


def _show_citations(answer: str, sources: list) -> None:
    """Print the book sections an answer's ``[n]`` markers resolve to, grouped by
    section (several retrieved chunks can share one), so the citations never dangle."""
    groups = {}            # section -> citation numbers
    for n, src in cited_sources(answer, sources):
        where = f"{src['chapter']} / {src['heading']}" if src["heading"] \
            else src["chapter"]
        groups.setdefault(where, []).append(n)
    for where, nums in groups.items():
        print(f"   [{', '.join(map(str, nums))}] {where}")


def show_faq(questions, index: BookIndex = None, k: int = 4) -> None:
    """Answer each reader question on its own, from a fresh retrieval.

    Unlike show_conversation, the history is cleared between questions: an FAQ
    is a list of independent questions, not one thread, so an earlier answer
    must never colour a later one. Each answer is asked for in a sentence or
    two, and the passages its ``[n]`` markers cite are listed beneath it so the
    citations resolve to a real section of the book.
    """
    sophia = Sophia(index)
    for q in questions:
        sophia.history = []
        answer = sophia.ask(f"{q} Answer in one or two sentences.", k=k)
        print(f"Q: {q}\nA: {answer}")
        _show_citations(answer, sophia.sources)
        print()

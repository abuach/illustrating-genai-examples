"""Research: turning an open question into a report you can trust.

Ask a model to research something and cite its sources and it will hand you a
fluent report with a tidy numbered source list. Check the list and the seams
show: the citations are invented, because the model never looked anything up. It
answered from memory and dressed the answer as scholarship. That is the failure
this chapter is about, and it is the same shape as every other chapter's: fluent
output that *looks* right, with no checker underneath. Here the checker we own is
provenance. **No claim ships unless a real retrieved passage actually supports
it.** Fewer claims survive, but every survivor can be traced to a source.

The chapter walks a small deep-research agent, the machinery behind the
"research" buttons in modern assistants. It **decomposes** the question into
sub-questions (``decompose``), **searches** the live web for each
(``search`` over the book's ddgs helper), **gathers** real passages, **drafts**
claims from them, and **gates** every claim against the passage it came from
(``supported``, a factored yes/no entailment check in the spirit of
chain-of-verification). What survives is synthesized into a cited report
(``deep_research``). Alongside it runs the ungrounded baseline
(``naive_research``) so the two can be measured against the same real evidence:
naive ships six claims that only two sources back; grounded ships fewer and every
one holds. A final loop shows *when to stop* — the new supported claims per round
decay to nothing (``saturation_curve``), which is the signal to quit.

The model is gemma4:latest (planner, drafter, verifier, writer) and
nomic-embed-text (passage and claim similarity). The web search is live at bake
time only: everything nondeterministic is generated once by
``scripts/_research_probe.py`` into ``_research_capture.json`` and replayed from
there, so the notebook runs offline and identical every time. Every speaker row
below is a model's real output, never a paraphrase.
"""
import json
import re
from pathlib import Path

from genai.agent import show_turn
from genai.rag import show_grounded

_PKG = Path(__file__).resolve().parent
_CAP_PATH = _PKG / "_research_capture.json"
# Baked by scripts/_research_probe.py. Absent only while the probe first runs
# (it imports the authored data below, never the bake), so tolerate a miss.
_CAP = json.loads(_CAP_PATH.read_text()) if _CAP_PATH.exists() else {}

MODEL = "gemma4:latest"

# The question that threads through the chapter. The pipeline is topic-agnostic:
# swap this one constant and re-run scripts/_research_probe.py to research
# anything else.
RESEARCH_QUESTION = ("What is the James Webb Space Telescope, "
                     "and what has it discovered?")


# ── Live steps (used by scripts/_research_probe.py to build the bake) ──────────
# None of these run at notebook time; the show_* and study functions below read
# the bake instead. They are kept here so the probe imports one module and the
# reader can see exactly what each stage does.

def _ask(prompt, model=MODEL, temp=0.0, n=220):
    import ollama
    return (ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                        think=False,
                        options={"temperature": temp, "seed": 0, "num_predict": n}
                        )["message"]["content"] or "").strip()


def _lines(reply):
    """Split a model's list reply into clean items, dropping bullets and numbers."""
    out = []
    for raw in reply.splitlines():
        item = re.sub(r"^[\-\*\d\.\)\s]+", "", raw).strip()
        if item:
            out.append(item)
    return out


def _sentences(text):
    """Split a short report into atomic claim sentences worth checking."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _is_meta(text):
    """Drop drafted sentences that describe the evidence instead of the subject
    (``the source does not mention...``), which would verify vacuously."""
    t = text.lower()
    return any(p in t for p in ("the source", "the passage", "the provided",
                                "the article", "the text", "this excerpt"))


def decompose(question, model=MODEL):
    """Break an open question into a handful of searchable sub-questions. This is
    the planning move: one search rarely covers everything the question asks."""
    reply = _ask("Break this research question into 4 focused sub-questions that "
                 "could each be answered by a web search. One per line, no "
                 f"numbering.\n\n{question}", model, n=180)
    subs = [s for s in _lines(reply) if len(s) > 12]
    return subs[:4]


def search(query, k=4):
    """One real web search. Thin wrapper over the book's ddgs helper so the whole
    chapter reaches the open web through a single retrieval surface."""
    from genai.crag import web_search
    return web_search(query, k=k)


def supported(claim, passage, model=MODEL):
    """The checker. Does this retrieved passage actually support this claim?
    Returns ``(ok, reason)``: a yes/no verdict and one sentence, both real output.
    Factored like chain-of-verification, the drafted report is out of view; only
    the lone claim and the evidence are on the table."""
    reply = _ask(f"PASSAGE:\n{passage}\n\nCLAIM: {claim}\n\nDoes the passage "
                 "support the claim? Answer 'yes' or 'no', then one sentence why.",
                 model, temp=0.0, n=60)
    m = re.search(r"\b(yes|no)\b", reply.lower())
    ok = m is not None and m.group(1) == "yes"
    reason = re.sub(r"^\W*(yes|no)\b[\.,:\s]*", "", reply, flags=re.I).strip()
    return ok, reason


def _parse_sources(block):
    """Pull ``{title, url}`` out of a model's SOURCES list, whether it wrote a real
    link or a placeholder. The placeholder is the tell: the model is citing from
    memory and knows it has no address to give."""
    cites = []
    for line in block.splitlines():
        m = re.match(r"\[?(\d+)[\].]?\s+(.+)", line.strip())
        if not m:
            continue
        rest = m.group(2)
        link = re.search(r"https?://\S+", rest)
        if link:
            url, title = link.group(0), rest[:link.start()]
        else:
            ph = re.search(r"\(([^)]*(?:url|link|http)[^)]*)\)", rest, re.I)
            url = ph.group(1).strip() if ph else "(no URL given)"
            title = re.sub(r"\([^)]*(?:url|link|http)[^)]*\)", "", rest, flags=re.I)
        cites.append({"title": re.sub(r"[*_`]", "", title).strip(" .—-"), "url": url})
    return cites


def naive_research(question, model=MODEL):
    """Research with no tools: write a cited report from memory. Nothing was looked
    up, so the sources are decoration, often a bare placeholder where a URL should
    be. Returns ``(report, claims, citations)`` — every field real model output."""
    full = _ask("Research this question and write a short report of about four "
                "sentences. Add a numbered citation marker like [1] after each "
                "fact, and end with a SOURCES list giving the title and URL for "
                f"each number.\n\n{question}", model, n=320)
    parts = re.split(r"\n[\s*]*SOURCES?[\s*:]*\n", full, maxsplit=1, flags=re.I)
    report = re.sub(r"\n[*\s]*\n", "\n\n", parts[0]).strip().rstrip("*").strip()
    citations = _parse_sources(parts[1]) if len(parts) > 1 else []
    claims = [re.sub(r"\s*\[\d+\]", "", s).strip()
              for s in _sentences(report)][:6]
    return report, claims, citations[:6]


def _novel(claim, kept, model=MODEL, thresh=0.82):
    """Is this supported claim genuinely new, or a near-duplicate of one already
    kept? Dedup by embedding similarity so the saturation count stays honest."""
    from genai.embed import similarity
    return all(similarity(claim, k) < thresh for k in kept)


def gather_round(subq, k=3, model=MODEL):
    """One research round. Search a sub-question, keep the passages that came
    back, draft a short answer from each, and gate every drafted sentence against
    the passage it was drawn from. Returns the evidence and the graded claims."""
    hits = [h for h in search(subq, k=k) if h.get("snippet")]
    evidence = [{"subq": subq, "passage": h["snippet"], "title": h["title"],
                 "url": h["url"]} for h in hits]
    claims = []
    for ev in evidence[:2]:
        draft = _ask("Answer the question in two or three short factual sentences "
                     "about the subject, using the source and what you already "
                     "know. State facts directly; do not mention 'the source' or "
                     f"'the passage'.\n\nQUESTION: {subq}\nSOURCE: {ev['passage']}",
                     model, n=150)
        for text in _sentences(draft):
            if _is_meta(text):
                continue
            ok, reason = supported(text, ev["passage"], model)
            claims.append({"text": text, "supported": ok, "reason": reason,
                           "title": ev["title"], "url": ev["url"], "subq": subq})
    return evidence, claims


def deep_research(question, k=3, model=MODEL):
    """The grounded loop. Decompose the question, then round by round search a
    sub-question, gather real passages, draft from them, and keep only the claims
    a passage actually supports and that were not already collected. Returns the
    whole run: sub-questions, the evidence pool, every drafted claim with its
    verdict, the kept claims with citation indices, the per-round novelty, the
    distinct sources, and a report synthesized over the kept claims."""
    subqs = decompose(question, model)
    evidence, drafted, kept, rounds = [], [], [], []
    for subq in subqs:
        evs, claims = gather_round(subq, k, model)
        evidence.extend(evs)
        new = 0
        for c in claims:
            drafted.append(c)
            if c["supported"] and _novel(c["text"], [x["text"] for x in kept], model):
                kept.append(c)
                new += 1
        rounds.append({"subq": subq, "new_supported": new, "cumulative": len(kept)})
    # Citations: the distinct sources behind kept claims, numbered in first use.
    citations, url_to_n = [], {}
    for c in kept:
        if c["url"] not in url_to_n:
            url_to_n[c["url"]] = len(citations) + 1
            citations.append({"title": c["title"], "url": c["url"]})
        c["n"] = url_to_n[c["url"]]
    facts = "\n".join(f"[{c['n']}] {c['text']}" for c in kept)
    report = _ask("Write a short report of about four sentences answering the "
                  "question in plain prose (no markdown, no LaTeX math). Use ONLY "
                  "these facts, and keep each [n] citation marker where its fact is "
                  f"used.\n\nQUESTION: {question}\n\n{facts}", model, n=240)
    return {"question": question, "subqs": subqs, "evidence": evidence,
            "drafted": drafted, "kept": kept, "rounds": rounds,
            "citations": citations, "report": report}


# ── Deterministic studies over the bake (feed the two figures) ────────────────

def grounding_study():
    """Naive versus grounded on the same real evidence: how many claims each
    ships, and how many a real source actually backs. Naive ships everything and
    only a couple hold; grounded ships only what it could verify."""
    naive = _CAP["naive"]
    kept = _CAP["kept"]
    return {
        "naive": {"shipped": len(naive["claims"]),
                  "supported": sum(c["supported"] for c in naive["checks"])},
        "grounded": {"shipped": len(kept),
                     "supported": sum(c["supported"] for c in kept),
                     "dropped": sum(not c["supported"] for c in _CAP["drafted"])},
    }


def round_coverage():
    """New supported claims contributed by each research round, and the running
    total. With well-separated sub-questions the per-round count does not trail
    off: each round reaches different sources and adds real ground, which is the
    quantitative case for decomposing the question in the first place."""
    return {"rounds": _CAP["rounds"]}


# ── Replayed transcripts (read the bake; every speaker row is real output) ────

def _sources_block(citations, width=58):
    print("\nSOURCES")
    for i, c in enumerate(citations, 1):
        title = " ".join(c["title"].split())
        print(f"  [{i}] {title[:width]}{'…' if len(title) > width else ''}")
        print(f"      {c['url'][:width]}")


def show_naive_report(cap=None):
    """The cold open: a confident report whose citations point nowhere real."""
    c = cap or _CAP["naive"]
    show_turn("you", f"Research and cite your sources: {_CAP['question']}")
    show_turn("gemma4", c["report"])
    _sources_block(c["citations"])
    ok = sum(x["supported"] for x in c["checks"])
    print()
    show_turn("CHECK", f"do real sources back these claims? "
                       f"{ok} of {len(c['checks'])} supported")
    for x in c["checks"]:
        if not x["supported"]:
            show_turn("UNBACKED", x["text"])
            break


def show_search(cap=None):
    """The first real move: retrieve instead of remember."""
    s = cap or _CAP["search"]
    show_turn("you", f"search: {s['query']}")
    show_turn("RETRIEVAL", f"{len(s['hits'])} results from the open web")
    for i, h in enumerate(s["hits"], 1):
        show_turn("SOURCE", f"[{i}] {h['title']}")
        # Emit the URL unwrapped (width huge) so neither show_turn nor the
        # reflow pass breaks a long link across lines and welds a space into it.
        show_turn("", h["url"], width=10**6)


def show_plan(cap=None):
    """Decomposition: the sub-questions the agent will chase."""
    subqs = cap or _CAP["subqs"]
    show_turn("you", _CAP["question"])
    show_turn("gemma4", "I'll answer this in parts:")
    for i, q in enumerate(subqs, 1):
        show_turn("PLAN", f"{i}. {q}")


def show_gather(cap=None):
    """The evidence pool filling up, round by round."""
    ev = cap or _CAP["evidence"]
    show_turn("PLAN", f"{len(_CAP['subqs'])} sub-questions to answer")
    for r in _CAP["rounds"]:
        n = sum(e["subq"] == r["subq"] for e in ev)
        show_turn("RETRIEVAL", f"{r['subq'][:44]} -> {n} passages")
    sources = {e["url"] for e in ev}
    show_turn("POOL", f"{len(ev)} passages from {len(sources)} distinct sources")


def show_verification(cap=None, keeps=2, drops=2):
    """The gate at work: a few claims kept because a passage backs them, a few
    dropped because it does not. Shows both verdicts, in the order drafted."""
    drafted = cap or _CAP["drafted"]
    kept = [c for c in drafted if c["supported"]][:keeps]
    cut = [c for c in drafted if not c["supported"]][:drops]
    for c in sorted(kept + cut, key=drafted.index):
        show_turn("CLAIM", c["text"])
        reason = c["reason"] if len(c["reason"]) <= 150 else c["reason"][:149] + "…"
        verdict = "yes" if c["supported"] else f"no — {reason}"
        show_turn("CHECK", f"passage supports it? {verdict}")
        show_turn("KEEP" if c["supported"] else "DROP", c["text"][:52])


def show_report(cap=None):
    """The grounded report: only verified claims, each citing a real source. The
    sources shown are exactly the ones the report cites, renumbered in order of
    first use, so no bracket points at a source that is not on the page."""
    r = cap or _CAP
    cites, order, remap = r["citations"], [], {}

    def _renumber(m):
        n = int(m.group(1))
        if not 1 <= n <= len(cites):
            return ""
        if n not in remap:
            order.append(n)
            remap[n] = len(order)
        return f"[{remap[n]}]"

    text = re.sub(r"\[(\d+)\]", _renumber, r["report"])
    sources = [f"{cites[n - 1]['title']} — {cites[n - 1]['url']}" for n in order]
    show_grounded(text, sources)


# ── When the sources disagree (the support gate's blind spot) ──────────────────
# The gate reads one claim against one passage, so it never learns that a second
# credible passage gives a different figure for the same fact. This section's
# demo plants a real conflicting page in the chapter's evidence pool (never
# touching the passages already there), reruns the gather-and-gate loop on a
# question whose answer hinges on the disputed figure, and then adds the guard:
# a contradiction check that asks every relevant source what value it gives and
# makes the report show a split instead of asserting one side. Everything
# nondeterministic is baked by scripts/_contradiction_probe.py into
# ``_contradiction_capture.json`` and replayed from there.

_CONFLICT_PATH = _PKG / "_contradiction_capture.json"
_CONFLICT = json.loads(_CONFLICT_PATH.read_text()) if _CONFLICT_PATH.exists() else {}

# The narrow question whose answer hinges on one disputed figure. Phrased with
# the same vocabulary the disagreeing pages use, so both rank at the top of the
# pool for it.
CONFLICT_QUESTION = ("How many times greater is JWST's light-gathering "
                     "capability than Hubble's?")

# A passage "speaks to" a claim when their embedding similarity clears this
# threshold — the same on-topic band the Corrective RAG evaluator uses.
CONFLICT_T = 0.70

# The page we add to the pool: NASA's own Webb FAQ, exactly as the chapter's
# ddgs search returns it for CONFLICT_QUESTION (title, url, snippet verbatim;
# captured 2026-07-08 and pinned so the demo doesn't depend on the day's
# result shuffle). It says "about 6 times larger in area" where the pool's
# mission-paper passage says "7X the light-gathering capability".
CONFLICT_DOC = {
    "subq": CONFLICT_QUESTION,
    "title": "Webb FAQs - NASA Science",
    "url": "https://science.nasa.gov/mission/webb/faqs-full/",
    "passage": ("In order to do this, Webb has a much larger primary mirror "
                "than Hubble (2.7 times larger in diameter, or about 6 times "
                "larger in area), giving it more light-gathering power."),
}


def _ask_at(prompt, seed, model=MODEL, temp=0.7, n=150):
    """``_ask`` with the sampler on: temperature 0.7 and a per-run seed, so
    repeated runs are genuinely different tellings, not one telling replayed.
    The gate itself stays at temperature 0, exactly as in the chapter."""
    import ollama
    return (ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                        think=False,
                        options={"temperature": temp, "seed": seed, "num_predict": n}
                        )["message"]["content"] or "").strip()


def rank_passages(question, pool, k=3):
    """The search step, aimed at a fixed pool instead of the live web: the ``k``
    passages most similar to the question by the chapter's embeddings. Using the
    pool keeps every rerun looking at the same pages."""
    from genai.embed import embed, similarity
    qv = embed(question)
    return sorted(pool, key=lambda e: similarity(qv, embed(e["passage"])),
                  reverse=True)[:k]


def gather_from_pool(question, pool, seed, model=MODEL):
    """One gather round, the chapter's own recipe: draft a short answer from each
    of the top passages, gate every drafted sentence against the passage it came
    from, and drop near-duplicates. Returns ``(drafted, kept)`` with real verdicts
    on every claim."""
    drafted, kept = [], []
    for ev in rank_passages(question, pool)[:2]:
        draft = _ask_at("Answer the question in two or three short factual "
                        "sentences about the subject, using the source and what "
                        "you already know. State facts directly; do not mention "
                        "'the source' or 'the passage'.\n\n"
                        f"QUESTION: {question}\nSOURCE: {ev['passage']}", seed)
        for text in _sentences(draft):
            if _is_meta(text):
                continue
            ok, reason = supported(text, ev["passage"], model)
            claim = {"text": text, "supported": ok, "reason": reason,
                     "title": ev["title"], "url": ev["url"]}
            drafted.append(claim)
            if ok and _novel(text, [k_["text"] for k_ in kept], model):
                kept.append(claim)
    return drafted, kept


def stated_value(claim, passage, model=MODEL):
    """One judge call: what figure does this passage give for the quantity the
    claim asserts? Returns the model's reply verbatim — a number, a ratio, or
    'none' when the passage is silent."""
    return _ask(f"PASSAGE:\n{passage}\n\nCLAIM: {claim}\n\nThe claim states a "
                "numeric figure. What figure does the PASSAGE give for that same "
                "quantity? Reply with only the number or ratio exactly as the "
                "passage states it, or 'none' if it gives none.", model, n=30)


_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _factor(reply):
    """Normalize a ``stated_value`` reply to a comparable number, or ``None``.
    Word numbers count ('seven times' -> 7); a leading 'none' means silence."""
    r = (reply or "").strip().lower()
    if r.startswith("none"):
        return None
    for word, num in _WORD_NUMS.items():
        if re.search(rf"\b{word}\b", r):
            return float(num)
    m = re.search(r"\d+(?:\.\d+)?", r)
    return float(m.group()) if m else None


def value_scan(claim, pool, model=MODEL, thresh=CONFLICT_T):
    """The contradiction check. Every pool passage above the relevance threshold
    is asked what figure it gives for the claim's quantity; silence is fine,
    a different figure is not. Returns the real per-source replies."""
    from genai.embed import embed, similarity
    cv = embed(claim)
    rows = []
    for ev in pool:
        if similarity(cv, embed(ev["passage"])) < thresh:
            continue
        reply = stated_value(claim, ev["passage"], model)
        rows.append({"title": ev["title"], "url": ev["url"], "reply": reply,
                     "value": _factor(reply)})
    return rows


def distinct_values(scans):
    """Fold one or more value scans into the distinct figures found, each with
    one representative source row (first seen). Two or more distinct figures
    mean the sources disagree."""
    vals = {}
    for scan in scans:
        for row in scan:
            if row["value"] is not None and row["value"] not in vals:
                vals[row["value"]] = row
    return vals


def write_answer(question, kept, seed, dispute=None, model=MODEL):
    """The chapter's synthesis step at mini scale: one or two sentences over the
    kept claims, each [n] marker tied to its source. When a ``dispute`` block is
    passed (from ``distinct_values``), the writer must present every disputed
    figure with its citation instead of picking one. Returns ``(report,
    citations)``."""
    citations, url_to_n = [], {}

    def _cite(row):
        if row["url"] not in url_to_n:
            url_to_n[row["url"]] = len(citations) + 1
            citations.append({"title": row["title"], "url": row["url"]})
        return url_to_n[row["url"]]

    facts = "\n".join(f"[{_cite(c)}] {c['text']}" for c in kept)
    note = ""
    if dispute and len(dispute) > 1:
        lines = "; ".join(f"[{_cite(row)}] {row['title']} says {row['reply']}"
                          for row in dispute.values())
        note = ("\n\nDISPUTED: the sources give different figures for the same "
                f"quantity: {lines}. Present BOTH values with their citations "
                "and say plainly that the sources differ; do not pick one.")
    report = _ask_at("Write one or two sentences answering the question in plain "
                     "prose (no markdown, no LaTeX math). Use ONLY these facts, "
                     "and keep each [n] citation marker where its fact is used."
                     f"{note}\n\nQUESTION: {question}\n\n{facts}", seed)
    return report, citations


# ── Deterministic classification and studies over the conflict bake ───────────

# Which figure does a report state? Guarded against the decoys in this corpus:
# '2.7 times larger in diameter' must not read as 7, '6.5-meter' must not read
# as 6, and 'a factor of six' must still read as 6.
_V7 = re.compile(r"seven[-\s]*(?:times|fold)|factor of (?:seven|7)|"
                 r"(?<![\d.])7\s*(?:x\b|×|[-\s]*times|[-\s]*fold)", re.I)
_V6 = re.compile(r"six[-\s]*(?:times|fold)|factor of (?:six|6(?!\.\d))|"
                 r"(?<![\d.])6(?!\.\d)\s*(?:x\b|×|[-\s]*times|[-\s]*fold)", re.I)
# Does the report say the SOURCES give rival answers? Anchored on the sources
# ("sources differ", "sources provide conflicting figures") or the contrastive
# "[1], while another source ..." construction, so that merely mentioning a
# second figure ("Additionally...", "Another source notes...") or fusing the
# two ("this difference means...") does not count. Audited by hand against
# every baked report.
_FLAG = re.compile(r"sources?\s+(?:\w+\s+){0,2}(?:differ|disagree|different|"
                   r"differing|conflicting|discrepan)|"
                   r"while\s+(?:one|another|other)\b|whereas", re.I)


def _classify(report):
    """One report -> which figures it states and whether it flags the split."""
    return {"v7": bool(_V7.search(report)), "v6": bool(_V6.search(report)),
            "flag": bool(_FLAG.search(report))}


_BUCKETS = ("7x only", "~6x only", "both, no flag", "names the split", "neither")


def _bucket(report):
    """One report -> exactly one bucket, from the classification above."""
    c = _classify(report)
    if c["v7"] and c["v6"]:
        return "names the split" if c["flag"] else "both, no flag"
    return "7x only" if c["v7"] else ("~6x only" if c["v6"] else "neither")


def conflict_study():
    """Count, across the baked runs, what the naive pipeline did with the
    disagreement versus what the guarded one did: which figure each report
    asserts, and whether it tells the reader the sources disagree."""
    out = {}
    for arm in ("naive", "guarded"):
        reports = [r[arm]["report"] for r in _CONFLICT["runs"]]
        out[arm] = {b: sum(_bucket(rep) == b for rep in reports)
                    for b in _BUCKETS}
    out["m"] = len(_CONFLICT["runs"])
    return out


def show_conflict_study(study=None):
    """What the reports asserted, bucket by bucket, one column per policy."""
    s = study or conflict_study()
    print(f"{'each report asserts':22}{'naive':>8}{'guarded':>10}")
    for b in _BUCKETS:
        if b == "neither" and not (s["naive"][b] or s["guarded"][b]):
            continue
        print(f"{b:22}{s['naive'][b]:>8}{s['guarded'][b]:>10}")
    print(f"{'(runs per policy)':22}{s['m']:>8}{s['m']:>10}")


# ── Replayed transcripts for the conflict demo (all rows real output) ─────────

def show_conflict_setup(cap=None):
    """The plant: one real page added to the chapter's evidence pool, stating a
    different figure than a page already there."""
    c = cap or _CONFLICT
    plant, incumbent = c["plant"], c["incumbent"]
    show_turn("IN POOL", incumbent["title"])
    show_turn("SAYS", incumbent["quote"])
    show_turn("ADDED", plant["title"])
    show_turn("", plant["url"], width=10**6)
    show_turn("SAYS", plant["quote"])
    show_turn("POOL", f"{c['pool_size']} passages from "
                      f"{c['pool_sources']} distinct sources")


def _numeric_claims(run):
    """The drafted claims that state one of the disputed figures."""
    return [c for c in run["drafted"]
            if _V7.search(c["text"]) or _V6.search(c["text"])]


def _demo_run(c):
    """The run both transcript cells show, so the guard replays the same seed:
    the first whose naive report picks a single figure while its guarded one
    states both and names the disagreement."""
    for j, r in enumerate(c["runs"]):
        naive = _classify(r["naive"]["report"])
        if naive["v7"] != naive["v6"] and all(_classify(r["guarded"]["report"]).values()):
            return j
    return 0


def show_conflict_run(i=None, cap=None):
    """One real gather-and-gate run on the planted pool: both figures drafted,
    both pass the gate, and the report asserts an answer."""
    c = cap or _CONFLICT
    run = c["runs"][_demo_run(c) if i is None else i]
    show_turn("you", c["question"])
    for claim in _numeric_claims(run)[:2]:
        show_turn("CLAIM", claim["text"])
        show_turn("CHECK", "passage supports it? "
                           f"{'yes' if claim['supported'] else 'no'}")
    _grounded_block(run["naive"])


def show_conflict_guard(i=None, cap=None):
    """The same run (same seed) with the contradiction check in the gather step:
    the scan's real per-source replies, the guard's verdict, and a report that
    shows the split instead of asserting one side."""
    c = cap or _CONFLICT
    run = c["runs"][_demo_run(c) if i is None else i]
    claim = next(iter(_numeric_claims(run)), run["drafted"][0])
    show_turn("CLAIM", claim["text"])
    # One real scan reply per distinct figure; where several sources gave the
    # same figure, show the one the report itself cites.
    cited = {x["url"] for x in run["kept"]}
    rows = [r for s in run["scans"] for r in s["rows"] if r["value"] is not None]
    for v in sorted({r["value"] for r in rows}, reverse=True):
        row = next((r for r in rows if r["value"] == v and r["url"] in cited),
                   next(r for r in rows if r["value"] == v))
        show_turn("VALUES", f"{row['title'][:44]} says {row['reply']}")
    show_turn("GUARD", "sources disagree: report both figures, cite both")
    _grounded_block(run["guarded"])


def _grounded_block(arm):
    """Print one arm's report through the chapter's SOURCES-block convention:
    the sources shown are exactly the ones the report cites, renumbered in
    order of first use (same rule as ``show_report``)."""
    cites, order, remap = arm["citations"], [], {}

    def _renumber(m):
        n = int(m.group(1))
        if not 1 <= n <= len(cites):
            return ""
        if n not in remap:
            order.append(n)
            remap[n] = len(order)
        return f"[{remap[n]}]"

    text = re.sub(r"\[(\d+)\]", _renumber, arm["report"])
    show_grounded(text, [f"{cites[n - 1]['title']} — {cites[n - 1]['url']}"
                         for n in order])


# ── Two sources that are the same source ──────────────────────────────────────
# The per-claim gate counts citations, not independent sources. A claim backed by
# three articles looks solid, but if the three are one wire story re-hosted, it's
# backed once. Counting them as three is how a single press release launders itself
# into a grounded report. A dedup pass over the retrieved passages, cluster them by
# embedding similarity before counting, tells corroboration from an echo: near-
# identical text collapses to one cluster, genuinely different reporting stays
# apart. nomic-embed is deterministic, so the cell runs live.

SIM_THRESHOLD = 0.90     # near-identical re-hosts cluster; different reporting doesn't
INDEPENDENCE_BAR = 2     # a claim needs two independent sources to count as sourced

# Two claims. The first is cited three times, but the three are one wire story
# re-hosted, so it rests on a single source. The second is cited twice by genuinely
# independent reporting. The bare per-claim gate counts citations and passes both.
CLAIM_A = "Atlas-7 contains 4 billion images."
CLAIM_A_PASSAGES = [
    ("Reuters", "GaiaCorp on Tuesday released Atlas-7, a dataset of 4 billion "
                "images, the company said in a statement."),
    ("TechDaily", "GaiaCorp released Atlas-7, a 4-billion-image dataset, the "
                  "company said Tuesday in a statement."),
    ("AIWire", "On Tuesday GaiaCorp said in a statement it had released Atlas-7, a "
               "dataset containing 4 billion images."),
]
CLAIM_B = "Atlas-7's shards overlap heavily, inflating its real size."
CLAIM_B_PASSAGES = [
    ("researcher blog", "I pulled Atlas-7 apart shard by shard; the same images "
                        "keep reappearing across shards, so the real count is lower."),
    ("university audit", "A near-duplicate analysis finds that roughly a third of "
                         "Atlas-7's images repeat, cutting its effective size."),
]


def dedup_sources(passages: list, threshold: float = SIM_THRESHOLD) -> list:
    """Cluster near-duplicate passages by embedding similarity (greedy single-link
    over cosine). Returns a list of clusters, each a list of the original indices;
    the cluster count is the number of independent sources."""
    from genai.embed import embed, similarity
    vecs = [embed(p) for p in passages]
    clusters = []
    for i in range(len(passages)):
        for c in clusters:
            if any(similarity(vecs[i], vecs[j]) >= threshold for j in c):
                c.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def independence_report(sources: list, threshold: float = SIM_THRESHOLD):
    """For one claim's sources, the citation count, the independent-source count
    after dedup, whether it clears the bar, and the clusters (as outlet lists)."""
    clusters = dedup_sources([p for _, p in sources], threshold)
    return {"citations": len(sources), "independent": len(clusters),
            "passes": len(clusters) >= INDEPENDENCE_BAR,
            "clusters": [[sources[i][0] for i in c] for c in clusters]}


def show_source_independence() -> None:
    """Two claims the bare per-claim gate passes on citation count. Dedup by
    embedding shows the first rests on one wire story re-hosted (below the bar),
    the second on two independent reports (clears it)."""
    for claim, passages in [(CLAIM_A, CLAIM_A_PASSAGES),
                            (CLAIM_B, CLAIM_B_PASSAGES)]:
        r = independence_report(passages)
        show_turn("claim", claim)
        show_turn("cited by", ", ".join(o for o, _ in passages) +
                  f"  ({r['citations']} sources)")
        for c in r["clusters"]:
            tag = "one story, re-hosted" if len(c) > 1 else "independent"
            show_turn("cluster", f"{', '.join(c)}  ({tag})")
        verdict = "clears the 2-source bar" if r["passes"] else \
            "one real source: FAILS the 2-source bar"
        show_turn("INDEPENDENT", f"{r['independent']} of {r['citations']}; {verdict}")

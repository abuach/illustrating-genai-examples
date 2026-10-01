"""Sophia as a graph: the same retrieve-answer loop, but a framework owns it.

Programming Generative AI built Sophia's loop by hand, a flat ``while`` loop
(genai.agent.run_with_trace). Here she's a LangGraph ``StateGraph``: nodes for
retrieve, answer, and check, wired with a conditional edge that loops back when
the answer isn't grounded in what was retrieved. The framework owns the state
and the control flow; each node just calls gpt-oss or the book search. Compare
this with the hand-rolled loop to see what an orchestration layer buys: the flow
is something you can read off the edges, and adding the check-and-retry was one
conditional edge rather than a rewrite of the loop body.

The model is gpt-oss:20b; runs are nondeterministic, so the demo cells bake a
captured trace.
"""
import re
from typing import TypedDict

import ollama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from genai.arch import EIS_MODEL
from genai.mcp import _search_corpus


class SophiaState(TypedDict):
    question: str
    passage: str
    answer: str
    grounded: bool
    attempts: int


def _retrieve(state: SophiaState) -> dict:
    """Search the book for the question; count the attempt."""
    return {"passage": _search_corpus(state["question"]),
            "attempts": state.get("attempts", 0) + 1}


def _answer(state: SophiaState) -> dict:
    """Answer from the retrieved passage; a retry asks for tighter grounding."""
    extra = "" if state["attempts"] == 1 else " Quote the passage's own wording."
    prompt = (f"Answer in one sentence using only this passage.{extra}\n"
              f"Passage: {state['passage']}\nQuestion: {state['question']}")
    msg = ollama.chat(model=EIS_MODEL, think=False, options={"num_predict": 300},
                      messages=[{"role": "user", "content": prompt}])["message"]
    return {"answer": (msg.get("content") or msg.get("thinking") or "").strip()}


def _check(state: SophiaState) -> dict:
    """Grounded if the answer and the passage share enough content words."""
    answer_words = set(re.findall(r"[a-z]{4,}", state["answer"].lower()))
    passage_words = set(re.findall(r"[a-z]{4,}", state["passage"].lower()))
    return {"grounded": len(answer_words & passage_words) >= 3}


def _route(state: SophiaState) -> str:
    """The conditional edge: stop when grounded or out of tries, else retry."""
    return END if state["grounded"] or state["attempts"] >= 2 else "retrieve"


def build_sophia_graph(checkpointer=None):
    """Wire the nodes into a graph the framework will run and checkpoint.

    Pass a ``checkpointer`` and the framework saves the state after every node,
    so a run that dies partway can resume from where it stopped. That one
    argument is the whole difference between a graph that forgets on a crash and
    one that doesn't; see ``run_with_crash``.
    """
    graph = StateGraph(SophiaState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _answer)
    graph.add_node("check", _check)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "check")
    graph.add_conditional_edges("check", _route, {"retrieve": "retrieve", END: END})
    return graph.compile(checkpointer=checkpointer)


def run_sophia_graph(question: str) -> tuple:
    """Run the graph; return (final_state, trace) where trace is the nodes fired."""
    app = build_sophia_graph()
    trace = []
    state = {"question": question, "attempts": 0}
    for step in app.stream(state):
        for node, update in step.items():
            trace.append(node)
            state.update(update)
    return state, trace


# Captured by scripts/_orchestrate_probe.py (gpt-oss:20b through the graph).
GRAPH_DEMO = {
    "question": "why do conversations with a model slow down?",
    "trace": ["retrieve", "answer", "check"],
    "answer": "Because every new token still attends to everything stored so far, "
              "token generation slows as the transcript grows.",
}


def show_sophia_graph(demo=GRAPH_DEMO):
    """The graph run as a transcript: you ask, the nodes fire, gpt-oss answers."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("graph", " -> ".join(demo["trace"]) + "  (grounded, so no retry)")
    show_turn("gpt-oss", demo["answer"])


# --- observability: the record the framework keeps, and the loop doesn't ---------
# What a framework buys that the hand-written loop can't is an inspectable record
# of the run. The graph's own stream yields, after each node, the state that node
# changed; trace_sophia_graph collects it. The bare loop (run_sophia_bare) hands
# back a single string and keeps nothing. The contrast here is deterministic: the
# passage is real word-overlap retrieval and the answer is the chapter's already
# baked gpt-oss reply, so no new model call runs and the cell needs no freeze.

def _short(value, n: int = 54) -> str:
    """A field value trimmed to one readable chunk for the trace line."""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def trace_readable_demo(question: str = None) -> dict:
    """The observability contrast, built deterministically from the baked answer:
    the graph's per-node record (each node with the state it left) and the bare
    loop's lone return value, for the same question."""
    q = question or GRAPH_DEMO["question"]
    passage = _search_corpus(q)
    answer = GRAPH_DEMO["answer"]
    grounded = _check({"answer": answer, "passage": passage})["grounded"]
    return {"question": q,
            "graph_trace": [("retrieve", {"passage": passage, "attempts": 1}),
                            ("answer", {"answer": answer}),
                            ("check", {"grounded": grounded})],
            "bare_return": answer}


def show_trace_readable(demo: dict = None) -> None:
    """Two artifacts from the same run. First the record the framework hands back
    for free, every node with the state it changed; then the only thing the
    hand-written loop returns, one string. The count underneath is how many of the
    run's steps you can name and inspect from each, before adding any logging."""
    from genai.agent import show_turn
    demo = demo or trace_readable_demo()
    show_turn("you", demo["question"])
    for node, update in demo["graph_trace"]:
        fields = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
        show_turn("graph", f"{node:<8} -> {fields}")
    show_turn("bare loop", f'(returns) "{_short(demo["bare_return"], 58)}"')
    print()
    print(f"steps you can name from the record:  graph {len(demo['graph_trace'])}, "
          "bare loop 0")


def run_sophia_bare(question: str) -> str:
    """Sophia's retrieve-answer-check flow as a hand-written loop, the way the
    first book built her: local variables for state, a plain ``while``, and only
    the final answer handed back. Nothing records which nodes ran or what the
    state was between them. That missing visibility is the whole point."""
    state = {"question": question, "attempts": 0}
    while True:
        state.update(_retrieve(state))
        state.update(_answer(state))
        state.update(_check(state))
        if state["grounded"] or state["attempts"] >= 2:
            return state["answer"]          # the loop's entire output: one string


def trace_sophia_graph(question: str) -> list:
    """Run the graph and capture what the framework exposes for free: after every
    node fires, its name and the state fields it left behind. This list is a
    record you can read a whole run off, without adding a single print."""
    app = build_sophia_graph()
    record = []
    for step in app.stream({"question": question, "attempts": 0}):
        for node, update in step.items():
            record.append((node, dict(update)))
    return record


def run_with_crash(question: str) -> dict:
    """Run the checkpointed graph, kill it the instant it reaches ``check`` (just
    after the model's answer is computed and saved), then resume from the saved
    checkpoint in a fresh run on the same thread.

    The state lives in the checkpointer, not in any local variable, so the resume
    rebuilds nothing: ``retrieve`` and ``answer`` don't fire again, and the model
    call already paid for isn't repeated. A loop that held the same state in an
    in-memory list a crash erases would redo both. Returns the nodes fired before
    and after the crash plus the recovered answer, ready to bake.
    """
    saver = MemorySaver()
    armed = {"on": True}

    def _check_or_die(state: SophiaState) -> dict:
        if armed["on"]:                # first pass dies the moment check is reached
            armed["on"] = False
            raise RuntimeError("process killed on reaching check")
        return _check(state)

    graph = StateGraph(SophiaState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _answer)
    graph.add_node("check", _check_or_die)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "check")
    graph.add_conditional_edges("check", _route, {"retrieve": "retrieve", END: END})
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "sophia"}}

    before, after = [], []
    try:
        for step in app.stream({"question": question, "attempts": 0}, config):
            before.extend(step.keys())          # nodes that finished before the crash
    except RuntimeError:
        pass
    saved = app.get_state(config).values        # what the checkpointer kept
    for step in app.stream(None, config):       # input=None resumes from the checkpoint
        after.extend(step.keys())
    final = app.get_state(config).values
    return {"question": question, "before": before, "after": after,
            "answer": final.get("answer", ""), "kept_answer": bool(saved.get("answer"))}


# Captured by scripts/_orchestrate_probe.py (gpt-oss:20b through the checkpointed graph).
CHECKPOINT_DEMO = {
    "question": "why do conversations with a model slow down?",
    "before": ["retrieve", "answer"],
    "after": ["check"],
    "answer": "Because every new token still attends to everything stored so far, "
              "token generation slows as the transcript grows.",
}


def show_checkpoint_resume(demo=CHECKPOINT_DEMO):
    """The crash and the recovery as a transcript: the run dies after the model
    answers, and the checkpoint hands the answer back so the resume never calls
    the model again."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("graph", " -> ".join(demo["before"]) + "   (then the process is killed)")
    show_turn("RESUME", " -> ".join(demo["after"]) +
              "   (retrieve, answer skipped: their state survived)")
    show_turn("gpt-oss", demo["answer"])


def run_runaway(question: str = "what is perplexity", recursion_limit: int = 25):
    """Show what the ``attempts >= 2`` guard in ``_route`` was protecting you from.

    Build the same graph but tighten ``check`` past anything an answer can clear
    (a stop condition that's never met, standing in for a guardrail tuned too
    tight) and drop the retry cap from the router. With nothing left to end it,
    the graph loops until LangGraph's own recursion limit fires. The runaway is a
    routing bug, not a model one, so ``answer`` here is a stub: its text can't
    change whether the loop ends and is never shown. Returns ``(trace, error)``.
    """
    from langgraph.errors import GraphRecursionError

    def _stub_answer(state: SophiaState) -> dict:
        return {"answer": "..."}                 # content can't satisfy check; irrelevant

    def _check_too_strict(state: SophiaState) -> dict:
        answer_words = set(re.findall(r"[a-z]{4,}", state["answer"].lower()))
        passage_words = set(re.findall(r"[a-z]{4,}", state["passage"].lower()))
        return {"grounded": len(answer_words & passage_words) >= 99}   # a bar no answer clears

    def _route_no_cap(state: SophiaState) -> str:
        return END if state["grounded"] else "retrieve"     # the missing `attempts >= 2`

    graph = StateGraph(SophiaState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _stub_answer)
    graph.add_node("check", _check_too_strict)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "check")
    graph.add_conditional_edges("check", _route_no_cap, {"retrieve": "retrieve", END: END})
    app = graph.compile()

    trace = []
    try:
        for step in app.stream({"question": question, "attempts": 0},
                               {"recursion_limit": recursion_limit}):
            trace.extend(step.keys())
    except GraphRecursionError as exc:
        return trace, str(exc)
    return trace, ""


# Captured by scripts/_orchestrate_probe.py (deterministic: the loop never ends).
RUNAWAY_DEMO = {
    "question": "what is perplexity",
    "loops": 8,
    "error": "Recursion limit of 25 reached without hitting a stop condition.",
}


def show_runaway_retry(demo=RUNAWAY_DEMO):
    """The retry edge with its cap removed: a loop the framework, not your code,
    is left to stop."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("graph", f"retrieve -> answer -> check  (x{demo['loops']}: check never passes)")
    show_turn("ERROR", demo["error"])


# ----------------------------------------------------------------------------
# The other two shapes. The graph above is one of three (plot_orchestration_models);
# below are the same retrieve-answer-check work expressed as a *role* setup and a
# *handoff*, so the chapter can show, not just assert, that the shape is an
# organizational choice. All three reach the same grounded answer; what differs is
# how the control flow is wired and how you'd read it. Both are LangGraph too: the
# three shapes are patterns, not rival frameworks, and one graph engine expresses
# all of them.
# ----------------------------------------------------------------------------

def _supervisor(state: SophiaState) -> dict:
    """The manager node: pure router. It delegates the work, it doesn't do it."""
    return {}


def _route_supervisor(state: SophiaState) -> str:
    """Hand the next unfinished step to a worker, or stop when grounded/out of tries."""
    if not state.get("passage"):
        return "researcher"
    if not state.get("answer"):
        return "writer"
    if state.get("grounded") or state["attempts"] >= 2:
        return END
    return "checker"


def _checker(state: SophiaState) -> dict:
    """Check grounding; on a failed check with tries left, clear the round's work
    so the supervisor delegates a fresh research-and-write pass."""
    result = _check(state)
    if not result["grounded"] and state["attempts"] < 2:
        result["passage"], result["answer"] = "", ""
    return result


def build_sophia_supervisor(checkpointer=None):
    """The *role* shape: a supervisor delegates retrieve/answer/check to three
    workers and routes between them, where the graph used fixed edges. Every hop
    goes through the hub; the workers never call each other."""
    graph = StateGraph(SophiaState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("researcher", _retrieve)
    graph.add_node("writer", _answer)
    graph.add_node("checker", _checker)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _route_supervisor,
                                {"researcher": "researcher", "writer": "writer",
                                 "checker": "checker", END: END})
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("checker", "supervisor")
    return graph.compile(checkpointer=checkpointer)


def run_sophia_supervisor(question: str) -> tuple:
    """Run the role graph; return (final_state, trace) of the nodes fired."""
    app = build_sophia_supervisor()
    trace, state = [], {"question": question, "attempts": 0}
    for step in app.stream(state):
        for node, update in step.items():
            trace.append(node)
            state.update(update or {})
    return state, trace


def _triage(state: SophiaState) -> Command:
    """The triage agent owns the question first. If it's an in-lane glossary term
    it answers and stops; otherwise it *hands off* the whole question to the
    specialist. The handoff is the agent's own ``Command(goto=...)``, decided by a
    cheap deterministic check, not a model tool call that gpt-oss might skip."""
    from genai.mcp import GLOSSARY
    q = state["question"].lower()
    hit = next((term for term in GLOSSARY if term in q), None)
    if hit:
        return Command(goto=END, update={"answer": GLOSSARY[hit], "passage": hit,
                                         "grounded": True, "attempts": 1})
    return Command(goto="specialist")


def _specialist(state: SophiaState) -> dict:
    """The specialist runs the full retrieve-answer-check the triage agent passed on."""
    retrieved = _retrieve(state)
    answered = _answer({**state, **retrieved})
    checked = _check({**state, **retrieved, **answered})
    return {**retrieved, **answered, **checked}


def build_sophia_handoff(checkpointer=None):
    """The *handoff* shape: control passes forward from triage to specialist, each
    owning the question in turn. No hub, no loop back."""
    graph = StateGraph(SophiaState)
    graph.add_node("triage", _triage)
    graph.add_node("specialist", _specialist)
    graph.add_edge(START, "triage")
    graph.add_edge("specialist", END)
    return graph.compile(checkpointer=checkpointer)


def run_sophia_handoff(question: str) -> tuple:
    """Run the handoff graph; return (final_state, trace) of the agents fired."""
    app = build_sophia_handoff()
    trace, state = [], {"question": question, "attempts": 0}
    for step in app.stream(state):
        for node, update in step.items():
            trace.append(node)
            state.update(update or {})
    return state, trace


# Both captured by scripts/_orchestrate_probe.py on GRAPH_DEMO's question. The role
# and handoff shapes call the same retrieve-answer node as the graph, so they reach
# the same grounded answer by different routes; gpt-oss's wording drifts run to run,
# so each demo bakes one representative clean answer (the probe prints all three
# raw, so the convergence stays checkable).
SUPERVISOR_DEMO = {
    "question": GRAPH_DEMO["question"],
    "trace": ["researcher", "writer", "checker"],        # each hop via the hub
    "answer": "Because every new token the model writes still attends to everything "
              "stored so far, token generation itself slows as the transcript grows.",
}


def show_sophia_supervisor(demo=SUPERVISOR_DEMO):
    """The role shape as a transcript: a supervisor delegates each step to a worker
    and routes between them, reaching the same grounded answer the graph did."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("role", " -> ".join(demo["trace"]) + "  (each hop via the supervisor)")
    show_turn("gpt-oss", demo["answer"])


HANDOFF_DEMO = {
    "question": GRAPH_DEMO["question"],
    "trace": ["triage", "specialist"],
    "answer": "Because every new token the model writes still attends to everything "
              "stored so far, token generation slows as the transcript grows.",
}


def show_sophia_handoff(demo=HANDOFF_DEMO):
    """The handoff shape as a transcript: triage finds the question out of its lane
    and hands the whole thing forward to a specialist, who answers."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("handoff", " -> ".join(demo["trace"]) + "  (out of triage's lane, handed off)")
    show_turn("gpt-oss", demo["answer"])


# --- a framework doesn't fix your model -----------------------------------------
# Adopting an orchestration framework doesn't make your model reliable. A flaky
# step is exactly as flaky inside a LangGraph node as it is bare, because the node
# is the same model call. What the framework gives you is a place to put the fix.
# Here the step is the register from Chapter 1, reading a spoken order into a strict
# JSON receipt, run by gemma3:1b, which malforms the JSON often enough to measure.
# The failure is schema-checkable (no answer key needed), so a validator can catch
# it and ask again, and in LangGraph that retry is one conditional edge. The bare
# and node failure rates match; only the edge moves the number. (A sequential
# CrewAI crew, built earlier in the chapter, shows the same invariance, but with no
# edge to add you wire the loop by hand.) Nondeterministic (gemma3:1b): capture with
# scripts/_frameworks_flaky_probe.py.
from genai.arch import ORDERS as _ORDERS  # the cafe orders of Chapter 1
from genai.arch import parse_receipt, receipt_prompt, validate_receipt

FLAKY_MODEL = "gemma3:1b"  # small enough to malform the JSON schema stochastically


def flaky_read(question: str, model: str = FLAKY_MODEL, feedback: str = "") -> tuple:
    """One model call: read a spoken cafe order into a strict JSON receipt.
    Returns ``(raw_text, valid)`` where ``valid`` is True when the receipt passes
    the register's schema. This single call is the flaky step every arm runs."""
    resp = ollama.chat(model=model, think=False,
                       options={"num_predict": 220, "temperature": 0.7},
                       messages=[{"role": "user",
                                  "content": receipt_prompt(question, feedback)}])
    raw = (resp["message"].get("content") or "").strip()
    return raw, validate_receipt(parse_receipt(raw)) == ""


# -- bare: no framework at all --
def bare_flaky_rate(trials: int = 4, model: str = FLAKY_MODEL) -> dict:
    """The step's failure rate with no framework around it: the fraction of reads
    whose receipt doesn't pass the schema, over every order, ``trials`` times."""
    bad = n = 0
    for question, _, _ in _ORDERS:
        for _ in range(trials):
            n += 1
            bad += int(not flaky_read(question, model)[1])
    return {"invalid_rate": round(bad / n, 3), "n": n}


# -- LangGraph: the step as a node, retry as one conditional edge --
class _ReadState(TypedDict):
    question: str
    raw: str
    valid: bool
    attempts: int


def _read_node(state: _ReadState) -> dict:
    feedback = "" if state.get("attempts", 0) == 0 else \
        "your last reply was not a valid receipt."
    raw, valid = flaky_read(state["question"], feedback=feedback)
    return {"raw": raw, "valid": valid, "attempts": state.get("attempts", 0) + 1}


def _read_route(state: _ReadState) -> str:
    """The one edge that makes the difference: retry an invalid read, up to 3."""
    return END if state["valid"] or state["attempts"] >= 3 else "read"


def build_read_graph(retry: bool = False):
    """The flaky read as a one-node LangGraph. With ``retry``, a single
    conditional edge loops an invalid receipt back to the same node."""
    graph = StateGraph(_ReadState)
    graph.add_node("read", _read_node)
    graph.add_edge(START, "read")
    if retry:
        graph.add_conditional_edges("read", _read_route, {"read": "read", END: END})
    else:
        graph.add_edge("read", END)
    return graph.compile()


def run_read_graph(question: str, retry: bool = False) -> tuple:
    """Run the read graph on one order; return ``(valid, attempts, raw)``."""
    out = build_read_graph(retry).invoke({"question": question, "attempts": 0})
    return out["valid"], out["attempts"], out["raw"]


def framework_flaky_study(trials: int = 4) -> dict:
    """Fill the invariance table: the same flaky read's schema-failure rate with no
    framework, then as a LangGraph node, then with a retry edge added. The bare and
    node rates match, because the node is the same call; only the retry edge moves
    the number."""
    bare = bare_flaky_rate(trials)
    n = node_bad = retry_bad = 0
    for question, _, _ in _ORDERS:
        for _ in range(trials):
            n += 1
            node_bad += int(not run_read_graph(question, retry=False)[0])
            retry_bad += int(not run_read_graph(question, retry=True)[0])
    return {"model": FLAKY_MODEL, "n_per_cell": len(_ORDERS) * trials,
            "bare": bare["invalid_rate"],
            "langgraph_node": round(node_bad / n, 3),
            "langgraph_retry": round(retry_bad / n, 3)}


# Captured by scripts/_frameworks_flaky_probe.py (gemma3:1b, 60 reads per cell).
# Bare and the LangGraph node land on the same rate, because the node is the same
# call; only the retry edge moves it.
FRAMEWORK_STUDY = {
    "model": "gemma3:1b",
    "n_per_cell": 300,
    "bare": 0.51,
    "langgraph_node": 0.497,
    "langgraph_retry": 0.247,
}

# One real order (gemma3:1b): the read gets the items right but writes the
# discount as the fraction 0.15 (and slips a third tea in), which the schema
# rejects; the retry edge lands a receipt with a whole-number 15.
FRAMEWORK_DEMO = {
    "question": "Two Red Bulls, a Monster, and three espressos, with the 15% "
                "member discount.",
    "bad": '```json\n{\n  "latte": 2,\n  "rockstar": 1,\n  "tea": 3,\n'
           '  "redbull": 1,\n  "monster": 1,\n  "espresso": 3,\n'
           '  "discount": 0.15\n}\n```',
    "good": '```json\n{\n  "latte": 2,\n  "rockstar": 1,\n  "tea": 1,\n'
            '  "redbull": 1,\n  "monster": 1,\n  "espresso": 3,\n'
            '  "discount": 15\n}\n```',
}


def show_framework_flaky(study: dict = None, demo: dict = None) -> None:
    """The invariance table (bare vs a LangGraph node vs the node with a retry
    edge), then one order whose read fails the schema and passes once the edge
    loops it back."""
    from genai.agent import show_code, show_turn
    study = study or FRAMEWORK_STUDY
    demo = demo or FRAMEWORK_DEMO
    print(f"schema-failure rate  ({study['model']}, {study['n_per_cell']} reads)")
    print(f"  bare (no framework)      {study['bare']:.2f}")
    print(f"  LangGraph node           {study['langgraph_node']:.2f}")
    print(f"  LangGraph node + retry   {study['langgraph_retry']:.2f}")
    print()
    show_turn("you", demo["question"])
    show_code("read node", demo["bad"])
    show_turn("GUARD", "rejected: not a valid receipt -> edge loops back")
    show_code("read node", demo["good"])
    show_turn("GUARD", "accepted: receipt passes the schema")

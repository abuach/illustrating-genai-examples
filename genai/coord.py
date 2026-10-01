"""When one agent becomes many: the shared-state hazards of concurrent agents.

One agent is a program you can read top to bottom. Fan that agent out into
several that run at once and share state, and you have a small distributed
system, with its classic hazards: races, lost updates, deadlocks. This module
stages the simplest one. Two of Sophia's sub-agents each research a sub-question
and append their finding to one shared notes file. Done without coordination,
the two read-modify-write cycles interleave and one finding is silently lost;
done under a lock, both survive.

``run_unsync`` / ``run_sync`` run the two writers against a shared file and hand
back the notes that survived. ``race_study`` repeats that to measure the
corruption rate; ``fanout_study`` times K real gpt-oss sub-tasks run one after
another versus all at once. The model is gpt-oss:20b, and every timing and
concurrency number is nondeterministic, so the studies are captured once
(scripts/_coord_probe.py), baked into the constants below, and their notebook
cells are frozen.

``build_fanout_graph`` / ``run_graph`` / ``graph_study`` run the same race one
layer up, as a LangGraph parallel branch whose two writer nodes share one
``findings`` channel. A reducer declared on that channel merges the concurrent
writes; leave it off and LangGraph raises ``InvalidUpdateError``. Either way the
silent lost update is gone by construction, the same move the lock made, lifted
out of the node bodies and into the state definition.
"""
import json
import operator
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from genai.arch import EIS_MODEL

# A deliberately widened race window. A real lost update hides in a window a few
# nanoseconds long, far too short to land on reliably in a teaching demo, so we
# pause between the read and the write to make the hazard show every time. The
# bug is the same; we have only slowed the film down enough to see the frame.
_WINDOW = 0.03

# Two fixed payloads for the corruption study. Their content is irrelevant to the
# race (it measures the file mechanics, not the model), so the study runs fast
# without a single model call; the real model findings live in RACE_DEMO below.
FINDINGS = {"agent-1": "race conditions corrupt shared state",
            "agent-2": "deadlocks freeze every waiter"}


def _append_finding(path: str, agent: str, finding: str, lock=None) -> None:
    """One agent's read-modify-write: read the notes, add mine, write them back.

    With no lock the read and the write straddle the pause, so a second writer
    reads the same old notes and overwrites the first writer's entry. The file
    stays valid JSON; it just quietly holds one finding where it should hold two.
    """
    if lock:
        lock.acquire()
    try:
        notes = json.load(open(path)) if os.path.getsize(path) else []
        time.sleep(_WINDOW)                         # the window the race lives in
        notes.append({"agent": agent, "finding": finding})
        json.dump(notes, open(path, "w"))
    finally:
        if lock:
            lock.release()


def _safe_load(path: str):
    """Read the notes file, returning ``None`` when concurrent writes left it
    malformed. Two writes that interleave at the byte level can splice their JSON
    into something no parser will accept, which is corruption louder than, but no
    different in kind from, a silent lost update."""
    try:
        return json.load(open(path)) if os.path.getsize(path) else []
    except json.JSONDecodeError:
        return None


def _run_writers(findings: dict, use_lock: bool):
    """Two agents append to one shared file at once; return the notes that survived
    (or ``None`` if the file was left unparseable)."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    open(path, "w").write("")                       # an empty shared notes file
    lock = threading.Lock() if use_lock else None
    with ThreadPoolExecutor(max_workers=len(findings)) as pool:
        for agent, finding in findings.items():
            pool.submit(_append_finding, path, agent, finding, lock)
    notes = _safe_load(path)
    os.unlink(path)
    return notes


def run_unsync(findings: dict = FINDINGS) -> list:
    """The two writers with no coordination: expect a lost update."""
    return _run_writers(findings, use_lock=False)


def run_sync(findings: dict = FINDINGS) -> list:
    """The two writers under one lock: each takes its turn, both survive."""
    return _run_writers(findings, use_lock=True)


def race_study(trials: int = 60, findings: dict = FINDINGS) -> dict:
    """Run the two writers many times each way; count the runs that lost a finding."""
    want = len(findings)

    def corrupt_pct(fn) -> int:
        runs = (fn(findings) for _ in range(trials))
        bad = sum(notes is None or len(notes) != want for notes in runs)
        return round(100 * bad / trials)

    return {"unsync": corrupt_pct(run_unsync),
            "sync": corrupt_pct(run_sync),
            "trials": trials}


# --- sync vs async fan-out (real gpt-oss timing) -------------------------------

SUBTASKS = [
    "In one sentence, what is a race condition in concurrent programming?",
    "In one sentence, what is a deadlock in concurrent programming?",
    "In one sentence, what is a mutex (lock) used for?",
    "In one sentence, what is a context switch?",
]


def _answer(task: str, model: str = EIS_MODEL) -> str:
    """One sub-agent: ask the model a sub-question, return its sentence."""
    msg = ollama.chat(model=model, think=False, options={"num_predict": 200},
                      messages=[{"role": "user", "content": task}])["message"]
    return (msg.get("content") or msg.get("thinking") or "").strip()


def fanout_study(tasks: list = SUBTASKS, model: str = EIS_MODEL) -> dict:
    """Time K sub-tasks run sequentially vs all at once, and count empty replies."""
    _answer("warm up", model)                       # pay the load cost once, up front

    t0 = time.time()
    seq = [_answer(t, model) for t in tasks]
    seq_s = time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        conc = list(pool.map(lambda t: _answer(t, model), tasks))
    conc_s = time.time() - t0

    return {"k": len(tasks),
            "sequential_s": round(seq_s, 1),
            "concurrent_s": round(conc_s, 1),
            "speedup": round(seq_s / conc_s, 2),
            "failures": sum(not a for a in seq + conc)}


# --- baked captures (scripts/_coord_probe.py, gpt-oss:20b on a quiet machine) ---
# Real model findings and real measured studies, kept verbatim so the cells are
# reproducible without a rerun. Every number was computed, never typed.

_RACE_FINDING = ("A race condition occurs when two or more threads try to read or "
                 "write shared data at the same time without proper synchronization, "
                 "causing the program's outcome to become unpredictable.")
_DEADLOCK_FINDING = ("A deadlock occurs when two or more threads each wait "
                     "indefinitely for resources held by the others, causing all of "
                     "them to halt.")

RACE_DEMO = {
    "task": "Research two concurrency hazards and save both findings to the notes file.",
    "findings": {"agent-1": _RACE_FINDING, "agent-2": _DEADLOCK_FINDING},
    # An observed lost-update run: agent-2's write landed last and overwrote
    # agent-1's, so only the deadlock finding survived.
    "unsync_file": [{"agent": "agent-2", "finding": _DEADLOCK_FINDING}],
    "sync_file": [{"agent": "agent-1", "finding": _RACE_FINDING},
                  {"agent": "agent-2", "finding": _DEADLOCK_FINDING}],
}

RACE_STUDY = {"unsync": 100, "sync": 0, "trials": 60}

# Measured isolated on a quiet machine (the contended first capture inflated the
# sequential leg ~4x; this is the trustworthy run). Ollama batches the concurrent
# requests on one GPU, so four at once finish in well under four times one.
FANOUT_STUDY = {"k": 4, "sequential_s": 137.2, "concurrent_s": 52.6,
                "speedup": 2.61, "failures": 0}


def show_race_demo(demo=RACE_DEMO):
    """The race as a transcript: two sub-agents write, one finding is lost, then kept."""
    from genai.agent import show_turn
    show_turn("you", demo["task"])
    show_turn("agent-1", demo["findings"]["agent-1"])
    show_turn("agent-2", demo["findings"]["agent-2"])
    kept = ", ".join(n["agent"] for n in demo["unsync_file"])
    show_turn("FILE", f"unsynchronized -> notes.json holds 1 of 2 findings ({kept} "
                      "overwrote the other): a silent lost update")
    both = ", ".join(n["agent"] for n in demo["sync_file"])
    show_turn("FILE", f"under a lock     -> notes.json holds 2 of 2 findings ({both}): "
                      "both writers survive")


# --- the same race, through a framework (a LangGraph parallel branch) -----------
# The lock fixed the race in code we wrote, inside the node bodies. A framework can
# fix it one layer up, in the state definition. LangGraph runs the two writers as
# parallel branches of a graph that share one ``findings`` channel, and how that
# channel is declared is the whole story: give it a reducer and the two concurrent
# writes merge; leave it a bare value and the second write raises InvalidUpdateError.
# Neither outcome is the silent lost update the raw file produced, because that
# interleaving is no longer in the space of things the graph can do.


def _writer(agent: str, finding: str):
    """One graph node: hand this agent's finding back as an update to the channel."""
    return lambda state: {"findings": [{"agent": agent, "finding": finding}]}


def build_fanout_graph(findings: dict = FINDINGS, reduce: bool = True):
    """The two writers as a LangGraph fan-out: both fire from the start and write
    the one shared ``findings`` channel. The reducer on that channel, not a lock in
    the node bodies, is what makes the concurrent writes safe."""
    if reduce:
        class Notes(TypedDict):
            findings: Annotated[list, operator.add]  # reducer: merge, don't clobber
    else:
        class Notes(TypedDict):
            findings: list                           # bare: one write per step only

    graph = StateGraph(Notes)
    for agent, finding in findings.items():
        graph.add_node(agent, _writer(agent, finding))
        graph.add_edge(START, agent)                 # fan out: both from the start
        graph.add_edge(agent, END)
    return graph.compile()


def run_graph(findings: dict = FINDINGS, reduce: bool = True) -> list:
    """Invoke the fan-out. Reduced, it returns both findings merged; unreduced, it
    raises InvalidUpdateError rather than ever returning one finding of two."""
    return build_fanout_graph(findings, reduce).invoke({"findings": []})["findings"]


def graph_study(trials: int = 60, findings: dict = FINDINGS) -> dict:
    """race_study, one layer up. Run the reduced fan-out ``trials`` times and count
    the runs that lost a finding; then run it unreduced and count how many raised
    loudly instead of dropping a finding in silence."""
    want = len(findings)
    lost = sum(len(run_graph(findings, reduce=True)) != want for _ in range(trials))
    raised = 0
    for _ in range(trials):
        try:
            run_graph(findings, reduce=False)
        except Exception:
            raised += 1
    return {"reducer_lost_pct": round(100 * lost / trials),
            "raised_pct": round(100 * raised / trials), "trials": trials}


# The framework study, captured once (scripts/_coord_probe.py). The reducer loses
# nothing over sixty runs, matching the lock; without it, all sixty writes raise
# InvalidUpdateError rather than dropping one in silence.
GRAPH_STUDY = {"reducer_lost_pct": 0, "raised_pct": 100, "trials": 60}

# The three-condition scale the race-fix chart draws once the framework is in play:
# the raw race, the lock, and the reducer, sixty runs each, straight from the two
# studies above so the bars stay a single source of truth.
RACE_FIX_3WAY = {"unsync": RACE_STUDY["unsync"], "sync": RACE_STUDY["sync"],
                 "reducer": GRAPH_STUDY["reducer_lost_pct"], "trials": RACE_STUDY["trials"]}

# The real InvalidUpdateError from a bare (no-reducer) channel, kept verbatim (its
# troubleshooting-URL footer elided). The framework names its own fix in the error.
_NO_REDUCER_ERROR = ("InvalidUpdateError: At key 'findings': Can receive only one "
                     "value per step. Use an Annotated key to handle multiple values.")

# The reduced fan-out's merged channel, captured once: both real findings survive,
# in the order the reducer combined them. Reuses the same two gpt-oss findings the
# file demo showed, so the only thing that changed is the substrate.
GRAPH_RACE = {
    "task": "Research two concurrency hazards and record both findings in shared state.",
    "merged": [{"agent": "agent-1", "finding": _RACE_FINDING},
               {"agent": "agent-2", "finding": _DEADLOCK_FINDING}],
    "error": _NO_REDUCER_ERROR,
}


def show_graph_race(demo=GRAPH_RACE):
    """The same race through a LangGraph fan-out: two writers, one channel, no lock.
    With a reducer the writes merge; without one the framework raises loudly. Neither
    is the silent lost update the raw file produced."""
    from genai.agent import show_turn
    show_turn("you", demo["task"])
    kept = ", ".join(n["agent"] for n in demo["merged"])
    show_turn("MERGE", f"reducer channel -> state holds 2 of 2 ({kept}): both writes "
                       "merge, nothing lost")
    show_turn("ERROR", f"bare channel    -> {demo['error']}")


# ── The agent that waited forever ─────────────────────────────────────────────
# The race lost an update because two agents wrote at once. Deadlock is the
# opposite failure: two agents that need the same two resources, and each grabs
# them in the other's order, so each ends up holding what the other is waiting for
# and neither ever moves. Nothing crashes; the system just stops. The cure is a
# discipline, not a lock: everyone acquires shared resources in one agreed global
# order, which makes a waiting cycle impossible. Real threads and real locks; the
# outcome is deterministic (a brief hold forces the overlap), so the cell runs
# live, and a timeout is what turns "waits forever" into an observation.

def run_two_agents(ordered: bool, hold: float = 0.05, timeout: float = 0.5) -> dict:
    """Two agents each need the shared report and the shared database. In the
    ``ordered`` version both take them in one global order; otherwise the writer
    takes them in the opposite order and the two deadlock. Each holds its first
    lock briefly so the overlap is forced. Returns ``{"finished", "deadlocked"}``."""
    report, database = threading.Lock(), threading.Lock()
    start, finished = threading.Event(), []

    def agent(name, locks):
        start.wait()
        first, second = locks
        with first:
            time.sleep(hold)                 # hold it while the other grabs theirs
            with second:
                finished.append(name)

    planner_order = (report, database)
    writer_order = (report, database) if ordered else (database, report)
    threads = [threading.Thread(target=agent, args=("planner", planner_order),
                                daemon=True),
               threading.Thread(target=agent, args=("writer", writer_order),
                                daemon=True)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout)
    return {"finished": sorted(finished), "deadlocked": len(finished) < 2}


def show_deadlock(timeout: float = 0.5) -> None:
    """Two agents needing the same two resources in opposite order deadlock; the
    same two under one global acquisition order both finish."""
    from genai.agent import show_turn
    show_turn("SETUP", "two agents each need the shared report and the shared database")
    bad = run_two_agents(ordered=False, timeout=timeout)
    show_turn("planner", "locks the report, then reaches for the database")
    show_turn("writer", "locks the database, then reaches for the report")
    show_turn("DEADLOCK", f"each holds what the other waits for: "
              f"{len(bad['finished'])} of 2 finish within {timeout:g}s")
    good = run_two_agents(ordered=True, timeout=timeout)
    show_turn("FIX", "both take the report first, then the database: one global order")
    show_turn("DONE", f"{len(good['finished'])} of 2 finish: one waits for the "
              "report, then goes; no cycle to wait on")


# ── The worker that reported done too soon ────────────────────────────────────
# The race clobbered a write; the deadlock froze two waiters. This third hazard is
# a coordinator that stops waiting too early. It fans out sub-agents, each of which
# researches its piece then appends a finding to the shared notes, and it reads the
# notes the moment every worker has reported "done". The bug is in what "done"
# means. Signal it when the finding is *computed*, a beat before the append is
# written, and the coordinator's read can land while the slowest worker is still
# mid-write, so the report ships one finding short. Move the signal to *after* the
# write and the identical wait becomes a real completion barrier. Timing forces the
# difference, so the outcome is stable (200/200 either way); baked and frozen per this
# module's convention for concurrency numbers.

# (agent, research seconds, finding). Two quick sub-tasks and one slow one; the slow
# finding is, fittingly, the very hazard this section is about.
COMPLETION_WORKERS = [
    ("agent-1", 0.02, "a race lets two writers clobber one another's update"),
    ("agent-2", 0.02, "a deadlock freezes every agent in a waiting cycle"),
    ("agent-3", 0.15, "a premature read ships the report before the slow write lands"),
]


def run_coordinated(signal_before_write: bool, workers: list = None,
                    window: float = 0.03, timeout: float = 2.0) -> list:
    """Fan out ``workers`` that each research (a delay), append a finding to shared
    notes, and signal done. The coordinator waits for every done-signal, then reads.
    When ``signal_before_write`` the done-signal fires a ``window`` before the append
    lands (the bug), so the slowest worker is still mid-write when the read happens;
    otherwise it fires only after the append (the fix). Returns the notes read."""
    workers = workers or COMPLETION_WORKERS
    notes, lock, done = [], threading.Lock(), threading.Semaphore(0)

    def worker(name, delay, finding):
        time.sleep(delay)                                   # research this sub-question
        if signal_before_write:
            done.release()                                  # "done" = computed (the bug)
            time.sleep(window)                              # ...still writing
            with lock:
                notes.append({"agent": name, "finding": finding})
        else:
            time.sleep(window)
            with lock:
                notes.append({"agent": name, "finding": finding})
            done.release()                                  # "done" = written (the fix)

    threads = [threading.Thread(target=worker, args=w, daemon=True) for w in workers]
    for t in threads:
        t.start()
    for _ in workers:
        done.acquire()                                      # the barrier: N done-signals
    with lock:
        report = list(notes)                                # the coordinator reads here
    for t in threads:
        t.join(timeout)                                     # let stragglers finish, then exit
    return report


def show_completion_barrier(workers: list = None) -> None:
    """Two coordinators over the same fan-out. The first reads when every worker has
    signalled done-as-computed, catching the slowest one mid-write; the second waits
    for done-as-written, and its identical read misses nothing."""
    from genai.agent import show_turn
    workers = workers or COMPLETION_WORKERS
    n = len(workers)
    show_turn("SETUP", f"a coordinator fans out {n} sub-agents; each appends one "
                       "finding to the shared notes, then reads when all report done")
    early = run_coordinated(True, workers)
    present = {note["agent"] for note in early}
    missing = next(w for w in workers if w[0] not in present)
    show_turn("EARLY READ", f"done fires when the finding is computed, before the "
                            f"append lands: {len(early)} of {n} on the page")
    show_turn("LOST", f"{missing[0]} was still mid-write: \"{missing[2]}\"")
    barrier = run_coordinated(False, workers)
    show_turn("BARRIER", f"done fires only after the append is written: the same wait "
                        f"reads {len(barrier)} of {n}, nothing missed")

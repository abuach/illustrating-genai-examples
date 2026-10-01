"""Planning as a tool the model reaches for, the way Chapter 4 reached for Z3.

A plan is a sequence of moves you can check before you act: replay it from the
start, confirm every step's preconditions held, and confirm the goal is reached
at the end. A generative model is fluent at *proposing* such a sequence and
unreliable at getting a long one right, the same split this book keeps finding.
The neuro-symbolic fix is LLM-Modulo: the model translates a goal in words into a
formal planning *problem*, a classical planner searches for a plan that provably
reaches it, and a verifier checks the plan before anyone moves. The model does the
translating; the planner does the reasoning.

The domain is a courier running errands around a small town: move along roads,
pick up one parcel at a time, drop it where it belongs. ``DOMAIN_PDDL`` is the
fixed action library (shown on the page). ``formalize`` asks the model for the
*problem* (objects, init, goal); ``plan_bfs`` searches it; ``verify_plan`` replays
it. The model is gpt-oss:20b; its real outputs are captured once by
``scripts/_plan_probe.py`` and baked into ``PLAN_DEMO`` / ``DIRECT_DEMO`` so the
chapter is reproducible. The planner and verifier are deterministic.
"""
import re

import ollama

from genai.arch import EIS_MODEL


# ── The town and the errand (the ground truth, never hand-typed into a demo) ───
# A small road network and a Saturday errand: mail the letter, return the book,
# bring the coffee beans home. The courier carries one parcel at a time, which is
# what turns three quick drop-offs into a routing puzzle worth a planner.
LOCATIONS = ["home", "cafe", "post", "library"]
ROADS = [("home", "cafe"), ("cafe", "post"), ("cafe", "library"),
         ("post", "library")]                       # undirected; both ways added
PARCELS = {                                          # parcel: (starts at, belongs at)
    "letter": ("home", "post"),
    "book":   ("home", "library"),
    "beans":  ("cafe", "home"),
}

SCENARIO = (
    "A courier starts at home, empty-handed, and can carry only one parcel at a "
    "time. Roads connect home-cafe, cafe-post, cafe-library, and post-library "
    "(each road runs both ways). The letter is at home and must be mailed from "
    "the post office. The book is at home and must be returned to the library. "
    "The coffee beans are at the cafe and must be brought home. Find a sequence "
    "of moves that delivers all three.")


def _canonical():
    """The true start state and goal as atom sets, built from the town above."""
    init = {("at", "home"), ("hand-empty",)}
    for a, b in ROADS:
        init |= {("road", a, b), ("road", b, a)}
    for p, (start, _) in PARCELS.items():
        init.add(("parcel-at", p, start))
    goal = {("parcel-at", p, dest) for p, (_, dest) in PARCELS.items()}
    return frozenset(init), frozenset(goal)


CANON_INIT, CANON_GOAL = _canonical()


# ── The domain: three actions, shown on the page as PDDL ──────────────────────
# This is the fixed "library" the model targets. The model never writes it; it
# writes only the problem (which parcels start where, where they belong).
DOMAIN_PDDL = """(define (domain errands)
  (:requirements :strips)
  (:predicates (at ?l) (road ?a ?b) (parcel-at ?p ?l)
               (holding ?p) (hand-empty))
  (:action move
     :parameters (?from ?to)
     :precondition (and (at ?from) (road ?from ?to))
     :effect (and (at ?to) (not (at ?from))))
  (:action pick
     :parameters (?p ?l)
     :precondition (and (at ?l) (parcel-at ?p ?l) (hand-empty))
     :effect (and (holding ?p) (not (parcel-at ?p ?l)) (not (hand-empty))))
  (:action drop
     :parameters (?p ?l)
     :precondition (and (at ?l) (holding ?p))
     :effect (and (parcel-at ?p ?l) (hand-empty) (not (holding ?p)))))"""


# ── The classical planner: forward breadth-first search over ground states ────
# Same algorithm class as a real STRIPS planner like pyperplan, just small enough
# to read. A state is a frozenset of true atoms; BFS returns the shortest action
# sequence that makes the goal a subset of the state, or None if none exists.

def _objects(init):
    """Pull the locations and parcels named anywhere in the start state."""
    locs = {a[1] for a in init if a[0] in ("at", "road")} | {a[2] for a in init
                                                              if a[0] == "road"}
    parcels = {a[1] for a in init if a[0] in ("parcel-at", "holding")}
    return sorted(locs), sorted(parcels)


def _grounded_actions(init):
    """Every concrete move/pick/drop the town allows, as (name, pre, add, del)."""
    locs, parcels = _objects(init)
    roads = sorted((a[1], a[2]) for a in init if a[0] == "road")  # stable order
    actions = []
    for a, b in roads:
        actions.append((f"move({a}, {b})", {("at", a), ("road", a, b)},
                        {("at", b)}, {("at", a)}))
    for p in parcels:
        for l in locs:
            actions.append((f"pick({p}, {l})",
                            {("at", l), ("parcel-at", p, l), ("hand-empty",)},
                            {("holding", p)}, {("parcel-at", p, l), ("hand-empty",)}))
            actions.append((f"drop({p}, {l})", {("at", l), ("holding", p)},
                            {("parcel-at", p, l), ("hand-empty",)}, {("holding", p)}))
    return actions


def plan_search(init, goal, actions, max_states: int = 200000):
    """Shortest plan over an explicit action set; the domain-generic BFS core.

    ``actions`` is a list of ``(name, pre, add, del)`` ground moves, so the same
    search drives the courier town and the blocks world below. A state is a
    frozenset of true atoms; returns the shortest sequence whose replay makes
    ``goal`` a subset of the state, or None if none exists within ``max_states``.
    """
    init, goal = frozenset(init), frozenset(goal)
    if goal <= init:
        return []
    seen, frontier = {init}, [(init, [])]
    while frontier and len(seen) < max_states:
        state, plan = frontier.pop(0)
        for name, pre, add, rem in actions:
            if pre <= state:
                nxt = (state - rem) | add
                if nxt in seen:
                    continue
                if goal <= nxt:
                    return plan + [name]
                seen.add(nxt)
                frontier.append((nxt, plan + [name]))
    return None


def plan_bfs(init, goal, max_states: int = 200000):
    """Shortest courier plan from ``init`` to a state containing ``goal``; None if
    unsolvable. The town's roads and parcels are read straight out of ``init``."""
    return plan_search(init, goal, _grounded_actions(init), max_states)


def search_size(init, goal, actions, max_states: int = 200000):
    """Run the same BFS but report the cost of the search, not the plan: the number
    of distinct states it had to expand before it found a goal or gave up, and
    whether it hit the ``max_states`` ceiling. This is how the planner's own wall
    is measured, the one it announces out loud instead of papering over."""
    init, goal = frozenset(init), frozenset(goal)
    seen, frontier, expanded = {init}, [(init, [])], 0
    while frontier and len(seen) < max_states:
        state, plan = frontier.pop(0)
        expanded += 1
        if goal <= state:
            return {"states": len(seen), "expanded": expanded,
                    "plan_len": len(plan), "solved": True, "hit_cap": False}
        for name, pre, add, rem in actions:
            if pre <= state:
                nxt = (state - rem) | add
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, plan + [name]))
    return {"states": len(seen), "expanded": expanded, "plan_len": None,
            "solved": False, "hit_cap": len(seen) >= max_states}


def plan_ucs(init, goal, actions, cost, default=0, max_states: int = 200000):
    """Cheapest plan, not shortest: uniform-cost (Dijkstra) search where each move
    carries a price from ``cost`` (a name->number map; anything unlisted costs
    ``default``, so moves cost mileage and free actions like pick/drop cost 0).
    Returns ``(plan, total_cost)`` or ``(None, inf)``. The distance between this and
    ``plan_search`` is exactly the distance between *valid* and *cheap*."""
    import heapq
    init, goal = frozenset(init), frozenset(goal)
    best, frontier, tie = {init: 0}, [(0, 0, init, [])], 0
    while frontier and len(best) < max_states:
        spent, _, state, plan = heapq.heappop(frontier)
        if goal <= state:
            return plan, spent
        for name, pre, add, rem in actions:
            if pre <= state:
                nxt = (state - rem) | add
                spent2 = spent + cost.get(name, default)
                if nxt not in best or spent2 < best[nxt]:
                    best[nxt] = spent2
                    tie += 1
                    heapq.heappush(frontier, (spent2, tie, nxt, plan + [name]))
    return None, float("inf")


_ACT = re.compile(r"(move|pick|drop)\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)")


def _phrase(atom):
    """One missing precondition in plain words, for an honest failure message."""
    if atom[0] == "at":
        return f"the courier at {atom[1]}"
    if atom[0] == "road":
        return f"a road {atom[1]}->{atom[2]}"
    if atom[0] == "parcel-at":
        return f"{atom[1]} still at {atom[2]}"
    if atom[0] == "holding":
        return f"the courier holding {atom[1]}"
    return "an empty hand"


def _apply(state, name):
    """Run one action string against a state; return (next_state, error_or_None)."""
    m = _ACT.match(name.strip())
    if not m:
        return state, f"can't read action {name!r}"
    verb, x, y = m.group(1), m.group(2).strip(), m.group(3).strip()
    if verb == "move":
        pre, add, rem = ({("at", x), ("road", x, y)}, {("at", y)}, {("at", x)})
    elif verb == "pick":
        pre = {("at", y), ("parcel-at", x, y), ("hand-empty",)}
        add, rem = {("holding", x)}, {("parcel-at", x, y), ("hand-empty",)}
    else:                                            # drop
        pre = {("at", y), ("holding", x)}
        add, rem = {("parcel-at", x, y), ("hand-empty",)}, {("holding", x)}
    missing = pre - state
    if missing:
        return state, "it needs " + _phrase(sorted(missing)[0])
    return (state - rem) | add, None


def verify_plan(init, goal, plan):
    """Replay ``plan`` from ``init``; return (ok, step_index, reason).

    The honest check a plan has to pass before anyone acts: every action's
    preconditions must hold when its turn comes, and the goal must be true at the
    end. ``step_index`` points at the first broken step, or -1 if the plan is clean.
    """
    state = frozenset(init)
    for i, name in enumerate(plan):
        state, err = _apply(state, name)
        if err:
            return False, i, err
    if not frozenset(goal) <= state:
        missing = sorted(frozenset(goal) - state)
        return False, len(plan), f"plan ends but goal unmet ({len(missing)} left)"
    return True, -1, "every step's preconditions held and all parcels delivered"


def score_plan(init, goal, plan) -> dict:
    """Replay a plan as far as its preconditions allow and grade it with partial
    credit: how many goal facts hold where it stops. A graded score (parcels
    delivered) is a steadier signal than pass/fail when contrasting noisy models."""
    state, broke_at = frozenset(init), -1
    for i, name in enumerate(plan):
        nxt, err = _apply(state, name)
        if err:
            broke_at = i
            break
        state = nxt
    goal = frozenset(goal)
    return {"valid": broke_at < 0 and goal <= state,
            "delivered": sum(g in state for g in goal), "of": len(goal),
            "steps": len(plan), "broke_at": broke_at}


# ── Parsing the model's PDDL problem into atoms the planner can search ─────────

def _atoms(block):
    """Pull flat atoms like ``(parcel-at letter post)`` out of a PDDL block,
    skipping the ``and`` / ``not`` connectives."""
    out = []
    for inner in re.findall(r"\(([^()]+)\)", block):
        toks = inner.split()
        if toks and toks[0] not in ("and", "not"):
            out.append(tuple(toks))
    return out


def _section(text, head):
    """Return the balanced-paren span of ``(:head ...)`` from a PDDL string."""
    start = text.find(head)
    if start < 0:
        return ""
    depth = 0
    for j in range(start, len(text)):            # start at the head's own '('
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return ""


def parse_problem(pddl: str):
    """Recover (init, goal) atom sets from a model-written PDDL problem file."""
    init = frozenset(_atoms(_section(pddl, "(:init")))
    goal = frozenset(_atoms(_section(pddl, "(:goal")))
    return init, goal


# ── The model's two jobs: plan directly, or write the problem to formalize ─────

def _chat(model: str, prompt: str, n_predict: int = 1600):
    """One model turn, with keep_alive so the 12GB model isn't reloaded per call."""
    return ollama.chat(model=model, think=True, keep_alive="10m",
                       options={"num_predict": n_predict},
                       messages=[{"role": "user", "content": prompt}])["message"]


def _emit(model: str, prompt: str, n_predict: int = 1600) -> str:
    """gpt-oss often answers in its thinking channel, so read both."""
    msg = _chat(model, prompt, n_predict)
    return (msg.get("content") or "") + "\n" + (msg.get("thinking") or "")


_PLACEHOLDER = {"place", "parcel", "from", "to"}            # format-echo tokens


def _plan_lines(block: str) -> list:
    """Action strings in a block, dropping any that name a format placeholder."""
    out = []
    for m in _ACT.finditer(block):
        if not ({m.group(2).strip().lower(), m.group(3).strip().lower()}
                & _PLACEHOLDER):
            out.append(m.group(0))
    return out


def direct_plan(scenario: str = SCENARIO, model: str = EIS_MODEL,
                n_predict: int = 1600) -> list:
    """Ask the model to plan the errand itself, no solver. Parse its FINAL plan
    (a fenced block, or the answer channel), never the scratch reasoning.
    Reasoning models need a larger ``n_predict`` to finish thinking and still
    reach the plan."""
    msg = _chat(model, scenario + "\n\nThink it through, then give your final plan "
                "as a fenced code block:\n```plan\n<one action per line>\n```\n"
                "Each line is move(place, place), pick(parcel, place), or "
                "drop(parcel, place), using the real names above.", n_predict)
    content, think = msg.get("content") or "", msg.get("thinking") or ""
    for text in (content, think):                  # final answer first, not scratch
        fenced = re.search(r"```(?:plan|\w+)?\s*\n(.*?)```", text, re.S)
        if fenced:
            lines = _plan_lines(fenced.group(1))
            if lines:
                return lines
    return _plan_lines(content) or _plan_lines(think)


def formalize(scenario: str = SCENARIO, model: str = EIS_MODEL,
              n_predict: int = 1600) -> str:
    """Ask the model to translate the errand into a PDDL problem for the planner.
    Reasoning models need a larger ``n_predict`` to finish thinking and still emit
    the problem."""
    prompt = (
        "Here is a planning domain:\n\n" + DOMAIN_PDDL + "\n\n"
        "Translate this errand into a PDDL problem for that domain. Use exactly "
        "these object names: locations home, cafe, post, library; parcels letter, "
        "book, beans (the coffee beans). Declare the objects, give the full :init "
        "(the courier's start, every road both ways, and where each parcel starts) "
        "and a :goal for where each parcel belongs. Output only the PDDL problem."
        "\n\nErrand: " + scenario)
    text = _emit(model, prompt, n_predict)
    fenced = re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start = body.find("(define")
    return (body[start:] if start >= 0 else body).strip()


def formalize_and_plan(scenario: str = SCENARIO, model: str = EIS_MODEL,
                       n_predict: int = 2500) -> dict:
    """LLM-Modulo end to end: the model writes a PDDL problem, the planner solves
    it, and the resulting plan is replayed against the TRUE errand. Grades the
    *translation*: a good problem yields a plan that delivers all parcels in the
    real world; a mis-translation yields a short, broken, or unsolvable one.
    Returns the grade plus the raw PDDL so a clean run can be baked."""
    pddl = formalize(scenario, model, n_predict)
    init, goal = parse_problem(pddl)
    plan = plan_bfs(init, goal) if goal else None
    if plan is None:
        return {"parsed": bool(goal), "planned": False, "valid": False,
                "delivered": 0, "of": len(CANON_GOAL), "pddl": pddl}
    grade = score_plan(CANON_INIT, CANON_GOAL, plan)   # replay in the real world
    return {"parsed": True, "planned": True, "valid": grade["valid"],
            "delivered": grade["delivered"], "of": grade["of"], "pddl": pddl}


# ── Captured runs (gpt-oss:20b via scripts/_plan_probe.py), baked for the page ─
# DIRECT_DEMO: the model's REAL one-shot plan, kept verbatim. The route is nearly
# right, but it calls the coffee *beans* "coffee", a name the errand world doesn't
# have, so step 6 can't run. The break is recomputed live by verify_plan against
# the true town, so the verdict on the page stays honest for this captured plan.
DIRECT_DEMO = {
    "plan": [
        "pick(letter, home)", "move(home, cafe)", "move(cafe, post)",
        "drop(letter, post)", "move(post, cafe)", "pick(coffee, cafe)",
        "move(cafe, home)", "drop(coffee, home)", "pick(book, home)",
        "move(home, cafe)", "move(cafe, library)", "drop(book, library)",
    ],
}

# PLAN_DEMO: the model's REAL emitted PDDL problem, kept verbatim (typed objects,
# its own comments, roads spelled both ways). The plan and the verdict shown on the
# page are searched and checked live from this captured text, so nothing is
# hand-asserted; only the model's translation is baked.
PLAN_DEMO = {
    "problem": """(define (problem errands-3)
  (:domain errands)

  (:objects
    home cafe post library - location
    letter book beans - parcel)

  (:init
    ; Courier starts at home, hand empty
    (at home)
    (hand-empty)

    ; Parcels initial locations
    (parcel-at letter home)
    (parcel-at book home)
    (parcel-at beans cafe)

    ; Roads (both directions)
    (road home cafe)   (road cafe home)
    (road cafe post)   (road post cafe)
    (road cafe library)(road library cafe)
    (road post library)(road library post))

  (:goal
    (and
      (parcel-at letter post)
      (parcel-at book library)
      (parcel-at beans home))))""",
}


# ── Showing the work on the page (built on the agentic transcript helpers) ─────

def show_direct_attempt(demo=DIRECT_DEMO):
    """ACT 1: the model plans the errand on its own, and the verifier catches the
    step where the plan breaks. The plan is the model's real output; the verdict is
    recomputed live, so it stays honest if the town ever changes."""
    from genai.agent import show_code, show_turn
    plan = demo["plan"]
    ok, idx, reason = verify_plan(CANON_INIT, CANON_GOAL, plan)
    show_turn("you", SCENARIO)
    show_code("gpt-oss", "\n".join(plan))
    if ok:
        show_turn("CHECK", "valid: " + reason)
    else:
        broke = plan[idx] if idx < len(plan) else "(the end)"
        show_turn("CHECK", f"invalid at step {idx + 1}, {broke}: {reason}")


def show_plan_demo(demo=PLAN_DEMO):
    """ACT 2: the model writes the PDDL problem, the planner searches it, the
    verifier replays the result. The plan and the verdict are computed live from
    the model's captured problem; only the translation is baked."""
    from genai.agent import show_code, show_turn
    init, goal = parse_problem(demo["problem"])
    plan = plan_bfs(init, goal)
    ok, _, reason = verify_plan(init, goal, plan)
    show_turn("you", "Mail the letter, return the book, bring the beans home.")
    show_code("gpt-oss", demo["problem"])
    show_turn("PLANNER", f"found a {len(plan)}-step plan: " + " -> ".join(plan))
    show_turn("CHECK", "valid: " + reason)


# ── The formalize contrast (which models can write the problem?) ───────────────
# Measured by scripts/_plan_models_probe.py: each model writes the PDDL problem
# for the SAME errand (object names pinned, so this grades structure not
# vocabulary), the planner solves it, and the plan is replayed against the true
# world. `solved` is how many of three runs delivered all three parcels; `kind`
# is whether the model reasons before answering. The point of the table: gpt-oss
# couldn't plan the errand directly (it broke in the first section), yet it
# formalizes it perfectly, and a non-reasoning model nearly matches it, while the
# two heavier reasoners stumble, not on the logic but on emitting a clean,
# solvable problem. Reasoning helps you plan; it doesn't help you translate.
PLAN_MODELS = {
    "gpt-oss:20b":        {"kind": "reasoning", "solved": 3, "trials": 3,
                           "note": "clean, solvable problems"},
    "gemma4:latest":      {"kind": "instruct",  "solved": 2, "trials": 3,
                           "note": "once wrote no problem at all"},
    "phi4-reasoning:14b": {"kind": "reasoning", "solved": 1, "trials": 3,
                           "note": "over-thinks, often no output"},
    "qwen3:4b":           {"kind": "reasoning", "solved": 0, "trials": 3,
                           "note": "parses, but unsolvable"},
}


def show_plan_models(study=PLAN_MODELS):
    """Print the formalize contrast as a small table, ranked by problems solved."""
    print(f"{'model':<20}{'kind':<11}{'solved':<8}what happened")
    rows = sorted(study.items(), key=lambda kv: -kv[1]["solved"])
    for model, d in rows:
        solved = f"{d['solved']}/{d['trials']}"
        print(f"{model:<20}{d['kind']:<11}{solved:<8}{d['note']}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Blocks world: why planning isn't just doing the subgoals in order.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# A second tiny domain, stacking blocks, because it stages a trap the courier
# errand can't: subgoals that fight each other. The classic example is the
# Sussman anomaly, where achieving either half of the goal first wrecks the
# other half, so no in-order, one-subgoal-at-a-time strategy can solve it. That
# is exactly the strategy you get when you ask a model to "do the steps", which
# is why this motivates a *searching* planner before we hand the model a syntax.

BLOCKS_DOMAIN_PDDL = """(define (domain blocks)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (holding ?x) (handempty))
  (:action pickup
     :parameters (?x)
     :precondition (and (clear ?x) (ontable ?x) (handempty))
     :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x))
                  (not (handempty))))
  (:action putdown
     :parameters (?x)
     :precondition (holding ?x)
     :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))
  (:action stack
     :parameters (?x ?y)
     :precondition (and (holding ?x) (clear ?y))
     :effect (and (on ?x ?y) (clear ?x) (handempty)
                  (not (holding ?x)) (not (clear ?y))))
  (:action unstack
     :parameters (?x ?y)
     :precondition (and (on ?x ?y) (clear ?x) (handempty))
     :effect (and (holding ?x) (clear ?y)
                  (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))))"""


def blocks_actions(blocks):
    """Every ground stack/unstack/pickup/putdown over ``blocks``, in the same
    ``(name, pre, add, del)`` shape ``plan_search`` expects."""
    acts = []
    for x in blocks:
        acts.append((f"pickup({x})",
                     {("clear", x), ("ontable", x), ("handempty",)},
                     {("holding", x)},
                     {("ontable", x), ("clear", x), ("handempty",)}))
        acts.append((f"putdown({x})", {("holding", x)},
                     {("ontable", x), ("clear", x), ("handempty",)},
                     {("holding", x)}))
        for y in blocks:
            if x == y:
                continue
            acts.append((f"stack({x}, {y})", {("holding", x), ("clear", y)},
                         {("on", x, y), ("clear", x), ("handempty",)},
                         {("holding", x), ("clear", y)}))
            acts.append((f"unstack({x}, {y})",
                         {("on", x, y), ("clear", x), ("handempty",)},
                         {("holding", x), ("clear", y)},
                         {("on", x, y), ("clear", x), ("handempty",)}))
    return acts


# The Sussman anomaly: C sits on A, B is on the table, and the goal is the tower
# A-on-B-on-C. Either subgoal, achieved first and then defended, blocks the other.
SUSSMAN_BLOCKS = ["A", "B", "C"]
SUSSMAN_INIT = frozenset({("on", "C", "A"), ("ontable", "A"), ("ontable", "B"),
                          ("clear", "C"), ("clear", "B"), ("handempty",)})
SUSSMAN_GOAL = frozenset({("on", "A", "B"), ("on", "B", "C")})


def _apply_ground(state, name, actions):
    """Apply a named ground action from ``actions`` to ``state`` (no error path;
    callers only ever apply moves a search already proved legal)."""
    for n, pre, add, rem in actions:
        if n == name:
            return (frozenset(state) - rem) | add
    return frozenset(state)


def _protected_search(init, subgoal, actions, keep, max_states: int = 200000):
    """Shortest plan from ``init`` to ``subgoal`` that never passes through a state
    breaking any atom in ``keep``. Goal protection is what makes a one-at-a-time
    planner *linear*, and linear planning is what the Sussman anomaly defeats."""
    init, subgoal, keep = frozenset(init), frozenset(subgoal), frozenset(keep)
    if subgoal <= init:
        return []
    seen, frontier = {init}, [(init, [])]
    while frontier and len(seen) < max_states:
        state, plan = frontier.pop(0)
        for name, pre, add, rem in actions:
            if pre <= state:
                nxt = (state - rem) | add
                if not keep <= nxt or nxt in seen:    # guard the won subgoals
                    continue
                if subgoal <= nxt:
                    return plan + [name]
                seen.add(nxt)
                frontier.append((nxt, plan + [name]))
    return None


def greedy_plan(init, goals, actions, protect: bool = True):
    """The intuitive strategy: take the goal facts one at a time, in the given
    order, and achieve each without disturbing the ones already won. Returns where
    it got stuck (or the whole plan if it somehow finished). On the Sussman anomaly
    it always stalls, whichever goal it starts with, because the second goal can't
    be reached without dismantling the first."""
    state, full, won = frozenset(init), [], set()
    for g in goals:
        sub = _protected_search(state, {g}, actions, won if protect else set())
        if sub is None:
            return {"ok": False, "stuck_on": g, "plan": full, "won": sorted(won)}
        for a in sub:
            state = _apply_ground(state, a, actions)
        full += sub
        won.add(g)
    return {"ok": True, "stuck_on": None, "plan": full, "won": sorted(won)}


def _phrase_goal(atom):
    """A goal atom as plain words, e.g. ('on','A','B') -> 'A on B'."""
    return f"{atom[1]} on {atom[2]}" if atom[0] == "on" else " ".join(atom)


def show_sussman(order=(("on", "A", "B"), ("on", "B", "C"))):
    """The motivating contrast: a linear, goal-protecting strategy stalls on the
    anomaly, then the searching planner threads both goals at once. Both results
    are computed live; nothing is asserted."""
    from genai.agent import show_turn
    acts = blocks_actions(SUSSMAN_BLOCKS)
    want = " and ".join(_phrase_goal(g) for g in SUSSMAN_GOAL)
    show_turn("GOAL", f"build the tower {want} (C starts on A)")
    g = greedy_plan(SUSSMAN_INIT, list(order), acts)
    won = ", ".join(_phrase_goal(a) for a in g["won"]) or "nothing"
    show_turn("GREEDY", f"got {_phrase_goal(g['stuck_on'])} blocked after winning "
              f"{won}: no move reaches it without undoing that")
    plan = plan_search(SUSSMAN_INIT, SUSSMAN_GOAL, acts)
    show_turn("PLANNER", f"{len(plan)} steps, both goals at once: "
              + " -> ".join(plan))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Replanning: a plan is a guess about a world that holds still.             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Everything above is open loop: plan once, replay once, act. But the world moves
# while you act. The same replay that VERIFIES a plan before you start is the
# MONITOR that catches the world drifting from it mid-route: an action whose
# preconditions no longer hold is exactly what ``verify_plan`` was built to flag.
# Catch the divergence, replan from where you actually are, and keep going.

def run_executor(init=None, goal=None, closure=("cafe", "library")):
    """Execute the plan against a world that changes underfoot. The named road
    closes the moment the courier reaches it; the monitor (the same precondition
    replay as ``verify_plan``) catches the now-illegal step, and the planner
    replans from the true current state. Returns a transcript of what happened."""
    init = CANON_INIT if init is None else frozenset(init)
    goal = CANON_GOAL if goal is None else frozenset(goal)
    plan, world, log = plan_bfs(init, goal), init, []
    log.append(("PLAN", list(plan)))
    blocked = f"move({closure[0]}, {closure[1]})"
    i, closed = 0, False
    while not goal <= world and i < len(plan):
        act = plan[i]
        if not closed and act == blocked and ("at", closure[0]) in world:
            world -= {("road", closure[0], closure[1]),
                      ("road", closure[1], closure[0])}      # the road shuts
            closed = True
            log.append(("CLOSED", closure))
        world2, err = _apply(world, act)
        if err:                                              # the monitor fires
            log.append(("MONITOR", (act, err)))
            plan, i = plan_bfs(world, goal), 0
            log.append(("REPLAN", list(plan)))
            continue
        world, i = world2, i + 1
        log.append(("DO", act))
    return {"log": log, "ok": goal <= world,
            "steps": sum(t == "DO" for t, _ in log)}


def show_replan(closure=("cafe", "library")):
    """ACT: the courier sets out on a valid plan, a road closes mid-errand, the
    monitor catches the dead step, and a replan from the current spot still
    delivers all three. Computed live from the true town; only the closure is set."""
    from genai.agent import show_turn
    run = run_executor(closure=closure)
    log = run["log"]
    plan0 = next(v for t, v in log if t == "PLAN")
    show_turn("PLANNER", f"a {len(plan0)}-step plan: " + " -> ".join(plan0[:4])
              + " -> ...")
    before = [v for t, v in log[:[t for t, _ in log].index("CLOSED")]
              if t == "DO"]
    show_turn("COURIER", f"runs {len(before)} steps to {closure[0]}: "
              + " -> ".join(before))
    show_turn("EVENT", f"the {closure[0]}-{closure[1]} road closes")
    act, why = next(v for t, v in log if t == "MONITOR")
    show_turn("MONITOR", f"{act} can't run, {why}")
    detour = next(v for t, v in log if t == "REPLAN")
    show_turn("REPLAN", f"from {closure[0]}, a new {len(detour)}-step route: "
              + " -> ".join(detour[:3]) + " -> ...")
    ok = "all three delivered" if run["ok"] else "errand unfinished"
    show_turn("CHECK", f"valid: {ok}, {run['steps']} steps walked in all")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  The goal it couldn't reach: the proof of no plan a model can't give.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Cut the roads to the library and the book has a destination no sequence of
# moves can reach. The planner searches forward, exhausts the whole finite
# reachable space, finds no state with the book delivered, and returns None: a
# proof of impossibility, not a timeout, because search_size shows it hit no cap.
# The model, asked for "the moves that deliver it", can't answer "there are none";
# it hands back a legal sequence anyway that quietly drops the book at the nearest
# reachable place. Captured by scripts/_plan_impossible_probe.py; the planner side
# is deterministic.

STRANDED_SCENARIO = (
    "A courier starts at home, empty-handed. Roads connect home-cafe and cafe-post "
    "(each runs both ways). The book is at home and must be returned to the "
    "library. Give the sequence of moves that delivers it.")

STRANDED_INIT = frozenset({
    ("at", "home"), ("hand-empty",), ("parcel-at", "book", "home"),
    ("road", "home", "cafe"), ("road", "cafe", "home"),
    ("road", "cafe", "post"), ("road", "post", "cafe")})     # no road reaches library
STRANDED_GOAL = frozenset({("parcel-at", "book", "library")})

# gpt-oss:20b's REAL plan for the impossible errand, identical across three runs:
# unable to reach the library, it delivers the book to the post office (as far as
# it can go) and stops, because it has no verdict for "the library can't be
# reached". Captured verbatim.
IMPOSSIBLE_DEMO = {
    "model": "gpt-oss:20b",
    "plan": ["pick(book, home)", "move(home, cafe)", "move(cafe, post)",
             "drop(book, post)"],
}


def show_impossible(demo=None):
    """The errand with no solution. The model hands back a legal sequence anyway
    (real, baked) that delivers the book to the wrong reachable place, while the
    planner exhausts the finite state space and proves no plan exists. Plan replay
    and the search census are computed live; only the model's plan is baked."""
    from genai.agent import show_code, show_turn
    demo = demo or IMPOSSIBLE_DEMO
    plan = demo["plan"]
    state = STRANDED_INIT
    for act in plan:                        # replay to find where the book ends up
        state, err = _apply(state, act)
        if err:
            break
    book_at = next((a[2] for a in state if a[0] == "parcel-at" and a[1] == "book"), "?")
    census = search_size(STRANDED_INIT, STRANDED_GOAL, _grounded_actions(STRANDED_INIT))
    show_turn("you", STRANDED_SCENARIO)
    show_code(demo["model"], "\n".join(plan))
    show_turn("CHECK", f"every step runs, but the goal is unmet: the book ends at "
                       f"the {book_at}, never the library")
    show_turn("PLANNER", f"no plan exists: it explored every reachable state "
                         f"({census['states']}) and none delivers the book, so this "
                         "is a proof, not a search that gave up")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  The spec shortfall: the one failure the verifier can't catch.                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# verify_plan answers "does this plan reach the stated goal?" It cannot answer
# "is the stated goal what you meant?" Here the errand carries an ORDERING wish
# (the beans are perishable, bring them home first) that a STRIPS goal, a flat set
# of end-states, simply cannot hold. So the model's translation drops it, the
# planner returns a plan that is genuinely valid for the goal it was handed, and
# the plan still delivers the beans third. A second critic, back-translating the
# goal to English, makes the omission visible but still can't judge whether it
# matters: that part is on you.

ORDERED_SCENARIO = (SCENARIO + " The coffee beans are perishable, so they must "
                    "be brought home before either other parcel is delivered.")


def beans_first(plan):
    """Does ``plan`` honour the perishable-beans wish, dropping the beans home
    before the letter or book is delivered? Returns the verdict and the 1-based
    delivery position of each parcel, computed straight off the plan."""
    pos = {p: i + 1 for i, s in enumerate(plan)
           for p in ("beans", "book", "letter") if s == f"drop({p}, "
           f"{'home' if p == 'beans' else 'library' if p == 'book' else 'post'})"}
    beans = pos.get("beans", 99)
    return {"ok": beans < pos.get("book", 0) and beans < pos.get("letter", 0),
            "beans_pos": beans, "of": len(plan), "order": pos}


def backtranslate(pddl: str, model: str = EIS_MODEL, n_predict: int = 1200) -> str:
    """Read the model's PDDL goal back into plain English, the second critic. A
    swapped destination would surface here as words that don't match the errand;
    a wish the goal never recorded leaves nothing to read back."""
    goal = _section(pddl, "(:goal")
    prompt = ("Here is the goal of a delivery planning problem:\n\n" + goal +
              "\n\nIn one plain-English sentence, state exactly the end situation "
              "this goal requires. Mention only what the goal says. Do not add "
              "timing, order, or anything not written above.")
    text = _emit(model, prompt, n_predict).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), text)
    return line[:200]


# Captured once by scripts/_spec_probe.py (gpt-oss:20b), baked for the page. The
# PROBLEM is the model's real translation of the ordered errand; its :goal holds
# the three destinations and, necessarily, nothing about order. The READBACK is
# the model's real back-translation of that goal. The plan, the validity verdict,
# and the order check are all recomputed live, so only the translation is baked.
SPEC_DEMO = {
    "problem": """(define (problem errand)
  (:domain errands)

  (:objects
    home cafe post library letter book beans)

  (:init
    ; Courier starts at home and has an empty hand
    (at home) (hand-empty)

    ; Roads (bidirectional)
    (road home cafe) (road cafe home)
    (road cafe post) (road post cafe)
    (road cafe library) (road library cafe)
    (road post library) (road library post)

    ; Initial parcel locations
    (parcel-at letter home)
    (parcel-at book home)
    (parcel-at beans cafe))

  (:goal (and
           (parcel-at letter post)
           (parcel-at book library)
           (parcel-at beans home))))""",
    "readback": "The letter is at the post, the book is at the library, and the "
                "beans are at home.",
}


def show_spec_gap(demo=None):
    """The plan that passes and still fails: the planner returns a valid plan that
    delivers the perishable beans third, and the verifier is right to call it
    valid, because the wish the model couldn't formalize was never in the goal."""
    from genai.agent import show_turn
    demo = demo or SPEC_DEMO
    problem = demo.get("problem") or PLAN_DEMO["problem"]
    init, goal = parse_problem(problem)
    plan = plan_bfs(init, goal)
    ok, _, reason = verify_plan(init, goal, plan)
    order = beans_first(plan)
    show_turn("you", "Mail the letter, return the book, bring the beans home. The "
              "beans are perishable: bring them home before either other delivery.")
    show_turn("PLANNER", f"a valid {len(plan)}-step plan; "
              f"delivers the beans {order['beans_pos']}th of {len(plan)}")
    show_turn("CHECK", "valid: " + reason)
    show_turn("INTENT", "but 'beans first' is " +
              ("kept" if order["ok"] else "broken") +
              ": the goal never said it, so the planner couldn't honour it")
    readback = demo.get("readback")
    if readback:
        show_turn("READ-BACK", f"the goal in English: '{readback}' "
                  "(destinations, no order)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Who writes the rulebook? Letting the model write the DOMAIN, not the      ║
# ║  problem. The safe half was the problem; the domain is where a slip hides. ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# So far the model only ever wrote the PROBLEM, and the verifier replayed it
# against the fixed, hand-checked rulebook. Hand the model the rulebook too and
# the mistranslation moves somewhere the verifier can't see it: a dropped
# precondition or a missing delete-effect parses cleanly, the planner obeys it,
# and the plan is "valid" against a rulebook that lets the courier hold three
# parcels at once. To show that we have to ground an ARBITRARY model-written
# domain, not just the courier's, so here is a small general STRIPS grounder.

def _action_blocks(domain: str):
    """Every balanced ``(:action ...)`` span in a PDDL domain string."""
    blocks, i = [], 0
    while True:
        i = domain.find("(:action", i)
        if i < 0:
            return blocks
        depth = 0
        for j in range(i, len(domain)):
            depth += (domain[j] == "(") - (domain[j] == ")")
            if depth == 0:
                blocks.append(domain[i:j + 1])
                i = j + 1
                break
        else:
            return blocks


def _kw_group(block: str, kw: str) -> str:
    """The balanced paren group following a bare PDDL keyword like ``:effect``.
    Domain keywords aren't self-wrapped (``:precondition (and ...)``), so we find
    the keyword and capture the next ``(...)``."""
    i = block.find(kw)
    if i < 0:
        return ""
    j = block.find("(", i)
    if j < 0:
        return ""
    depth = 0
    for k in range(j, len(block)):
        depth += (block[k] == "(") - (block[k] == ")")
        if depth == 0:
            return block[j:k + 1]
    return ""


def _effects(block: str):
    """(add, del) atom sets from an :effect, reading ``(not (...))`` as a delete."""
    eff = _kw_group(block, ":effect")
    dele = {tuple(x.split()) for x in re.findall(r"\(not\s*\(([^()]+)\)\s*\)", eff)}
    allatoms = {tuple(x.split()) for x in re.findall(r"\(([^()]+)\)", eff)
                if x.split() and x.split()[0] not in ("and", "not")}
    return allatoms - dele, dele


def parse_domain(domain: str):
    """A model-written domain as action schemas ``(name, params, pre, add, del)``,
    each atom still carrying its ``?variables``. Positive-precondition STRIPS, which
    is all the courier and blocks domains use."""
    schemas = []
    for block in _action_blocks(domain):
        name = block[len("(:action"):].split()[0]
        params = re.findall(r"\?[\w-]+", _kw_group(block, ":parameters"))
        pre = [tuple(x.split()) for x in
               re.findall(r"\(([^()]+)\)", _kw_group(block, ":precondition"))
               if x.split() and x.split()[0] not in ("and", "not")]
        add, dele = _effects(block)
        schemas.append((name, params, pre, add, dele))
    return schemas


def ground_domain(schemas, objects):
    """Instantiate every schema over every tuple of ``objects`` into the
    ``(name, pre, add, del)`` moves ``plan_search`` consumes. Nonsense bindings
    survive but never fire, because their preconditions can't hold."""
    from itertools import product
    ground = []
    for name, params, pre, add, dele in schemas:
        for combo in product(objects, repeat=len(params)):
            sub = dict(zip(params, combo))
            g = lambda atom: tuple(sub.get(t, t) for t in atom)   # noqa: E731
            ground.append((f"{name}({', '.join(combo)})",
                           {g(a) for a in pre}, {g(a) for a in add},
                           {g(a) for a in dele}))
    return ground


def plan_with_domain(domain: str, init=None, goal=None):
    """Plan the canonical errand using a MODEL-WRITTEN domain, then replay the
    result against the TRUE rules. Returns the plan plus the honest verdict: a
    plan the model's rulebook certified but the real world rejects is the tell
    that the bug is in the rulebook the verifier trusted."""
    init = CANON_INIT if init is None else frozenset(init)
    goal = CANON_GOAL if goal is None else frozenset(goal)
    locs, parcels = _objects(init)
    ground = ground_domain(parse_domain(domain), locs + parcels)
    plan = plan_search(init, goal, ground)
    if plan is None:
        return {"plan": None, "model_steps": None, "sound": False,
                "true_ok": False, "true_reason": "the model's domain has no plan"}
    true_ok, idx, reason = verify_plan(init, goal, plan)   # the real rulebook
    return {"plan": plan, "model_steps": len(plan), "sound": true_ok,
            "true_ok": true_ok, "broke_at": idx, "true_reason": reason}


def formalize_domain(model: str = EIS_MODEL, n_predict: int = 2000) -> str:
    """Ask the model to write the courier DOMAIN (the rulebook) from the rules in
    words, the riskier half of the translation. Returns the PDDL domain it emits."""
    prompt = (
        "Write a PDDL domain named 'errands' for a courier world. Predicates: "
        "(at ?l) where the courier is; (road ?a ?b) a one-way link; (parcel-at "
        "?p ?l); (holding ?p); (hand-empty).\n\n"
        "Three actions, with correct preconditions AND effects:\n"
        "- move ?from ?to: the courier walks a road. Needs to be at ?from with a "
        "road to ?to; ends at ?to and no longer at ?from.\n"
        "- pick ?p ?l: lift a parcel. Needs to be at ?l where the parcel is, with "
        "an empty hand; ends holding it, hand no longer empty, parcel no longer "
        "lying there.\n"
        "- drop ?p ?l: set a parcel down. Needs to be at ?l holding it; ends with "
        "the parcel there, hand empty, no longer holding it.\n\n"
        "The courier carries only one parcel at a time. Output only the PDDL "
        "domain, starting with (define (domain errands).")
    text = _emit(model, prompt, n_predict)
    fenced = re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start = body.find("(define")
    return (body[start:] if start >= 0 else body).strip()


# Captured by scripts/_domain_probe.py (gpt-oss:20b): the model's REAL courier
# domain, and it's correct, a clean 12-step plan that survives the true rules. The
# point isn't that the model slipped (it didn't, nor did gemma4); it's that the
# verifier couldn't have told if it had. We show that by breaking one line live.
DOMAIN_DEMO = {
    "model": "gpt-oss:20b",
    "domain": """(define (domain errands)
  (:requirements :strips)
  (:predicates
    (at ?l)              ; the courier is at location ?l
    (road ?a ?b)         ; a road from ?a to ?b
    (parcel-at ?p ?l)    ; parcel ?p lies at location ?l
    (holding ?p)         ; courier is holding parcel ?p
    (hand-empty))        ; the courier's hand is empty
  (:action move
    :parameters (?from ?to)
    :precondition (and (at ?from)
                       (road ?from ?to))
    :effect (and (not (at ?from))
                 (at ?to)))
  (:action pick
    :parameters (?p ?l)
    :precondition (and (at ?l)
                       (parcel-at ?p ?l)
                       (hand-empty))
    :effect (and (not (parcel-at ?p ?l))
                 (not (hand-empty))
                 (holding ?p)))
  (:action drop
    :parameters (?p ?l)
    :precondition (and (at ?l)
                       (holding ?p))
    :effect (and (parcel-at ?p ?l)
                 (hand-empty)
                 (not (holding ?p)))))""",
}


def _drop_precondition(schemas, action, atom):
    """Return ``schemas`` with one precondition atom struck from one action: the
    one-line slip that turns a correct rulebook into a wrong one the verifier
    still trusts."""
    return [(n, p, [a for a in pre if a != atom] if n == action else pre, ad, de)
            for n, p, pre, ad, de in schemas]


def _action_text(domain, action):
    """The model's own text for one ``(:action ...)`` block, to show on the page
    rather than the whole 20-line rulebook."""
    for block in _action_blocks(domain):
        if block[len("(:action"):].split()[0] == action:
            return block
    return ""


def show_domain_demo(demo=None):
    """The rulebook the verifier trusts. The model's real domain is correct, a clean
    12-step plan that survives the true rules. Then strike one precondition and the
    planner certifies a courier holding every parcel at once, a plan valid against a
    broken rulebook, because the verifier replays against the domain it was given."""
    from genai.agent import show_code, show_turn
    demo = demo or DOMAIN_DEMO
    locs, parcels = _objects(CANON_INIT)
    schemas = parse_domain(demo["domain"])
    good = plan_search(CANON_INIT, CANON_GOAL, ground_domain(schemas, locs + parcels))
    good_ok = verify_plan(CANON_INIT, CANON_GOAL, good)[0]
    show_code(demo["model"], _action_text(demo["domain"], "pick"))
    show_turn("REPLAY", f"the full domain is sound: a {len(good)}-step plan the "
              f"real rules accept ({'valid' if good_ok else 'invalid'})")
    broken = _drop_precondition(schemas, "pick", ("hand-empty",))
    bad = plan_search(CANON_INIT, CANON_GOAL, ground_domain(broken, locs + parcels))
    ok, idx, why = verify_plan(CANON_INIT, CANON_GOAL, bad)
    show_turn("BREAK", "delete one line, pick's (hand-empty) precondition")
    show_turn("PLANNER", f"obeys the broken rulebook: {len(bad)} steps, picking "
              f"parcels with a hand that's already full")
    step = bad[idx] if idx < len(bad) else "the end"
    show_turn("REPLAY", f"the real rules reject it at step {idx + 1}, {step}: {why}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Fails loud, fails silent: the planner has a wall too, but a visible one.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The model's wall in the first section was silent: it kept narrating confident
# routes as the errand grew, and only a replay showed they didn't run. The
# planner's wall is the opposite. Search cost grows fast with the number of
# parcels, and when it passes the ``max_states`` ceiling the planner stops and
# says so, returning no plan rather than a wrong one. A wall you can see beats a
# wall you can't, which is the whole reason to push the hard part onto search.

def scaling_town(k: int, spokes=("a", "b", "c", "d")):
    """A courier town with ``k`` parcels, all at the hub, each due at a spoke. More
    parcels means a combinatorially larger state space, the dial that walks the
    planner toward its wall."""
    init = {("at", "home"), ("hand-empty",)}
    for s in spokes:
        init |= {("road", "home", s), ("road", s, "home")}
    for a, b in zip(spokes, spokes[1:] + spokes[:1]):    # a ring between spokes
        init |= {("road", a, b), ("road", b, a)}
    goal = set()
    for i in range(k):
        p = f"p{i}"
        init.add(("parcel-at", p, "home"))
        goal.add(("parcel-at", p, spokes[i % len(spokes)]))
    return frozenset(init), frozenset(goal)


def scaling_series(ks=range(1, 9), max_states: int = 60000):
    """Walk the planner up the parcel count and record how many states each search
    burns and whether it hit the ceiling. The point of the curve: the failure is
    loud (it announces the ceiling), never a silent wrong answer."""
    out = []
    for k in ks:
        init, goal = scaling_town(k)
        s = search_size(init, goal, _grounded_actions(init), max_states)
        out.append({"parcels": k, **s})
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Valid isn't optimal: a road can be legal and still be the wrong road.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# "Shortest plan" quietly meant fewest steps. Put a price on the roads and fewest
# steps stops meaning cheapest: a one-hop highway beats a two-hop pair of streets
# on step count and loses on mileage. Both plans are valid, both deliver the
# letter; they differ five-fold in cost. The planner optimizes whatever metric you
# hand it (and will prove it optimal); the model optimized nothing, it just
# narrated a route. Validity was never the same question as quality.
COST_INIT = frozenset({
    ("at", "home"), ("hand-empty",), ("parcel-at", "letter", "home"),
    ("road", "home", "post"), ("road", "post", "home"),        # the highway
    ("road", "home", "cafe"), ("road", "cafe", "home"),        # two short streets
    ("road", "cafe", "post"), ("road", "post", "cafe")})
COST_GOAL = frozenset({("parcel-at", "letter", "post")})
COST_MILES = {"move(home, post)": 10, "move(post, home)": 10,
              "move(home, cafe)": 1, "move(cafe, home)": 1,
              "move(cafe, post)": 1, "move(post, cafe)": 1}


def plan_miles(plan):
    """Total road mileage of a courier plan under ``COST_MILES`` (picks/drops are
    free; only moves cost)."""
    return sum(COST_MILES.get(s, 0) for s in plan)


def show_cost():
    """Two valid plans for one delivery: the fewest-steps route takes the highway,
    the cheapest route takes the streets, and they differ five-fold in miles. Both
    computed live, one by step-count search and one by cost search."""
    from genai.agent import show_turn
    acts = _grounded_actions(COST_INIT)
    short = plan_search(COST_INIT, COST_GOAL, acts)
    cheap, miles = plan_ucs(COST_INIT, COST_GOAL, acts, COST_MILES)
    show_turn("you", "Take the letter home to the post office. The direct road is "
              "a 10-mile highway; home-cafe-post is two 1-mile streets.")
    short_mi, cheap_mi = plan_miles(short), plan_miles(cheap)
    show_turn("FEWEST STEPS", f"{len(short)} steps, {short_mi} miles: "
              + " -> ".join(short))
    show_turn("CHEAPEST", f"{len(cheap)} steps, {cheap_mi} miles: "
              + " -> ".join(cheap))
    show_turn("CHECK", f"both valid; fewest steps costs {short_mi // cheap_mi}x the "
              "miles. 'Shortest' wasn't 'best' until you said which you meant.")


# ── A plan you can explain: causal links read off the trace ───────────────────
# A verified plan tells you *that* the goal is reached, not *why* each step is
# there. The causal link fills that in: for every action, which earlier step (or
# the start) made each of its preconditions true. Read straight off the replayed
# trace, the annotation can't be wrong. Ask a model to narrate the same
# dependencies over a twelve-step plan and it invents a link the trace never had.
# The annotator is deterministic; the narration is gpt-oss:20b, captured once by
# scripts/_plan_causal_probe.py.

_DYNAMIC = ("at", "holding", "hand-empty", "parcel-at")  # the non-static predicates


def causal_links(init, plan):
    """For each 1-based step, which earlier step (or 0 for the start) established
    each of its dynamic preconditions. Returns a list of dicts with ``step``,
    ``action``, ``needs`` (a list of ``(phrase, supporter)``), and ``supporters``
    (the set of supporting step numbers). Deterministic, read off the replay."""
    acts = {name: (pre, add, rem) for name, pre, add, rem in _grounded_actions(init)}
    established = {atom: 0 for atom in init}          # atom -> step that made it true
    links = []
    for i, name in enumerate(plan, 1):
        pre, add, rem = acts[name]
        needs = sorted(((_phrase(p), established.get(p, 0)) for p in pre
                        if p[0] in _DYNAMIC), key=lambda t: t[1])
        links.append({"step": i, "action": name, "needs": needs,
                      "supporters": {s for _, s in needs}})
        for atom in rem:
            established.pop(atom, None)
        for atom in add:
            established[atom] = i
    return links


def _cite(step):
    """A supporter step as a citation: a step number, or 'start'."""
    return "start" if step == 0 else f"step {step}"


NARRATE_MODEL = "gemma4:latest"  # a plain instruction-follower for the line format


def narrate_dependencies(plan, model=NARRATE_MODEL):
    """Ask a model, for each numbered step, which earlier steps (or the start) its
    preconditions depend on. Returns ``{step: set_of_supporters}`` parsed from the
    reply (0 means the start)."""
    numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(plan, 1))
    prompt = ("Here is a courier's verified plan. A step depends on an earlier step "
              "when that earlier step made one of its preconditions true (moved the "
              "courier into place, or picked up the parcel it drops). For each step, "
              "list the earlier step numbers it depends on, or 'start' if the "
              "initial situation alone suffices. Reply with one line per step in the "
              "form  'N: a, b'  or  'N: start', nothing else.\n\n" + numbered)
    msg = ollama.chat(model=model, think=False, keep_alive="10m",
                      options={"num_predict": 1500},
                      messages=[{"role": "user", "content": prompt}])["message"]
    reply = (msg.get("content") or "").strip()
    claimed = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.\)]\s*(.+)", line)
        if not m:
            continue
        step = int(m.group(1))
        supporters = set()
        for tok in re.findall(r"start|\d+", m.group(2).lower()):
            supporters.add(0 if tok == "start" else int(tok))
        claimed[step] = supporters
    return claimed, reply


def grade_narration(links, claimed):
    """Compare a model's claimed dependencies against the trace-derived ones.
    Returns ``(correct, total, invented)`` where ``correct`` counts steps whose
    supporter set matches exactly, and ``invented`` lists the steps where the
    model cited an earlier step the trace has no causal link to, each as
    ``(step, action, invented_set, truth_set)``. The annotator's own links are all
    real, so it never appears in ``invented``."""
    correct, invented = 0, []
    for link in links:
        truth = link["supporters"]
        got = claimed.get(link["step"], set())
        if got == truth:
            correct += 1
        extra = got - truth
        if extra:
            invented.append((link["step"], link["action"], extra, truth))
    return correct, len(links), invented


# Captured by scripts/_plan_causal_probe.py (gemma4:latest on the 12-step plan).
# The model matches the trace on 10 of 12 steps, but on step 6 it cites the
# courier's first cafe visit (step 2) when the pickup was actually enabled by the
# return trip (step 5) and the hand freed by the earlier drop (step 4): a link the
# trace never had. The invention reproduces across runs.
CAUSAL_DEMO = {
    "plan": ["pick(book, home)", "move(home, cafe)", "move(cafe, library)",
             "drop(book, library)", "move(library, cafe)", "pick(beans, cafe)",
             "move(cafe, home)", "drop(beans, home)", "pick(letter, home)",
             "move(home, cafe)", "move(cafe, post)", "drop(letter, post)"],
    "claimed": {"1": [0], "2": [0], "3": [2], "4": [1, 3], "5": [3], "6": [2],
                "7": [5], "8": [6, 7], "9": [0], "10": [7], "11": [10],
                "12": [9, 11]},
}


def show_causal_plan(demo=None):
    """The annotator's causal chain for the whole plan, then the model's narration
    scored against it: the annotator is right by construction, the model invents a
    link the trace never had."""
    from genai.agent import show_turn
    demo = demo or CAUSAL_DEMO
    plan = demo["plan"]
    claimed = {int(k): set(v) for k, v in demo["claimed"].items()}
    links = causal_links(CANON_INIT, plan)
    show_turn("ANNOTATOR", "each step, and what earlier step makes it possible:")
    for link in links:
        cites = ", ".join(f"{ph} from {_cite(s)}" for ph, s in link["needs"])
        show_turn(f"step {link['step']}", f"{link['action']}  <-  {cites}")
    correct, total, invented = grade_narration(links, claimed)
    cites = "cites" if len(invented) == 1 else "cite"
    show_turn("gemma4", f"narrated the same {total} steps: {correct} match the trace "
              f"exactly, and {len(invented)} {cites} a step the trace never links to")
    for step, action, extra, truth in invented:
        cited = ", ".join(_cite(s) for s in sorted(extra))
        real = ", ".join(_cite(s) for s in sorted(truth))
        show_turn("INVENTED", f"step {step} {action}: model says it depends on "
                  f"{cited}, but the trace shows {real}")

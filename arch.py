"""Who owns control flow, the model or the code?

Two ways to total the same cafe order. In the *model-driven* version the model
owns the loop: it prices each item with a tool and does the arithmetic
(quantities, discount) itself, reporting the total. In the *code-driven* version
the code owns the loop: it asks the model for one bounded judgment (read the
order back) and then computes the total deterministically. The accuracy difference
between them is Chapter 1's measurement, and the lesson is to give the model the
smallest decision that still needs judgment.

Both modes hand their answer back through a tool call, never as free text, so a
reasoning model that buries its prose in a hidden thinking channel can't confuse
the score. The model is gpt-oss:20b (EIS_MODEL); gold totals are computed from
MENU, never typed. Nondeterministic (LLM sampling + wall clock): capture
CONTROL_STUDY once and freeze the notebook cell.

The same cafe world then carries the rest of the chapter: metered variants of
the two registers read the quality attributes (reliability, cost, consistency)
off repeated runs, and the receipt-reading task stages the three design
patterns, validator, cascade, and checker. Every study is captured once by its
probe script (scripts/_arch_quality_probe.py, scripts/_arch_patterns_probe.py)
and baked as a constant, so the chapter's cells are deterministic.
"""
import json
import re
import time
from statistics import mean

import ollama

from genai.agent import run_with_trace, show_turn, tool_spec
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)

EIS_MODEL = "gpt-oss:20b"

MENU = {"latte": 4.50, "rockstar": 3.25, "tea": 2.75,
        "redbull": 3.75, "monster": 2.25, "espresso": 3.00}

# (spoken order, {item: qty}, discount %). Gold totals are computed below.
ORDERS = [
    ("Three lattes, two Rockstars, and a tea, please.",
     {"latte": 3, "rockstar": 2, "tea": 1}, 0),
    ("Two Red Bulls, a Monster, and three espressos, with the 15% member discount.",
     {"redbull": 2, "monster": 1, "espresso": 3}, 15),
    ("Four teas and two lattes to go.",
     {"tea": 4, "latte": 2}, 0),
    ("A Rockstar, a Red Bull, and two Monsters, take 10% off for the loyalty card.",
     {"rockstar": 1, "redbull": 1, "monster": 2}, 10),
    ("Three espressos, two lattes, and a Rockstar, minus the 20% staff discount.",
     {"espresso": 3, "latte": 2, "rockstar": 1}, 20),
]


def gold_total(items: dict, discount: int) -> float:
    """The true total for an order, computed from MENU (never hand-typed)."""
    subtotal = sum(MENU[k] * q for k, q in items.items())
    return round(subtotal * (1 - discount / 100), 2)


def price(item: str) -> float:
    """Price tool: one menu item in dollars. Tolerates the spellings a spoken
    order arrives in: spacing ("Red Bull"), and a trailing plural ("Monsters").
    The plural is only stripped when the name isn't already on the menu, so a
    menu name that ends in s would survive intact."""
    key = re.sub(r"[\s_-]", "", item.lower().strip())
    if key not in MENU and key.endswith("s"):
        key = key[:-1]
    return MENU.get(key, 0.0)


# --- Mode A: the model owns the loop -------------------------------------------

def run_model_driven(question: str, model: str = EIS_MODEL) -> tuple:
    """The model prices each item and does the arithmetic, reporting via a tool."""
    box = {}
    handlers = {"price": price,
                "submit_total": lambda amount: box.update(total=round(float(amount), 2))}
    tools = [tool_spec("price", "Look up the price of one menu item in dollars.",
                       item="string"),
             tool_spec("submit_total", "Report the final order total in dollars.",
                       amount="number")]
    prompt = ("You are a cafe register. Price each item with the price tool, work "
              "out the order's total in dollars after any discount, then call "
              f"submit_total with the final amount.\nOrder: {question}")
    t0 = time.time()
    _, calls = run_with_trace(prompt, tools, handlers, model=model,
                              max_tokens=900, max_iter=14, think=True)
    return box.get("total"), time.time() - t0, len(calls)


# --- Mode B: the code owns the loop --------------------------------------------

def run_code_driven(question: str, model: str = EIS_MODEL) -> tuple:
    """The model reads the order into one tool call; code prices and totals it."""
    box = {}

    def submit_order(discount=0, **counts):
        items = {k: int(v) for k, v in counts.items() if k in MENU and int(v) > 0}
        box["order"] = (items, int(discount))
        return "recorded"

    params = {it: "integer" for it in MENU}
    params["discount"] = "integer"
    tools = [tool_spec("submit_order",
                       "Record the order: a count for each menu item (0 if not "
                       "ordered) and the percent discount.", **params)]
    prompt = ("Read this cafe order and record it with the submit_order tool, "
              f"giving a count for every menu item and the discount.\nOrder: {question}")
    t0 = time.time()
    run_with_trace(prompt, tools, {"submit_order": submit_order},
                   model=model, max_tokens=400, max_iter=4)
    secs = time.time() - t0
    if "order" not in box:
        return None, secs
    return gold_total(*box["order"]), secs


# --- The study -----------------------------------------------------------------

def control_study(model: str = EIS_MODEL, trials: int = 3) -> dict:
    """Score both ownership patterns on every order, ``trials`` times each."""
    out = {"model_driven": {"hit": 0, "n": 0, "secs": []},
           "code_driven": {"hit": 0, "n": 0, "secs": []}}
    for question, items, disc in ORDERS:
        gold = gold_total(items, disc)
        for _ in range(trials):
            tot, secs, _ = run_model_driven(question, model)
            out["model_driven"]["n"] += 1
            out["model_driven"]["secs"].append(secs)
            out["model_driven"]["hit"] += int(tot is not None and abs(tot - gold) < 0.01)

            tot2, secs2 = run_code_driven(question, model)
            out["code_driven"]["n"] += 1
            out["code_driven"]["secs"].append(secs2)
            out["code_driven"]["hit"] += int(tot2 is not None and abs(tot2 - gold) < 0.01)
    return {k: {"accuracy": round(v["hit"] / v["n"], 3),
                "mean_secs": round(mean(v["secs"]), 2),
                "n": v["n"]}
            for k, v in out.items()}


# Captured by scripts/_control_ownership_probe.py (gpt-oss:20b, trials=3, n=15).
CONTROL_STUDY = {
    "model_driven": {"accuracy": 0.4, "mean_secs": 38.46, "n": 15},
    "code_driven": {"accuracy": 0.933, "mean_secs": 8.77, "n": 15},
}


# --- Where does state live? ------------------------------------------------------
# A three-turn cafe order that revises itself: the customer adds items, then
# changes an earlier one, so the last turn only lands if the first is still on the
# books. In the *model-owned* version the running order lives in the chat history
# and the model has to carry it across turns; in the *code-owned* version the loop
# keeps the order in a dict, re-states it every turn, and asks the model for only
# that one turn's change. Same model, same order; the only difference is where the
# running state lives. Nondeterministic (LLM sampling): capture STATE_STUDY and
# the two demos once with scripts/_arch_state_probe.py and freeze the cell.

STATE_DIALOGUE = [
    "I'll have two lattes.",
    "Add a Rockstar and a tea.",
    "Actually, make that three lattes.",
    "Scratch the tea, give me two Monsters instead.",
    "And a Red Bull for my friend.",
    "One more thing: turn the Rockstar into a second Red Bull. That's everything.",
]
# The order after all six turns: turn three bumps the lattes, turn four drops the
# tea for Monsters, turn six swaps the Rockstar for a second Red Bull. Every turn
# edits an earlier one, so anything that forgets a revision lands somewhere else.
STATE_GOLD = ({"latte": 3, "monster": 2, "redbull": 2}, 0)
# The house model, competent enough that the one variable is where the state
# lives. A 2B model can't be used here: on a menu of three energy drinks it
# co-activates them (asked for two Monsters it also invents Red Bulls), so its
# failures measure reading ability rather than state-keeping.
STATE_MODEL = "gemma4:latest"


def _receipt_to_order(receipt) -> tuple:
    """Turn a parsed JSON receipt into an (items, discount) order, or None if it
    lists no items. Tolerates the junk a weak model emits (floats, strings)."""
    if not isinstance(receipt, dict):
        return None
    items = {k: int(v) for k, v in receipt.items()
             if k in MENU and isinstance(v, int) and not isinstance(v, bool) and v > 0}
    disc = receipt.get("discount", 0)
    disc = disc if isinstance(disc, int) and not isinstance(disc, bool) else 0
    return (items, disc) if items else None


def _order_key(res) -> str:
    """A hashable fingerprint of an order, for counting distinct final answers."""
    if res is None:
        return "none"
    items, disc = res
    return json.dumps([sorted(items.items()), disc])


def _order_ok(res, gold=STATE_GOLD) -> bool:
    """True when a final order matches the gold order (items and discount)."""
    return res is not None and dict(res[0]) == gold[0] and res[1] == gold[1]


_STATE_KEYS = ", ".join(MENU) + ", discount"


def run_state_model_owned(dialogue: list = None, model: str = STATE_MODEL) -> dict:
    """The running order lives only in the chat history. The register hears each
    turn in sequence and confirms the order so far, then reads the final order
    into a JSON receipt. Returns {"order", "turns"} where turns pairs each
    customer line with the model's real running-order confirmation."""
    dialogue = dialogue or STATE_DIALOGUE
    turns = []
    messages = [{"role": "system", "content":
                 "You are a cafe register taking one order over several turns. "
                 "Keep a running order as the customer adds to it or revises it, "
                 "and confirm the full running order back in one short line."}]
    for line in dialogue:
        messages.append({"role": "user", "content": line})
        resp = _client.chat(model=model, messages=messages, think=False,
                            options={"num_predict": 90})
        reply = resp["message"]["content"].strip()
        messages.append({"role": "assistant", "content": reply})
        turns.append((line, reply))
    messages.append({"role": "user", "content":
                     "Now reply with the complete final order as one JSON object, "
                     f"nothing else. Keys: {_STATE_KEYS}. Map each ordered item to "
                     "its quantity and discount to the percent off."})
    resp = _client.chat(model=model, messages=messages, think=False,
                        options={"num_predict": 160})
    order = _receipt_to_order(parse_receipt(resp["message"]["content"]))
    return {"order": order, "turns": turns}


def run_state_code_owned(dialogue: list = None, model: str = STATE_MODEL) -> dict:
    """The loop owns the order. Each turn it re-states the canonical order so far
    and asks the model to apply just this one turn's change, then stores the
    reading back as the new truth. The model never has to remember. Returns
    {"order", "turns"} where turns records (customer line, canonical-before,
    reading-after) for each turn."""
    dialogue = dialogue or STATE_DIALOGUE
    order, discount, turns = {}, 0, []
    for line in dialogue:
        state = {**order, "discount": discount}
        before = receipt_line(state) if order else "empty"
        prompt = (f"Current order as JSON: {json.dumps(state)}\n"
                  f"The customer now says: \"{line}\"\n"
                  "Return the updated order as one JSON object, nothing else. Keep "
                  "every entry the same except what this one change touches. Use "
                  f"the item names ({', '.join(MENU)}) as keys with whole-number "
                  'quantities, and "discount" as the percent off (0 if none).')
        resp = _client.chat(model=model, think=False,
                            messages=[{"role": "user", "content": prompt}],
                            options={"num_predict": 160})
        parsed = _receipt_to_order(parse_receipt(resp["message"]["content"]))
        if parsed is not None:
            order, discount = parsed
        turns.append((line, before, receipt_line(dict(order, discount=discount))))
    return {"order": (order, discount) if order else None, "turns": turns}


def state_study(dialogue: list = None, gold: tuple = STATE_GOLD,
                trials: int = 12, model: str = STATE_MODEL) -> dict:
    """Run the same revising order through both designs ``trials`` times and read
    two attributes off the runs: *reliability* (fraction whose final order is
    correct) and *consistency* (how many distinct final orders the same dialogue
    produced). The trap is turn three, which overrides turn one's latte count."""
    out = {}
    for name, runner in [("model_owned", run_state_model_owned),
                         ("code_owned", run_state_code_owned)]:
        hits, finals = 0, []
        for _ in range(trials):
            res = runner(dialogue, model)["order"]
            hits += int(_order_ok(res, gold))
            finals.append(_order_key(res))
        out[name] = {"reliability": round(hits / trials, 3),
                     "distinct_finals": len(set(finals))}
    return {"trials": trials, "model": model, **out}


# Captured by scripts/_arch_state_probe.py (gemma4:e2b, trials=8). Same model both
# ways: holding the running order in its head, it lands right half the time and
# commits four different final tallies; handed one edit at a time against a
# code-owned order, it is right every run and gives one answer.
STATE_STUDY = {
    "trials": 8,
    "model": "gemma4:latest",
    "model_owned": {"reliability": 0.125, "distinct_finals": 4},
    "code_owned": {"reliability": 0.875, "distinct_finals": 2},
}

# One real run of each design, captured by the same probe. Model rows carry the
# model's real text; STATE and ORDER are event labels for lines code owns. In the
# model-owned run the model loses the thread out loud on the last turn, talks
# itself from three Red Bulls to four, and commits the four: the state was never
# pinned down anywhere it could be checked.
STATE_DEMO_MODEL = {
    "model": "gemma4:latest",
    "order": ({"latte": 3, "redbull": 4, "monster": 2}, 0),
    "turns": [
        ("I'll have two lattes.",
         "Okay, that's two lattes so far. Anything else?"),
        ("Add a Rockstar and a tea.",
         "You'd like to add one Rockstar and one tea. Your current order is: "
         "two lattes, one Rockstar, and one tea. Correct?"),
        ("Actually, make that three lattes.",
         "No problem. Just to confirm, your updated order is: three lattes, "
         "one Rockstar, and one tea. Is that right?"),
        ("Scratch the tea, give me two Monsters instead.",
         "So, we're removing the tea and adding two Monsters. Your running "
         "order now is: three lattes, one Rockstar, and two Monsters. Correct?"),
        ("And a Red Bull for my friend.",
         "Got it. Adding one Red Bull to your order. The final order is: three "
         "lattes, one Rockstar, two Monsters, and one Red Bull. Is that "
         "everything?"),
        ("One more thing: turn the Rockstar into a second Red Bull. "
         "That's everything.",
         "Okay, changing the Rockstar for another Red Bull. Your full running "
         "order is now: three lattes, three Red Bulls, two Monsters, and one "
         "Red Bull. Wait, let me recheck that total count...\n\nJust to "
         "confirm, your final order is: three lattes, four Red Bulls (since "
         "you added one before, then changed the Rockstar into one more), and "
         "two Monsters. Is this correct?"),
    ],
}
STATE_DEMO_CODE = {
    "model": "gemma4:latest",
    "order": ({"latte": 3, "monster": 2, "redbull": 2}, 0),
    "turns": [
        ("I'll have two lattes.", "empty", "empty, discount 0%"),
        ("Add a Rockstar and a tea.", "empty",
         "rockstar x1, tea x1, discount 0%"),
        ("Actually, make that three lattes.",
         "rockstar x1, tea x1, discount 0%",
         "rockstar x1, tea x1, latte x3, discount 0%"),
        ("Scratch the tea, give me two Monsters instead.",
         "rockstar x1, tea x1, latte x3, discount 0%",
         "rockstar x1, latte x3, monster x2, discount 0%"),
        ("And a Red Bull for my friend.",
         "rockstar x1, latte x3, monster x2, discount 0%",
         "rockstar x1, latte x3, monster x2, redbull x1, discount 0%"),
        ("One more thing: turn the Rockstar into a second Red Bull. "
         "That's everything.",
         "rockstar x1, latte x3, monster x2, redbull x1, discount 0%",
         "latte x3, monster x2, redbull x2, discount 0%"),
    ],
}


def _no_zero_discount(line: str) -> str:
    """Drop a ", discount 0%" tail so orders without a discount read cleanly."""
    return line.replace(", discount 0%", "")


def _order_line(order) -> str:
    """An (items, discount) pair as one printable receipt line, with its total."""
    if order is None:
        return "no order recorded"
    items, disc = order
    line = _no_zero_discount(receipt_line(dict(items, discount=disc)))
    return f"{line} = ${gold_total(items, disc):.2f}"


def show_state_model_owned(demo: dict = None) -> None:
    """The model-owned run: the customer revises the order over six turns, the
    register confirms each from memory, and code reads back the final order it
    integrated (right or wrong)."""
    demo = demo or STATE_DEMO_MODEL
    for line, reply in demo["turns"]:
        show_turn("you", line)
        show_turn(demo["model"], reply)
    show_turn("ORDER", _order_line(demo["order"]))


def show_state_code_owned(demo: dict = None) -> None:
    """The code-owned run: the loop seeds an empty order, then each turn hands the
    model the order it holds and takes the model's reading back as the new order.
    The model never carries the running total; the loop does. One STATE line seeds
    it, because every later reading is the state, and reprinting it would just echo
    the model's own last line."""
    demo = demo or STATE_DEMO_CODE
    show_turn("STATE", "holding: empty")                     # the loop's order before turn one
    for line, _before, after in demo["turns"]:
        show_turn("you", line)                               # the one change this turn
        show_turn(demo["model"], _no_zero_discount(after))   # the reading the loop now holds
    show_turn("ORDER", _order_line(demo["order"]))


# --- Quality attributes, read off the same two registers -------------------------
# Latency is already in CONTROL_STUDY. The other measurable attributes need a
# token meter, so these runners mirror run_model_driven / run_code_driven exactly
# (same prompts, tools, and budgets; keep them in sync) but also sum the prompt
# and generation token counts Ollama reports for every model call in the loop.

def _metered_loop(prompt, tools, handlers, model, max_tokens, max_iter, think):
    """``run_with_trace``'s loop with a meter: returns ``(tokens_in, tokens_out)``
    summed over every model call the loop makes (thinking tokens included)."""
    from genai.llm import BRIEF
    opts = {"num_predict": max_tokens}
    kw = {} if think else {"think": False}
    messages = [{"role": "system", "content": BRIEF},
                {"role": "user", "content": prompt}]
    tin = tout = 0
    for _ in range(max_iter):
        resp = _client.chat(model=model, messages=messages,
                            tools=tools, options=opts, **kw)
        tin += resp.get("prompt_eval_count") or 0
        tout += resp.get("eval_count") or 0
        msg = resp["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        for call in tool_calls:
            name = call["function"]["name"]
            args = dict(call["function"]["arguments"])
            result = handlers.get(name, lambda **a: f"unknown tool: {name}")(**args)
            messages.append({"role": "tool", "content": json.dumps(result)})
    return tin, tout


def run_model_driven_metered(question: str, model: str = EIS_MODEL) -> tuple:
    """``run_model_driven`` with a token meter: returns ``(total, tokens)``."""
    box = {}
    handlers = {"price": price,
                "submit_total": lambda amount: box.update(total=round(float(amount), 2))}
    tools = [tool_spec("price", "Look up the price of one menu item in dollars.",
                       item="string"),
             tool_spec("submit_total", "Report the final order total in dollars.",
                       amount="number")]
    prompt = ("You are a cafe register. Price each item with the price tool, work "
              "out the order's total in dollars after any discount, then call "
              f"submit_total with the final amount.\nOrder: {question}")
    tin, tout = _metered_loop(prompt, tools, handlers, model,
                              max_tokens=900, max_iter=14, think=True)
    return box.get("total"), tin + tout


def run_code_driven_metered(question: str, model: str = EIS_MODEL) -> tuple:
    """``run_code_driven`` with a token meter: returns ``(total, tokens)``."""
    box = {}

    def submit_order(discount=0, **counts):
        items = {k: int(v) for k, v in counts.items() if k in MENU and int(v) > 0}
        box["order"] = (items, int(discount))
        return "recorded"

    params = {it: "integer" for it in MENU}
    params["discount"] = "integer"
    tools = [tool_spec("submit_order",
                       "Record the order: a count for each menu item (0 if not "
                       "ordered) and the percent discount.", **params)]
    prompt = ("Read this cafe order and record it with the submit_order tool, "
              f"giving a count for every menu item and the discount.\nOrder: {question}")
    tin, tout = _metered_loop(prompt, tools, {"submit_order": submit_order},
                              model, max_tokens=400, max_iter=4, think=False)
    if "order" not in box:
        return None, tin + tout
    return gold_total(*box["order"]), tin + tout


def quality_study(order: int = 1, trials: int = 12, model: str = EIS_MODEL) -> dict:
    """Run one order through both register designs ``trials`` times and read
    three quality attributes off the runs: *reliability* (fraction of runs that
    total correctly), *cost* (mean tokens per order, prompt + generated, summed
    over every call in the loop), and *consistency* (how many different answers
    the same question produced)."""
    question, items, disc = ORDERS[order]
    gold = gold_total(items, disc)
    study = {"order": question, "gold": gold, "trials": trials}
    for name, runner in [("model_driven", run_model_driven_metered),
                         ("code_driven", run_code_driven_metered)]:
        hits, toks, answers = 0, [], []
        for _ in range(trials):
            total, tokens = runner(question, model)
            hits += int(total is not None and abs(total - gold) < 0.01)
            toks.append(tokens)
            answers.append(total)
        study[name] = {
            "reliability": round(hits / trials, 3),
            "tokens_per_order": round(mean(toks)),
            "distinct_answers": len(set(answers)),
            "answers": sorted((a for a in set(answers) if a is not None)) +
                       (["none"] if None in answers else []),
        }
    return study


# Captured by scripts/_arch_quality_probe.py (gpt-oss:20b, order 1, trials=12).
# The model-owned loop's failures on this order were all silence: the loop
# ended without ever calling submit_total ("none"), not with a wrong number.
QUALITY_STUDY = {
    "order": "Two Red Bulls, a Monster, and three espressos, with the 15% member discount.",
    "gold": 15.94,
    "trials": 12,
    "model_driven": {"reliability": 0.583, "tokens_per_order": 3058,
                     "distinct_answers": 2, "answers": [15.94, "none"]},
    "code_driven": {"reliability": 0.917, "tokens_per_order": 818,
                    "distinct_answers": 2, "answers": [15.94, "none"]},
}


# --- Availability: what the register does when the model won't answer -------------
# Every attribute so far assumed the model answered. The one it can't is
# availability: the reader is a network call to a model server, and sometimes
# that call times out, drops, or comes back empty (a busy GPU, a rate limit, a
# bad deploy). Two registers meet the same outage, injected on a fixed schedule
# so the comparison is controlled: every third call the model is unreachable.
# The naive register has no handling and lets the error propagate; the guarded
# one catches it, degrades the order to a flagged manual-entry outcome, and
# stays up. The served/crashed/degraded counts follow from the schedule alone,
# so availability_study is deterministic and needs no model. The one real gemma4
# reading in AVAILABILITY_DEMO is captured once by
# scripts/_arch_availability_probe.py and baked.

READER_MODEL = "gemma4:latest"  # a competent reader, so the only variable is the outage


class Unavailable(Exception):
    """Stands in for a model call that timed out, dropped, or refused: the
    reader did not answer. A real outage raises the same kind of error."""


def _model_reader(model: str = READER_MODEL):
    """A reader that turns a spoken order into a receipt with a real model call,
    raising Unavailable if the call fails the way a live outage would surface."""
    def read(question: str):
        try:
            text, _ = read_receipt(question, model)
        except Exception as exc:                       # a real network/server failure
            raise Unavailable(str(exc))
        receipt = parse_receipt(text)
        if receipt is None:
            raise Unavailable("model returned no usable receipt")
        return receipt
    return read


def run_register_naive(question: str, read) -> dict:
    """A register with no fallback. It reads the order and totals it, and nothing
    here stops a reader failure: if ``read`` raises because the model is
    unreachable, the error propagates and this request dies with it."""
    receipt = read(question)                           # may raise Unavailable
    return {"status": "ok", "total": receipt_total(receipt)}


def run_register_guarded(question: str, read) -> dict:
    """The same register with the one fallback the naive version lacks. If the
    reader is unreachable it doesn't crash: it records the order as DEGRADED for
    manual entry and stays up for the next customer."""
    try:
        receipt = read(question)
        return {"status": "ok", "total": receipt_total(receipt)}
    except Unavailable:
        return {"status": "degraded", "total": None,
                "note": "reader unavailable -> order flagged for manual entry"}


def availability_schedule(trials: int = 3) -> list:
    """The register's calls for the study, with a deterministic outage on every
    third one. Returns (question, items, discount, unavailable) tuples."""
    runs, i = [], 0
    for question, items, disc in ORDERS:
        for _ in range(trials):
            runs.append((question, items, disc, i % 3 == 2))
            i += 1
    return runs


def _stub_reader(items: dict, disc: int, unavailable: bool):
    """A model-free reader for the study: raises Unavailable on an outage,
    otherwise returns the order's own receipt. Reading accuracy is not what this
    study measures, so the stub keeps it deterministic and off the model."""
    def read(_question):
        if unavailable:
            raise Unavailable("model did not respond (timeout)")
        return dict(items, discount=disc)
    return read


def availability_study(trials: int = 3) -> dict:
    """Run both registers against the same injected outage schedule and count how
    each request ended: served, crashed (an uncaught reader failure), or degraded
    (caught and flagged). The counts follow from the schedule, so this is
    deterministic and needs no model."""
    naive = {"served": 0, "crashed": 0}
    guarded = {"served": 0, "degraded": 0, "crashed": 0}
    for question, items, disc, unavailable in availability_schedule(trials):
        read = _stub_reader(items, disc, unavailable)
        try:
            run_register_naive(question, read)
            naive["served"] += 1
        except Unavailable:
            naive["crashed"] += 1
        out = run_register_guarded(question, read)
        guarded["degraded" if out["status"] == "degraded" else "served"] += 1
    return {"n": len(availability_schedule(trials)),
            "naive": naive, "guarded": guarded}


# Captured by scripts/_arch_availability_probe.py. The available reading is one
# real gemma4:latest receipt; the outage lines are the two registers' real return
# values when the reader raises Unavailable on the same order.
AVAILABILITY_STUDY = {
    "n": 15,
    "naive": {"served": 10, "crashed": 5},
    "guarded": {"served": 10, "degraded": 5, "crashed": 0},
}

AVAILABILITY_DEMO = {
    "model": "gemma4:latest",
    "available": {
        "question": "Three lattes, two Rockstars, and a tea, please.",
        "reading": '{"latte": 3, "rockstar": 2, "tea": 1, "redbull": 0, '
                   '"monster": 0, "espresso": 0, "discount": 0}',
        "total": 22.75,
    },
    "outage": {
        "question": "A Rockstar, a Red Bull, and two Monsters, take 10% off "
                    "for the loyalty card.",
        "error": "model gemma4:latest did not respond (timeout)",
        "naive": "uncaught Unavailable -> request failed, the order is lost",
        "guarded": "DEGRADED -> reader unavailable, order flagged for manual entry",
    },
}


def show_availability(demo: dict = None) -> None:
    """Both registers on the same two calls: first the model answers and both
    serve the total, then the model is unreachable and the designs part ways,
    the naive one dying where the guarded one degrades and stays up."""
    demo = demo or AVAILABILITY_DEMO
    av, out = demo["available"], demo["outage"]
    show_turn("you", av["question"])
    _show_reply(demo["model"], av["reading"])
    show_turn("REGISTER", f"served: code totals it at ${av['total']:.2f}")
    show_turn("you", out["question"])
    show_turn("OUTAGE", out["error"])
    show_turn("NAIVE", out["naive"])
    show_turn("GUARDED", out["guarded"])


# --- Design patterns: validator, cascade, checker --------------------------------
# The same cafe world, three wiring patterns. A *validator* checks the model's
# receipt against a hard schema and hands the error back for one retry. A
# *cascade* tries a cheap model first and escalates to a big one only when the
# validator rejects. A *checker* has one model audit another's receipt against
# the spoken order, the judgment code can't make. The cheap tier is gemma3:1b
# (the smallest model in the book's zoo, staging failures honestly); the big
# tier is gemma4, the house language model. All three studies are
# nondeterministic: capture them once with scripts/_arch_patterns_probe.py and
# bake the numbers plus one representative transcript each.

CHEAP_MODEL = "gemma3:1b"
BIG_MODEL = "gemma4:latest"
ALLOWED_DISCOUNTS = (0, 10, 15, 20)  # the percents posted at the register


def receipt_prompt(question: str, feedback: str = "") -> str:
    """The one bounded job: read a spoken order into a strict JSON receipt."""
    fix = f"\nYour last receipt was rejected: {feedback} Send a corrected one." \
        if feedback else ""
    return ("Read this cafe order and reply with one JSON object, nothing else. "
            f"Allowed keys: {', '.join(MENU)}, discount. Map each ordered item "
            "to its quantity as a whole number, and discount to the percent off "
            f"as a whole number (0 if none).\nOrder: {question}{fix}")


def read_receipt(question: str, model: str = CHEAP_MODEL,
                 feedback: str = "") -> tuple:
    """One model call: returns ``(raw_text, tokens)`` for the receipt attempt."""
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user",
                                   "content": receipt_prompt(question, feedback)}],
                        options={"num_predict": 220})
    tokens = (resp.get("prompt_eval_count") or 0) + (resp.get("eval_count") or 0)
    return resp["message"]["content"].strip(), tokens


def parse_receipt(text: str):
    """Pull the first JSON object out of a reply (code fences tolerated).
    Returns a dict, or None if nothing parses."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_receipt(receipt) -> str:
    """The guardrail: check a receipt against the register's schema. Returns ""
    when it passes, otherwise one plain-English error the retry can carry."""
    if receipt is None:
        return "the reply was not a JSON object."
    for key, val in receipt.items():
        if key != "discount" and key not in MENU:
            return (f'unknown key "{key}": allowed keys are '
                    f"{', '.join(MENU)}, discount.")
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            return f'"{key}" must be a whole number, not {json.dumps(val)}.'
    if receipt.get("discount", 0) not in ALLOWED_DISCOUNTS:
        return (f'discount must be one of {", ".join(map(str, ALLOWED_DISCOUNTS))} '
                "(a percent, not a fraction).")
    if not any(receipt.get(k, 0) > 0 for k in MENU):
        return "the receipt lists no items."
    return ""


def receipt_total(receipt) -> float:
    """What the register charges for a valid receipt (computed by code)."""
    items = {k: v for k, v in receipt.items() if k in MENU and v > 0}
    return gold_total(items, receipt.get("discount", 0))


def receipt_correct(receipt, order_idx: int) -> bool:
    """True when a receipt's computed total matches the order's gold total."""
    if receipt is None or validate_receipt(receipt):
        return False
    _, items, disc = ORDERS[order_idx]
    return abs(receipt_total(receipt) - gold_total(items, disc)) < 0.01


# --- Pattern 1: the validator (check, then one retry) ----------------------------

def validated_receipt(question: str, model: str = CHEAP_MODEL,
                      retries: int = 1) -> dict:
    """The validator wrap: read a receipt, check it, and on failure hand the
    error back to the same model for up to ``retries`` more attempts. Returns
    {"receipt", "attempts", "tokens", "error"} (error "" on success)."""
    tokens, error = 0, ""
    for attempt in range(1 + retries):
        text, t = read_receipt(question, model, feedback=error)
        tokens += t
        receipt = parse_receipt(text)
        error = validate_receipt(receipt)
        if not error:
            return {"receipt": receipt, "attempts": attempt + 1,
                    "tokens": tokens, "error": ""}
    return {"receipt": None, "attempts": retries + 1,
            "tokens": tokens, "error": error}


def validator_study(model: str = CHEAP_MODEL, trials: int = 3) -> dict:
    """Naked vs wrapped: how often the cheap model's receipt passes the schema
    on its first try, and how often one validator-guided retry rescues it. The
    arms are paired (the wrapped run's first attempt *is* the naked run). Also
    tracks how many receipts in each arm total correctly, because a receipt can
    be valid and still misread the order."""
    n = naked_valid = wrapped_valid = naked_ok = wrapped_ok = 0
    for idx, (question, _, _) in enumerate(ORDERS):
        for _ in range(trials):
            n += 1
            out = validated_receipt(question, model)
            if out["receipt"] is None:
                continue
            correct = int(receipt_correct(out["receipt"], idx))
            wrapped_valid += 1
            wrapped_ok += correct
            if out["attempts"] == 1:
                naked_valid += 1
                naked_ok += correct
    return {"model": model, "n": n,
            "naked_valid": naked_valid, "wrapped_valid": wrapped_valid,
            "naked_correct": naked_ok, "wrapped_correct": wrapped_ok}


# Captured by scripts/_arch_patterns_probe.py (gemma3:1b, trials=3, n=15).
VALIDATOR_STUDY = {
    "model": "gemma3:1b", "n": 15,
    "naked_valid": 8, "wrapped_valid": 10,
    "naked_correct": 2, "wrapped_correct": 3,
}

# One real exchange from the same probe: the cheap model reads every item
# correctly but writes the discount as the float 0.0, which the schema rejects
# on type; the retry sends the same receipt with a whole-number 0 and passes.
VALIDATOR_DEMO = {
    "question": "Three lattes, two Rockstars, and a tea, please.",
    "attempt1": '```json\n{\n    "latte": 3,\n    "rockstar": 2,\n'
                '    "tea": 1,\n    "discount": 0.0\n}\n```',
    "error": '"discount" must be a whole number, not 0.0.',
    "attempt2": '```json\n{\n    "latte": 3,\n    "rockstar": 2,\n'
                '    "tea": 1,\n     "discount": 0\n}\n```',
}


# --- Pattern 2: the cascade (cheap first, escalate on rejection) ------------------

def cascade_run(question: str, cheap: str = CHEAP_MODEL,
                big: str = BIG_MODEL) -> dict:
    """Try the cheap model once; if the validator rejects its receipt, send the
    same order to the big model. Returns {"receipt", "escalated",
    "cheap_tokens", "big_tokens"}."""
    text, cheap_tokens = read_receipt(question, cheap)
    receipt = parse_receipt(text)
    if not validate_receipt(receipt):
        return {"receipt": receipt, "escalated": False,
                "cheap_tokens": cheap_tokens, "big_tokens": 0}
    text, big_tokens = read_receipt(question, big)
    return {"receipt": parse_receipt(text), "escalated": True,
            "cheap_tokens": cheap_tokens, "big_tokens": big_tokens}


def cascade_study(cheap: str = CHEAP_MODEL, big: str = BIG_MODEL,
                  trials: int = 3) -> dict:
    """Three arms over every order: cheap alone, big alone, and the cascade.
    Scores each arm's valid and correct receipts, counts how often the cascade
    escalated, and meters the big model's tokens per order in both arms that
    use it."""
    arms = {a: {"valid": 0, "correct": 0} for a in ("cheap", "big", "cascade")}
    n = escalated = 0
    big_alone_tokens, cascade_big_tokens = [], []
    for idx, (question, _, _) in enumerate(ORDERS):
        for _ in range(trials):
            n += 1
            for arm, model in (("cheap", cheap), ("big", big)):
                text, toks = read_receipt(question, model)
                receipt = parse_receipt(text)
                if arm == "big":
                    big_alone_tokens.append(toks)
                if not validate_receipt(receipt):
                    arms[arm]["valid"] += 1
                    arms[arm]["correct"] += int(receipt_correct(receipt, idx))
            run = cascade_run(question, cheap, big)
            escalated += int(run["escalated"])
            cascade_big_tokens.append(run["big_tokens"])
            if run["receipt"] is not None and not validate_receipt(run["receipt"]):
                arms["cascade"]["valid"] += 1
                arms["cascade"]["correct"] += int(receipt_correct(run["receipt"], idx))
    return {"cheap_model": cheap, "big_model": big, "n": n,
            "arms": arms, "escalated": escalated,
            "big_tokens_per_order": {
                "always_big": round(mean(big_alone_tokens)),
                "cascade": round(mean(cascade_big_tokens)),
            }}


# Captured by scripts/_arch_patterns_probe.py (gemma3:1b -> gemma4, trials=3).
CASCADE_STUDY = {
    "cheap_model": "gemma3:1b", "big_model": "gemma4:latest", "n": 15,
    "arms": {"cheap": {"valid": 6, "correct": 2},
             "big": {"valid": 15, "correct": 15},
             "cascade": {"valid": 15, "correct": 10}},
    "escalated": 7,
    "big_tokens_per_order": {"always_big": 143, "cascade": 65},
}

# One real exchange from the same probe: the cheap model writes the 10% discount
# as the fraction 0.10 and invents a latte the order never mentioned, the guard
# rejects it on the discount's type, and the big model's receipt passes.
CASCADE_DEMO = {
    "question": "A Rockstar, a Red Bull, and two Monsters, take 10% off for "
                "the loyalty card.",
    "cheap": '```json\n{\n  "latte": 1,\n  "rockstar": 1,\n  "redbull": 1,\n'
             '  "monster": 2,\n  "discount": 0.10\n}\n```',
    "error": '"discount" must be a whole number, not 0.1.',
    "big": '{"latte": 0, "rockstar": 1, "tea": 0, "redbull": 1, "monster": 2, '
           '"espresso": 0, "discount": 10}',
}


# --- Pattern 3: the checker (one model audits another's receipt) -----------------
# Five clean receipts and five with one planted reading error each: a quantity
# slipped, a discount dropped or changed, an item lost. Every planted receipt
# still passes the schema validator, and its total is computed correctly from
# its (wrong) items, so no arithmetic check can see the error. Only comparing
# the receipt against the spoken sentence can, and that comparison is the one
# judgment in this register that needs a model.

CHECKER_EDITS = [
    (0, {"latte": 2}, "three lattes recorded as two"),
    (1, {"discount": 0}, "the 15% member discount dropped"),
    (2, {"latte": 0}, "the two lattes lost entirely"),
    (3, {"monster": 3}, "two Monsters recorded as three"),
    (4, {"discount": 10}, "the 20% staff discount recorded as 10%"),
]


def checker_receipts() -> list:
    """The ten audit tasks: for every order, its clean receipt and one with a
    planted reading error. Returns (order_idx, receipt, planted, note) tuples."""
    tasks = []
    for (idx, edits, note) in CHECKER_EDITS:
        _, items, disc = ORDERS[idx]
        clean = dict(items, discount=disc)
        bad = dict(clean)
        for key, val in edits.items():
            if key != "discount" and val == 0:
                bad.pop(key, None)
            else:
                bad[key] = val
        tasks.append((idx, clean, False, "clean"))
        tasks.append((idx, bad, True, note))
    return tasks


def receipt_line(receipt) -> str:
    """A receipt as one printable line: ``latte x3, rockstar x2, discount 0%``."""
    parts = [f"{k} x{v}" for k, v in receipt.items() if k in MENU and v > 0]
    body = ", ".join(parts) if parts else "empty"   # a reading that landed no items
    return body + f", discount {receipt.get('discount', 0)}%"


# The audit question is the checker's whole contract. The naive form ("is this
# receipt correct?") licenses the model to flag anything it can find; the scoped
# form pins the audit to exactly the three fields the receipt owns.
CHECKER_ASKS = {
    "naive": ("Does the receipt record the order correctly? Answer with one "
              "word, OK or WRONG, then one short sentence saying why."),
    "scoped": ("Compare the receipt to the order on exactly three points: which "
               "items were ordered, each item's quantity, and the discount "
               "percent. Ignore everything else (prices, wording, formatting, "
               'phrases like "to go"). Answer WRONG only if an item, a '
               "quantity, or the discount does not match; otherwise answer OK. "
               "One word, then one short sentence why."),
}


def checker_verdict(question: str, receipt, model: str = BIG_MODEL,
                    style: str = "scoped") -> tuple:
    """Ask one model to audit a receipt against the spoken order. ``style``
    picks the audit question from ``CHECKER_ASKS``. Returns
    ``(verdict, raw_reply)`` where verdict is "ok", "wrong", or "unclear"."""
    prompt = ("A cafe register listened to a spoken order and recorded a receipt.\n"
              f'Order: "{question}"\nReceipt: {receipt_line(receipt)}\n'
              + CHECKER_ASKS[style])
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": 80})
    reply = resp["message"]["content"].strip()
    m = re.search(r"\b(OK|WRONG)\b", reply, re.IGNORECASE)
    verdict = m.group(1).lower() if m else "unclear"
    return ("ok" if verdict == "ok" else "wrong" if verdict == "wrong"
            else "unclear"), reply


def checker_study(model: str = BIG_MODEL, trials: int = 3,
                  style: str = "scoped") -> dict:
    """Catch rate on the planted errors and false alarms on the clean receipts,
    ``trials`` audits of each of the ten receipts under one audit question."""
    caught = alarms = planted_n = clean_n = 0
    for idx, receipt, planted, _ in checker_receipts():
        question = ORDERS[idx][0]
        for _ in range(trials):
            verdict, _ = checker_verdict(question, receipt, model, style)
            if planted:
                planted_n += 1
                caught += int(verdict == "wrong")
            else:
                clean_n += 1
                alarms += int(verdict == "wrong")
    return {"model": model, "style": style, "caught": caught,
            "planted_n": planted_n, "alarms": alarms, "clean_n": clean_n}


# Captured by scripts/_arch_patterns_probe.py (gemma4:latest, trials=3):
# both audit questions catch every planted error; they differ only in how
# often they cry wolf on a clean receipt.
CHECKER_STUDY_NAIVE = {
    "model": "gemma4:latest", "style": "naive",
    "caught": 15, "planted_n": 15, "alarms": 9, "clean_n": 15,
}
CHECKER_STUDY = {
    "model": "gemma4:latest", "style": "scoped",
    "caught": 15, "planted_n": 15, "alarms": 4, "clean_n": 15,
}

# One real audit from the same probe: order 4's receipt with the 20% staff
# discount recorded as 10%, schema-valid and arithmetically self-consistent
# from its own wrong items, caught only by comparing it to the spoken sentence.
CHECKER_DEMO = {
    "question": "Three espressos, two lattes, and a Rockstar, minus the 20% "
                "staff discount.",
    "receipt": {"espresso": 3, "latte": 2, "rockstar": 1, "discount": 10},
    "note": "the 20% staff discount recorded as 10%",
    "verdict": "wrong",
    "reply": "WRONG The discount percentage is incorrect on the receipt "
             "compared to the order.",
}


# --- Showing the patterns on the page ---------------------------------------------
# One representative exchange per pattern, captured once by
# scripts/_arch_patterns_probe.py and baked, so the cells are deterministic.
# Model rows carry the model's real raw text; GUARD and RECEIPT are event
# labels for lines computed by code.

def _show_reply(speaker: str, text: str) -> None:
    """One model turn: multi-line replies (pretty-printed JSON) go through
    ``show_code`` so their own line breaks survive; single-line replies wrap
    like ordinary speech."""
    from genai.agent import show_code
    if "\n" in text.strip():
        show_code(speaker, text.strip())
    else:
        show_turn(speaker, text.strip())


def show_validator(demo: dict = None) -> None:
    """The validator wrap, one real exchange: the cheap model's first receipt,
    the schema error the guard hands back, and the retry that passes."""
    demo = demo or VALIDATOR_DEMO
    show_turn("you", demo["question"])
    _show_reply("gemma3:1b", demo["attempt1"])
    show_turn("GUARD", f"rejected: {demo['error']} -> retry")
    _show_reply("gemma3:1b", demo["attempt2"])
    show_turn("GUARD", "accepted: receipt passes the schema")


def show_cascade(demo: dict = None) -> None:
    """The cascade, one real exchange: the cheap model's rejected receipt, the
    escalation, and the big model's receipt with the total code computed."""
    demo = demo or CASCADE_DEMO
    show_turn("you", demo["question"])
    _show_reply("gemma3:1b", demo["cheap"])
    show_turn("GUARD", f"rejected: {demo['error']} -> escalate")
    _show_reply("gemma4", demo["big"])
    total = receipt_total(parse_receipt(demo["big"]))
    show_turn("GUARD", f"accepted: code totals it at ${total:.2f}")


def show_checker(demo: dict = None) -> None:
    """The checker, one real audit: a receipt with a planted reading error,
    and the auditing model's real verdict."""
    demo = demo or CHECKER_DEMO
    show_turn("you", demo["question"])
    show_turn("RECEIPT", receipt_line(demo["receipt"]))
    _show_reply("gemma4", demo["reply"])

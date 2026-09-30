"""A conversation that knows when to stop: AutoGen's two-agent chat.

The graph made control flow a thing you draw; a crew made it a running order.
AutoGen makes it a *conversation*. Two agents, a writer and a checker, take turns
in a group chat, and the control flow is the turn-taking itself: the chat runs
until a message matches a *termination condition*. Point the condition at the word
"GROUNDED" and the checker ends the chat the moment it approves the answer.

That the stop is a text match is the whole risk. Take the condition away, or
tighten the checker past anything an answer can satisfy, and the two agents talk
forever, the conversational twin of the runaway retry edge in the graph. The
backstop is a second condition, a cap on the number of messages, and it catches
the loop the way LangGraph's recursion limit caught the graph's.

AutoGen (autogen-agentchat) is in maintenance mode as of mid-2026, with the
Microsoft Agent Framework named as its successor; it's taught here as the research
lineage that made two-agent chat and termination conditions standard
({cite}`wu2023autogen`). It reaches Ollama through autogen-ext, and like CrewAI its
executor wants a non-empty message every turn, so the chat runs on gemma4 rather
than the graph's gpt-oss. Runs are nondeterministic, so the demo cells bake a
captured conversation (scripts/_groupchat_probe.py).
"""
import asyncio
import threading

from genai.llm import SERVER

GROUPCHAT_MODEL = "gemma4:latest"   # gpt-oss returns empty content inside the chat executor


def _run(coro):
    """Run one coroutine to completion in a fresh thread + loop (Jupyter-safe)."""
    box = {}

    def worker():
        loop = asyncio.new_event_loop()
        try:
            box["value"] = loop.run_until_complete(coro)
        finally:
            loop.close()
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return box["value"]


def _client(model: str = GROUPCHAT_MODEL):
    """An AutoGen model client pointed at a local Ollama model. AutoGen only ships
    capability metadata for the models it knows, so a local model needs its
    ``model_info`` spelled out; this chat uses no tools or structured output."""
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    info = {"vision": False, "function_calling": False, "json_output": False,
            "family": "unknown", "structured_output": False}
    return OllamaChatCompletionClient(model=model, host=SERVER, model_info=info,
                                      options={"temperature": 0})


def build_grounding_chat(model: str = GROUPCHAT_MODEL, strict: bool = False):
    """A writer and a checker in a round-robin chat, stopped by two conditions.

    The writer answers from the passage; the checker replies ``GROUNDED`` when the
    answer holds up, otherwise names what's missing so the writer can revise. The
    chat ends on whichever fires first: the checker saying ``GROUNDED`` (a
    text-mention condition) or a six-message cap (the backstop). ``strict=True``
    tells the checker never to approve, standing in for a stop condition tuned too
    tight, so only the cap can end the chat.
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import (TextMentionTermination,
                                              MaxMessageTermination)
    client = _client(model)
    writer = AssistantAgent(
        name="writer", model_client=client,
        system_message="Answer the reader's question in one sentence using only the "
                       "passage. If the checker names a gap, revise in one sentence.")
    checker_rule = ("Never say GROUNDED; always find one more nit to fix and ask for "
                    "a revision." if strict else
                    "If the answer is supported by the passage, reply with exactly "
                    "the single word GROUNDED. Otherwise name what's missing in one "
                    "line.")
    checker = AssistantAgent(name="checker", model_client=client,
                             system_message="You check the writer's answer against "
                                             "the passage. " + checker_rule)
    stop = MaxMessageTermination(6)
    if not strict:
        stop = TextMentionTermination("GROUNDED") | stop
    return RoundRobinGroupChat([writer, checker], termination_condition=stop)


def run_grounding_chat(question: str, passage: str, model: str = GROUPCHAT_MODEL,
                       strict: bool = False) -> dict:
    """Run the chat and return the message trace and why it stopped.

    Returns ``{trace, stop_reason, messages, capped}`` where ``trace`` is the list
    of ``(speaker, text)`` turns, ``stop_reason`` is AutoGen's own explanation, and
    ``capped`` is whether the six-message backstop fired (rather than a GROUNDED)."""
    chat = build_grounding_chat(model, strict=strict)
    task = f"Question: {question}\nPassage: {passage}"
    result = _run(chat.run(task=task))
    trace = [(m.source, str(m.content).strip()) for m in result.messages
             if m.source != "user"]
    reason = result.stop_reason or ""
    return {"trace": trace, "stop_reason": reason, "messages": len(result.messages),
            "capped": "Maximum" in reason or "message" in reason.lower()
                      and "GROUNDED" not in reason}


# The passage Sophia's retrieve step returns for the running question, trimmed.
GROUNDING_PASSAGE = ("Every new token the model writes still attends to everything "
                     "stored so far, so token generation itself slows as the "
                     "transcript grows.")
GROUNDING_QUESTION = "why do conversations with a model slow down?"


# Captured by scripts/_groupchat_probe.py (gemma4). Real writer/checker turns, kept
# verbatim. The clean chat ends the instant the checker says GROUNDED.
GROUPCHAT_DEMO = {
    "question": GROUNDING_QUESTION,
    "answer": "Token generation slows down because every new token the model writes "
              "still attends to everything stored so far, causing it to slow as the "
              "transcript grows.",
    "verdict": "GROUNDED",
    "stop_reason": "Text 'GROUNDED' mentioned",
}


def show_grounding_chat(demo=GROUPCHAT_DEMO):
    """The terminating chat as a transcript: the writer answers, the checker approves
    with the one word the condition watches for, and the chat stops itself."""
    from genai.agent import show_turn
    show_turn("you", demo["question"])
    show_turn("writer", demo["answer"])
    show_turn("checker", demo["verdict"])
    show_turn("GUARD", f'chat ends: {demo["stop_reason"]}')


# Captured with strict=True: the checker is told never to approve, so no message
# ever matches "GROUNDED" and the six-message cap is the only thing that ends it.
# The nit is the checker's real first complaint, kept verbatim and clipped on show.
GROUPCHAT_RUNAWAY_DEMO = {
    "question": GROUNDING_QUESTION,
    "answer": "Token generation slows down because every new token the model writes "
              "still attends to everything stored so far, causing it to slow as the "
              "transcript grows.",
    "nit": "The answer is accurate, but you repeat the idea of slowing down twice "
           "(\"Token generation slows down because...\" and \"...causing it to "
           "slow\"). You could make the sentence more concise by restructuring the "
           "second half to avoid this repetition while keeping all the original "
           "meaning. Can you revise the sentence to improve its flow?",
    "stop_reason": "Maximum number of messages 6 reached, current message count: 6",
}


def show_groupchat_runaway(demo=GROUPCHAT_RUNAWAY_DEMO):
    """The runaway: with the checker told never to approve, the writer and checker
    trade revisions until the message cap, not the work, ends the chat, the
    conversational twin of the graph's recursion-limit runaway."""
    from genai.agent import show_turn, _clip
    show_turn("you", demo["question"])
    show_turn("writer", demo["answer"])
    show_turn("checker", _clip(demo["nit"], 92))
    show_turn("chat", "... writer revises, checker finds one more nit, never GROUNDED")
    show_turn("GUARD", f"stopped: {demo['stop_reason']}")

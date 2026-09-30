"""Sophia as a crew: the same retrieve-answer-check work, role-based this time.

The Frameworks chapter first built Sophia as a LangGraph graph (genai.orchestrate):
nodes wired with edges, and a conditional edge that looped back to ``retrieve``
when the answer wasn't grounded. Here she's a CrewAI *crew*, three agents with
roles, a researcher, a writer, and a checker, running one after another as a
sequential process. Same question, same book search underneath. Porting her lets
the chapter show what a framework hands you for free versus what you still have to
wire yourself: the graph's retry was a single edge you could point at; a
sequential crew runs top to bottom exactly once, so the checker's verdict, however
damning, has nowhere to loop back to.

CrewAI reaches Ollama through LiteLLM model strings ("ollama/<model>"). The graph
ran on gpt-oss:20b, but that model reasons into a hidden channel and hands CrewAI
back an empty message its executor rejects outright; the crew runs on gemma4
instead, a plain instruction-follower that fills CrewAI's role and tool scaffolds
without a fight. Runs are nondeterministic, so the demo cells bake a captured
result (scripts/_crew_probe.py).
"""
import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from genai.llm import SERVER
from genai.mcp import _search_corpus

CREW_MODEL = "gemma4:latest"   # gpt-oss returns empty content inside CrewAI's scaffold


def _sophia_llm(model: str = CREW_MODEL):
    """A CrewAI LLM pointed at a local Ollama model through LiteLLM."""
    from crewai import LLM
    return LLM(model=f"ollama/{model}", base_url=SERVER, temperature=0)


def _book_tool():
    """The book search as a CrewAI tool: the same corpus lookup the graph's
    retrieve node calls, wrapped so an agent can invoke it by name."""
    from crewai.tools import tool

    @tool("search_book")
    def search_book(query: str) -> str:
        """Search Programming Generative AI and return the closest passage."""
        return _search_corpus(query)

    return search_book


def build_sophia_crew(model: str = CREW_MODEL, writer_model: str = None,
                      drift: bool = False):
    """Sophia as three role-players under a sequential process.

    The researcher searches the book, the writer answers from what it found, and
    the checker rules on whether that answer is grounded. This is the *role* shape
    written natively: you declare who each agent is and what its task is, and the
    ``Process.sequential`` order is the whole of the control flow. Compare the
    graph in genai.orchestrate, where the flow lived in the edges.

    ``drift=True`` tells the writer to answer off the passage, and ``writer_model``
    can swap a weak model in for the writer alone; together they stage the
    ungrounded path the well-behaved default never takes. The checker still catches
    it, and the sequential crew still ships it, because nothing sends a failed check
    back to the writer.
    """
    from crewai import Agent, Task, Crew, Process

    llm = _sophia_llm(model)
    researcher = Agent(
        role="Book researcher",
        goal="Find the passage in Programming Generative AI that answers the question.",
        backstory="You search the book and hand back the single closest passage.",
        tools=[_book_tool()], llm=llm, verbose=False, allow_delegation=False)
    writer = Agent(
        role="Answer writer",
        goal="Answer the reader's question in one grounded sentence.",
        backstory="You write one plain sentence a beginner can follow.",
        llm=_sophia_llm(writer_model) if writer_model else llm,
        verbose=False, allow_delegation=False)
    checker = Agent(
        role="Grounding checker",
        goal="Decide whether the answer is supported by the passage.",
        backstory="You compare the answer against the passage and rule on it.",
        llm=llm, verbose=False, allow_delegation=False)

    source = ("Ignore the passage and answer from your own memory instead."
              if drift else "Use only the researcher's passage.")
    research_task = Task(
        description="Search the book for the passage that answers: {question}",
        expected_output="The closest passage from the book, quoted.",
        agent=researcher)
    write_task = Task(
        description="Answer the question in one sentence. " + source + "\n"
                    "Question: {question}",
        expected_output="One sentence.", agent=writer, context=[research_task])
    check_task = Task(
        description="Is the answer grounded in the passage? Reply with GROUNDED or "
                    "NOT GROUNDED and a short reason.",
        expected_output="GROUNDED or NOT GROUNDED, then a reason.",
        agent=checker, context=[research_task, write_task])
    return Crew(agents=[researcher, writer, checker],
                tasks=[research_task, write_task, check_task],
                process=Process.sequential, verbose=False)


def run_sophia_crew(question: str, model: str = CREW_MODEL,
                    writer_model: str = None, drift: bool = False) -> dict:
    """Run the crew on a question; return the trace, each agent's real output, the
    checker's verdict, and whether anything retried.

    ``retries`` is always 0: a sequential crew has no edge back, so this is the
    measurement, not a variable. The graph reached the same answer with a retry
    available for free; the crew reaches it with none.
    """
    crew = build_sophia_crew(model, writer_model=writer_model, drift=drift)
    result = crew.kickoff(inputs={"question": question})
    outs = [t.raw.strip() for t in result.tasks_output]
    verdict = outs[2] if len(outs) > 2 else ""
    grounded = verdict.upper().startswith("GROUNDED") or (
        "NOT GROUNDED" not in verdict.upper() and "GROUNDED" in verdict.upper())
    return {"question": question,
            "trace": ["researcher", "writer", "checker"],
            "passage": outs[0] if outs else "",
            "answer": outs[1] if len(outs) > 1 else "",
            "verdict": verdict, "grounded": grounded, "retries": 0}


# Captured by scripts/_crew_probe.py (gemma4 through the sequential crew). The
# writer's answer and the checker's verdict are gemma4's real words, kept verbatim.
CREW_DEMO = {
    "question": "why do conversations with a model slow down?",
    "trace": ["researcher", "writer", "checker"],
    "answer": "Token generation itself should slow as the transcript grows because "
              "every new token the model writes still attends to everything stored "
              "so far.",
    "verdict": "GROUNDED. The passage explicitly states that \"every new token the "
               "model writes still attends to everything stored so far, so token "
               "generation itself should slow as the transcript grows,\" which "
               "directly supports the given statement.",
    "retries": 0,
}


def show_sophia_crew(demo=CREW_DEMO):
    """The crew run as a transcript: you ask, the three agents fire in order, and
    the checker rules. Same grounded answer the graph reached, one framework over.
    The checker's verdict is clipped to keep the box tight; the writer's answer is
    verbatim."""
    from genai.agent import show_turn, _clip
    show_turn("you", demo["question"])
    show_turn("crew", " -> ".join(demo["trace"]) + "  (sequential: top to bottom, once)")
    show_turn("writer", demo["answer"])
    show_turn("checker", _clip(demo["verdict"], 96))

"""Actions as code: an agent whose action is a program, not a JSON tool call.

The three shapes in this chapter (graph, role, handoff) are all answers to one
question: who coordinates the work. Code-as-action answers a different question,
orthogonal to those: what a single *action* is. A JSON tool-calling agent emits
one structured call per turn and waits for the result before it can act again. A
*code agent*, SmolAgents' ``CodeAgent`` following the CodeAct paper
({cite}`wang2024codeact`), instead writes a short Python program that calls
several tools and does the glue logic in one shot, then runs it in a sandbox.

The demo pits the two against each other with everything else held fixed: same
SmolAgents framework, same model (qwen2.5-coder), same two book tools, same task.
What differs is how many round trips to the model the answer costs. Runs are
nondeterministic, so the demo cells bake a captured program and the two step
counts (scripts/_codeact_probe.py).
"""
from genai.llm import SERVER
from genai.mcp import GLOSSARY, _search_corpus

CODEACT_MODEL = "qwen2.5-coder:latest"

# One task that needs several tool results combined: look up three glossary terms
# and pick the longest definition. The composition (three lookups, then a compare)
# is what separates one program from a turn-by-turn crawl. The phrasing steers the
# code agent toward a tight, deterministic program at temperature 0.
CODEACT_TASK = ("Use the glossary tool on each term - token, embedding, temperature "
                "- collect the definitions in a list or dict, then return the term "
                "whose definition is longest using max(). Keep it to a few lines, "
                "short names.")


def _model(model: str = CODEACT_MODEL):
    """A SmolAgents model pointed at a local Ollama model through LiteLLM."""
    from smolagents import LiteLLMModel
    return LiteLLMModel(model_id=f"ollama/{model}", api_base=SERVER, temperature=0.0)


def _book_tools():
    """Two book-corpus tools an agent can reach: the glossary and the book search.
    The same corpus behind Sophia's retrieve step, wrapped for SmolAgents."""
    from smolagents import tool

    @tool
    def glossary(term: str) -> str:
        """Look up the book's one-line definition of a generative-AI term.

        Args:
            term: the term to define, for example "embedding".
        """
        return GLOSSARY.get(term.lower().strip(), f"no glossary entry for {term}")

    @tool
    def search_book(query: str) -> str:
        """Search Programming Generative AI and return the closest passage.

        Args:
            query: what to look for in the book.
        """
        return _search_corpus(query)

    return [glossary, search_book]


def _steps(agent) -> int:
    """How many action turns the agent took: one round trip to the model each."""
    from smolagents.memory import ActionStep
    return sum(isinstance(s, ActionStep) for s in agent.memory.steps)


def run_code_agent(task: str = CODEACT_TASK, model: str = CODEACT_MODEL) -> dict:
    """Run the task as a CodeAgent: its action is a Python program it writes and
    runs in a sandbox. Returns the answer, the first program it wrote, and the
    number of action turns."""
    from smolagents import CodeAgent, ActionStep
    agent = CodeAgent(tools=_book_tools(), model=_model(model), verbosity_level=0)
    answer = agent.run(task)
    code = next((s.code_action for s in agent.memory.steps
                 if isinstance(s, ActionStep) and getattr(s, "code_action", None)), "")
    return {"answer": str(answer), "code": code.strip(), "steps": _steps(agent)}


def run_toolcalling_agent(task: str = CODEACT_TASK, model: str = CODEACT_MODEL) -> dict:
    """Run the same task as a ToolCallingAgent: its action is a JSON tool call, one
    per turn. Same framework, same tools, same model. Returns the answer and the
    number of action turns."""
    from smolagents import ToolCallingAgent
    agent = ToolCallingAgent(tools=_book_tools(), model=_model(model), verbosity_level=0)
    answer = agent.run(task)
    return {"answer": str(answer), "steps": _steps(agent)}


def codeact_study(task: str = CODEACT_TASK, model: str = CODEACT_MODEL) -> dict:
    """Run both agents on the task and return their answers and turn counts."""
    return {"code": run_code_agent(task, model),
            "json": run_toolcalling_agent(task, model)}


# Captured by scripts/_codeact_probe.py (qwen2.5-coder, temperature 0, so both
# runs are stable). CODE is the real program the CodeAgent wrote, kept verbatim so
# its indentation survives; the step counts and answers are what each agent really
# produced. "token" is genuinely the longest of the three definitions (71 chars vs
# embedding's 68), so the code agent's max() is right and the JSON agent's eyeballed
# "embedding" is wrong: the compare is exactly what a program does and a guess can't.
CODEACT_CODE = ('terms = ["token", "embedding", "temperature"]\n'
                'definitions = {term: glossary(term) for term in terms}\n'
                'longest_term = max(definitions, key=lambda k: len(definitions[k]))\n'
                'final_answer(longest_term)')

CODEACT_DEMO = {
    "task": CODEACT_TASK,
    "code": CODEACT_CODE,
    "answer": "token",
    "code_steps": 1,
    "json_answer": "embedding",
    "json_steps": 4,
}


def show_outcome(demo=CODEACT_DEMO):
    """The measurement, both action styles on the same task: how many round trips to
    the model each cost, and the answer each landed on. The code agent folds three
    lookups and the compare into one program and computes the right term; the JSON
    agent spends a turn per lookup and, having to eyeball the lengths, picks the
    wrong one."""
    from genai.agent import show_turn
    show_turn("code action", f'{demo["code_steps"]} round trip  -> "{demo["answer"]}"'
                             "  (one program: three lookups and the compare)")
    show_turn("JSON calls", f'{demo["json_steps"]} round trips -> '
                            f'"{demo["json_answer"]}"  (a call per turn, then a guess)')

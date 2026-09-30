"""Generate Snake implementations and inspect what the model actually built.

Snake is one of the most over-represented programs in any code model's
training data, which makes it a sharp little lens on prompting. A bare
request returns a complete *classic* Snake every time, so the interesting
question is not "can it write Snake" but "can you talk it into writing a
Snake that is not the classic one." Getting a non-default behaviour out of
the model means fighting that canonical prior on purpose.
"""
import ast
import re

from genai.llm import ask, CODING_MODEL


def generate_snake(prompt: str, model: str = CODING_MODEL,
                   max_tokens: int = 1800) -> str:
    """Ask a code model for Snake and return just the Python source it wrote.

    Models wrap code in Markdown and often emit several fenced blocks (a
    `pip install` line, then the program), so we keep the longest fenced
    block that actually parses as Python. Temperature 0 keeps the artifact
    reproducible from one run to the next.
    """
    raw = ask(prompt, model=model, system="", max_tokens=max_tokens,
              options={"temperature": 0})
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", raw, re.S)
    parses = [b for b in blocks if _parses(b)]
    return max(parses or blocks or [raw], key=len).strip()


def walls_behavior(src: str) -> str:
    """Return 'WRAP' if the snake wraps at the edges, else 'GAME OVER'.

    A wrapping build moves the head with modulo arithmetic on the grid
    dimensions; the classic build instead flags game-over when the head
    leaves the field. We detect the modulo-wrap signature.
    """
    wrap = bool(re.search(r"%\s*\w*(?:width|height|grid|cols|rows|screen|board)",
                          src, re.I)) or "wrap" in src.lower()
    return "WRAP" if wrap else "GAME OVER"


def wall_lines(src: str) -> list:
    """Return the source lines that decide what happens at the board's edge.

    For a classic build this is the boundary test that ends the game; for a
    wrapping build it is the modulo arithmetic that teleports the head to the
    far side. Useful for seeing *why* walls_behavior returned what it did.
    """
    pat = re.compile(r"%\s*\w*(?:width|height|grid|cols|rows)"   # modulo wrap
                     r"|[<>]=?\s*\w*(?:width|height|dis_)"        # boundary test
                     r"|<\s*0|wrap", re.I)
    return [ln.strip() for ln in src.splitlines() if pat.search(ln)]


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False

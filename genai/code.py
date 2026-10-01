"""Code embeddings via UniXcoder — semantic understanding of source code."""
import numpy as np

_cache = {}   # model_name -> (tokenizer, model)

def _load(model_name):
    if model_name not in _cache:
        import torch  # noqa: F401  (lazy — only needed for code embeddings)
        from transformers import AutoTokenizer, AutoModel
        _cache[model_name] = (AutoTokenizer.from_pretrained(model_name),
                              AutoModel.from_pretrained(model_name))
    return _cache[model_name]


def _embed(code: str, model_name: str) -> np.ndarray:
    """Mean-pooled transformer embedding for a code snippet."""
    import torch
    tokenizer, model = _load(model_name)
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    avg = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return avg.squeeze().numpy()


def embed_code(code: str) -> np.ndarray:
    """Return a UniXcoder embedding vector for a code snippet."""
    return _embed(code, "microsoft/unixcoder-base")


def embed_codebert(code: str) -> np.ndarray:
    """Return a CodeBERT embedding vector for a code snippet."""
    return _embed(code, "microsoft/codebert-base")


_st_cache = {}

def embed_codesearch(code: str) -> np.ndarray:
    """Embed a snippet with a model fine-tuned for description-to-code search.

    flax-sentence-embeddings/st-codesearch-distilroberta-base was trained on
    CodeSearchNet query/code pairs, so unlike CodeBERT it is tuned for the
    retrieval task itself, not just exposed to code.
    """
    from sentence_transformers import SentenceTransformer
    if "csn" not in _st_cache:
        _st_cache["csn"] = SentenceTransformer(
            "flax-sentence-embeddings/st-codesearch-distilroberta-base")
    return _st_cache["csn"].encode(code)


def code_similarity(a: str, b: str) -> float:
    """Cosine similarity between two code snippets."""
    ea, eb = embed_code(a), embed_code(b)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def code_search(query: str, snippets: list, top_k: int = 3) -> list:
    """Return the top_k snippets most semantically similar to the query."""
    qv = embed_code(query)
    scored = sorted(snippets, key=lambda s: code_similarity(query, s), reverse=True)
    return scored[:top_k]


def code_search_scores(query: str, named_snippets: list, embed=embed_code) -> list:
    """Rank (name, code) snippets against a query using any embedder.

    Returns [(name, score), ...] sorted high to low, so two embedding models
    can be compared on the same query. Pass embed=embed_codebert to swap models.
    """
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    qv = embed(query)
    scored = [(name, cos(qv, embed(code))) for name, code in named_snippets]
    return sorted(scored, key=lambda x: -x[1])


def code_analogy(pairs: list, query: str, candidates: list) -> tuple:
    """Multi-example code analogy: learn a transformation from (src,tgt) pairs.

    Example:
        pairs    = [(recursive_sum, iterative_sum), (recursive_count, iterative_count)]
        query    = recursive_factorial
        candidates = [iterative_factorial, fibonacci]
    Returns (best_candidate, score).
    """
    transforms = [embed_code(t) - embed_code(s) for s, t in pairs]
    avg_transform = np.mean(transforms, axis=0)
    target = embed_code(query) + avg_transform
    best, best_score = None, -1
    for c in candidates:
        ec = embed_code(c)
        s = float(np.dot(target, ec) / (np.linalg.norm(target) * np.linalg.norm(ec)))
        if s > best_score:
            best, best_score = c, s
    return best, round(best_score, 4)


# --- chapter snippet corpus ---------------------------------------------
# The *data* for the embedding demos: tiny Python functions we embed, compare,
# search, and do vector arithmetic over. They live here (rather than inline in
# the notebook) so each chapter cell can show them as a syntax-highlighted
# figure via genai.viz.render_code_listing while the runnable code stays clean.

# Syntax vs. semantics — same task, different shape; different task, same shape.
SUM_ITER = """def sum_list(numbers):
    total = 0
    for n in numbers: total += n
    return total"""
SUM_FAST = "def add_up(lst): return sum(lst)"
MULTIPLY_ITER = """def multiply_list(numbers):
    result = 1
    for n in numbers: result *= n
    return result"""

# One original and four edits: cosmetic (rename, type) vs. structural (rewrite).
ADD_NUMBERS = """def add_numbers(a, b):
    return a + b"""
ADD_VARIANTS = {
    "renamed vars": """def add_numbers(x, y):
    return x + y""",
    "typed signature": """def add_numbers(a: int, b: int) -> int:
    return a + b""",
    "different logic": """def multiply(a, b):
    return a * b""",
    "unrelated": """def send_email(r, m):
    print(f'Sending {m} to {r}')""",
}

# The operator flip — one character apart, opposite meaning.
VALIDATE_PASS = """def validate_age(age):
    return age >= 21"""
VALIDATE_FAIL = """def validate_age(age):
    return age < 21"""

# A miniature corpus for semantic code search.
SEARCH_SNIPPETS = [
    "def add(a, b): return a + b",
    "def multiply(x, y): return x * y",
    "def read_file(path): return open(path).read()",
    "def sum_list(items): return sum(items)",
    "def send_email(addr, msg): pass",
]

# Vector arithmetic over code — a recursive -> iterative analogy.
ANALOGY_PAIRS = [
    ("""def sum_rec(n):
  if n==0: return 0
  return n+sum_rec(n-1)""",
     """def sum_iter(n):
  t=0
  for i in range(n+1): t+=i
  return t"""),
    ("""def count_rec(n):
  if n==0: return
  count_rec(n-1)
  print(n)""",
     """def count_iter(n):
  for i in range(1,n+1): print(i)"""),
]
ANALOGY_QUERY = """def fact_rec(n):
  if n<=1: return 1
  return n*fact_rec(n-1)"""
ANALOGY_CANDIDATES = [
    """def fact_iter(n):
    result=1
    for i in range(1,n+1): result*=i
    return result""",
    """def fibonacci(n):
    if n<=1: return n
    return fibonacci(n-1)+fibonacci(n-2)""",
]

# Code-specific vs. generic embeddings — named functions + a plain-text query.
EMBED_SNIPPETS = [
    ("add",        "def add(a, b): return a + b"),
    ("multiply",   "def multiply(x, y): return x * y"),
    ("sum_list",   "def sum_list(items): return sum(items)"),
    ("find_max",   "def find_max(lst): return max(lst)"),
    ("sort_desc",  "def sort_desc(lst): return sorted(lst, reverse=True)"),
    ("read_file",  "def read_file(path): return open(path).read()"),
    ("send_email", "def send_email(addr, msg): print(f'Sending {msg} to {addr}')"),
]
EMBED_QUERY = "combine two numbers"

# When code models disagree — a five-function subset scored by three embedders.
DISAGREE_FUNCS = [
    ("add",        "def add(a, b): return a + b"),
    ("multiply",   "def multiply(x, y): return x * y"),
    ("sum_list",   "def sum_list(items): return sum(items)"),
    ("read_file",  "def read_file(path): return open(path).read()"),
    ("send_email", "def send_email(addr, msg): pass"),
]


# --- code generation quality: a two-mode scorecard -----------------------
# Capability is a profile, not a single number. Each local code model is graded
# on two modes. INSTRUCTION: write a whole function from a natural-language spec.
# FILL-IN-THE-MIDDLE (FIM): complete a hole in an existing function, the way an
# editor's autocomplete fills the line between your cursor and what follows. The
# exact output each model produced is captured verbatim in code_zoo.json (one
# live Ollama run at temperature 0), so the grid is reproducible and
# deterministic even though generation is not. Regenerate the captures with
# scripts/_codezoo_capture.py.
from pathlib import Path as _Path
import json as _json

# Each task: key, the scorecard column it belongs to, mode, the function to grade,
# and a few hidden cases (every case is (args_tuple, expected)). A column groups
# three small tasks, so a model that solves two of them colours the square
# two-thirds rather than landing on a flat pass or fail. FIM tasks also carry the
# prefix/suffix that bracket the hole. Cases mirror scripts/_codezoo_capture.py so
# grading reproduces the captured grid exactly.
def _w(name, sig, what):
    return (f"Write a Python function {name}{sig} that {what}. "
            "Only code, no explanation.")

CODE_TASKS = [
    {"key": "reverse_words", "col": "strings", "mode": "instruction",
     "fn": "reverse_words",
     "prompt": _w("reverse_words", "(s)", "reverses the order of the words in a string"),
     "cases": [(("hello world",), "world hello"), (("a b c",), "c b a")]},
    {"key": "roman", "col": "strings", "mode": "instruction", "fn": "roman_to_int",
     "prompt": _w("roman_to_int", "(s)",
                  "converts a Roman numeral string to an integer"),
     "cases": [(("III",), 3), (("IX",), 9), (("MCMXCIV",), 1994)]},
    {"key": "palindrome", "col": "strings", "mode": "instruction", "fn": "is_palindrome",
     "prompt": _w("is_palindrome", "(s)", "returns True if s is a palindrome, "
                  "ignoring case, spaces and punctuation"),
     "cases": [(("A man, a plan, a canal: Panama",), True), (("race a car",), False)]},
    {"key": "is_prime", "col": "math", "mode": "instruction", "fn": "is_prime",
     "prompt": _w("is_prime", "(n)", "returns True if n is prime"),
     "cases": [((2,), True), ((7,), True), ((9,), False), ((1,), False)]},
    {"key": "factorial", "col": "math", "mode": "instruction", "fn": "factorial",
     "prompt": _w("factorial", "(n)", "returns n factorial"),
     "cases": [((0,), 1), ((5,), 120)]},
    {"key": "digit_sum", "col": "math", "mode": "instruction", "fn": "digit_sum",
     "prompt": _w("digit_sum", "(n)",
                  "returns the sum of the decimal digits of a non-negative integer n"),
     "cases": [((123,), 6), ((9009,), 18)]},
    {"key": "flatten", "col": "lists", "mode": "instruction", "fn": "flatten",
     "prompt": _w("flatten", "(lst)",
                  "flattens an arbitrarily nested list into one flat list"),
     "cases": [(([[1, 2], [3, [4]]],), [1, 2, 3, 4]),
               (([1, [2, [3, [4]]]],), [1, 2, 3, 4]), (([],), [])]},
    {"key": "median", "col": "lists", "mode": "instruction", "fn": "median",
     "prompt": _w("median", "(nums)", "returns the median of a list of numbers"),
     "cases": [(([3, 1, 2],), 2), (([1, 2, 3, 4],), 2.5)]},
    {"key": "chunk", "col": "lists", "mode": "instruction", "fn": "chunk",
     "prompt": _w("chunk", "(lst, n)", "splits lst into consecutive sublists of "
                  "length n (the last may be shorter)"),
     "cases": [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]),
               (([1, 2, 3], 1), [[1], [2], [3]])]},
    {"key": "gcd", "col": "loops", "mode": "fim", "fn": "gcd",
     "prefix": "def gcd(a, b):\n    while b:\n        ", "suffix": "\n    return a\n",
     "cases": [((48, 18), 6), ((7, 5), 1), ((1071, 462), 21)]},
    {"key": "sum_squares", "col": "loops", "mode": "fim", "fn": "sum_squares",
     "prefix": "def sum_squares(nums):\n    total = 0\n    for x in nums:\n        ",
     "suffix": "\n    return total\n",
     "cases": [(([1, 2, 3],), 14), (([],), 0), (([4],), 16)]},
    {"key": "count_vowels", "col": "loops", "mode": "fim", "fn": "count_vowels",
     "prefix": ("def count_vowels(s):\n    n = 0\n    for ch in s:\n"
                "        if ch in 'aeiou':\n            "),
     "suffix": "\n    return n\n",
     "cases": [(("hello",), 2), (("xyz",), 0), (("aeiou",), 5)]},
    {"key": "dedup", "col": "build", "mode": "fim", "fn": "dedup",
     "prefix": ("def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n"
                "        if x not in seen:\n            "),
     "suffix": "\n    return out\n",
     "cases": [(([1, 1, 2, 3, 3, 2],), [1, 2, 3]), (([],), []),
               ((["a", "b", "a"],), ["a", "b"])]},
    {"key": "char_count", "col": "build", "mode": "fim", "fn": "char_count",
     "prefix": "def char_count(s):\n    counts = {}\n    for ch in s:\n        ",
     "suffix": "\n    return counts\n",
     "cases": [(("aab",), {"a": 2, "b": 1}), (("",), {})]},
    {"key": "running_max", "col": "build", "mode": "fim", "fn": "running_max",
     "prefix": ("def running_max(nums):\n    out = []\n    best = float('-inf')\n"
                "    for x in nums:\n        best = max(best, x)\n        "),
     "suffix": "\n    return out\n",
     "cases": [(([1, 3, 2],), [1, 3, 3]), (([5, 4],), [5, 5])]},
]
_TASK = {t["key"]: t for t in CODE_TASKS}

# CODE_COLUMNS: the scorecard's columns in order, each (label, mode, [task keys]).
CODE_COLUMNS = []
for _t in CODE_TASKS:
    if not CODE_COLUMNS or CODE_COLUMNS[-1][0] != _t["col"]:
        CODE_COLUMNS.append((_t["col"], _t["mode"], []))
    CODE_COLUMNS[-1][2].append(_t["key"])

# CODER_OUTPUTS[model][task] is the captured output: the function source for an
# instruction task, the hole-filling middle for a FIM task (already clipped to the
# hole, the way an editor keeps only the insertion), or None when a model has no
# FIM mode at all.
CODER_OUTPUTS = _json.loads(
    (_Path(__file__).resolve().parent / "code_zoo.json").read_text())


def _eq(got, want):
    try:
        return abs(got - want) < 1e-9 if isinstance(want, float) else got == want
    except Exception:
        return False


def _run_cases(src, fn_name, cases):
    """Exec src, call fn_name on every case, count matches. Any parse or run error
    scores zero, which is the whole point: looking right is not being right. stdout
    from the generated code (stray test prints) is swallowed so it can't leak."""
    import io
    import warnings
    from contextlib import redirect_stdout
    try:
        ns: dict = {}
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # generated code may raise SyntaxWarnings
            exec(src, ns)
            fn = ns[fn_name]
            passed = sum(_eq(fn(*args), want) for args, want in cases)
    except Exception as e:
        return 0, type(e).__name__
    return passed, ("all correct" if passed == len(cases)
                    else "partial" if passed else "wrong output")


def grade_code(task_key: str, captured) -> tuple:
    """Grade one captured output for one task; return (passed, total, note).

    passed is None when a model has no mode for the task (FIM on a model that was
    never FIM-trained). For a FIM task the captured middle is dropped into the
    hole between the task's prefix and suffix and the assembled function is run.
    """
    t = _TASK[task_key]
    total = len(t["cases"])
    if captured is None:
        return None, total, "no FIM"
    src = t["prefix"] + captured + t["suffix"] if t["mode"] == "fim" else captured
    passed, note = _run_cases(src, t["fn"], t["cases"])
    return passed, total, note


def score_scorecard(outputs: dict) -> list:
    """Aggregate one model's captured outputs into per-column scores.

    Returns [(solved, total), ...] in CODE_COLUMNS order, where a task counts as
    solved only if it passes every hidden case, and (None, total) for a column the
    model has no mode for (FIM on a model that was never FIM-trained).
    """
    row = []
    for _col, _mode, keys in CODE_COLUMNS:
        graded = [grade_code(k, outputs[k]) for k in keys]
        if all(passed is None for passed, _t, _n in graded):
            row.append((None, len(keys)))
        else:
            row.append((sum(passed == tot for passed, tot, _n in graded), len(keys)))
    return row


def _first_miss(task_key: str, captured: str) -> str:
    """Run a captured output and describe the first thing that goes wrong: the
    exception it raises, or the first hidden case where it returns the wrong value.
    This is the verdict the eye can't reach by reading, only by running."""
    import io
    import warnings
    from contextlib import redirect_stdout
    t = _TASK[task_key]
    src = t["prefix"] + captured + t["suffix"] if t["mode"] == "fim" else captured
    try:
        ns: dict = {}
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(src, ns)
            fn = ns[t["fn"]]
            for args, want in t["cases"]:
                got = fn(*args)
                if not _eq(got, want):
                    call = f"{t['fn']}({', '.join(map(repr, args))})"
                    return f"{call} = {got!r}, want {want!r} -> runs, wrong"
        return "passes every case"
    except Exception as e:
        detail = getattr(e, "msg", None) or str(e).split("\n")[0]
        return f"{type(e).__name__}: {detail[:34]} -> won't run"


def show_failure(model: str, task_key: str, max_lines: int = 8) -> None:
    """Show a model's real output for a task and what running it does.

    Renders the verbatim code under the model's name (indentation preserved by
    show_code), then a TEST line with the live verdict: the wrong value it returns
    or the error it raises. The dangerous failures are the ones that run, so the
    TEST line is the only place that danger shows. Returns None.
    """
    from genai.agent import show_code, show_turn
    code = CODER_OUTPUTS[model][task_key] or "(nothing)"
    lines = code.split("\n")
    shown = lines[:max_lines] + (["    ..."] if len(lines) > max_lines else [])
    # clip over-long lines so each fits under the transcript gutter without
    # wrapping (the gutter eats ~14 columns; PDF transcripts don't soft-wrap)
    shown = [ln if len(ln) <= 50 else ln[:46] + " ..." for ln in shown]
    show_code(model, "\n".join(shown))
    show_turn("TEST", _first_miss(task_key, code))
    print()


# Legacy is_prime grader, kept for the model-survey probes in scripts/. The book
# itself now grades code with the scorecard above (grade_code / CODE_TASKS).
# A small suite: known primes map to True, composites and edge cases to False.
_PRIME_CASES = {2: True, 3: True, 5: True, 7: True, 13: True, 97: True,
                0: False, 1: False, 4: False, 9: False, 15: False, 100: False}

def grade_is_prime(src: str, cases: dict = _PRIME_CASES) -> tuple:
    """Run a generated is_prime function against test cases.

    Returns (passed, total, note). Code that will not even parse or run
    scores zero, which is exactly the point: looking right is not the same
    as being right.
    """
    total = len(cases)
    try:
        ns: dict = {}
        exec(src, ns)
        fn = ns["is_prime"]
        passed = sum(bool(fn(n)) == want for n, want in cases.items())
    except Exception as e:
        return 0, total, type(e).__name__
    return passed, total, "all correct" if passed == total else "wrong output"


# --- display helpers -------------------------------------------------------

def compare_code_embeddings(query, snippets):
    """Code-aware (UniXcoder) vs general (nomic) similarity, per snippet."""
    from genai.embed import embed, similarity
    print(f"{'function':12}  UniXcoder   nomic-embed-text\n" + "-" * 44)
    results = []
    for name, fn in snippets:
        uni = similarity(embed_code(query), embed_code(fn))
        gen = similarity(embed(query), embed(fn))
        print(f"{name:12}  {uni:.4f}      {gen:.4f}")
        results.append((name, uni, gen))
    return results


def compare_code_embedders(query, funcs):
    """Score each function against the query under three code embedders."""
    uni = dict(code_search_scores(query, funcs, embed=embed_code))
    cb = dict(code_search_scores(query, funcs, embed=embed_codebert))
    csn = dict(code_search_scores(query, funcs, embed=embed_codesearch))
    print(f"{'function':12} UniXcoder  CodeBERT  CodeSearch")
    for name, _ in funcs:
        print(f"{name:12} {uni[name]:6.3f}    {cb[name]:6.3f}    {csn[name]:6.3f}")
    return [(name, uni[name], cb[name], csn[name]) for name, _ in funcs]


def run_codegen_benchmark(models, prompt):
    """Time each model generating is_prime; print tok/s and return sorted rows."""
    from genai import time_call
    bench = []
    for model_id, info in models.items():
        r = time_call(prompt, model_id)
        bench.append({**info, "model_id": model_id, "tps": r["tokens_per_sec"]})
        print(f"{info['label']:18} {r['tokens_per_sec']:6.1f} tok/s")
    bench.sort(key=lambda x: -x["tps"])
    return bench

"""Tokenization utilities built on real tokenizers.

We keep several tokenizers side by side so the differences between them are
visible rather than asserted:

    "general"       GPT-4's BPE            (tiktoken cl100k_base)
    "multilingual"  GPT-4o's BPE           (tiktoken o200k_base, stronger on non-English)
    "code"          StarCoder2's code-first BPE  (collapses runs of indentation)
    "old"           GPT-2's 2019 BPE       (for historical contrast)

Unlike a model's ``prompt_eval_count``, these count the raw text only, with no
chat-template scaffolding wrapped around it. So ``count_tokens("hello")`` is 1,
not 11.
"""
from functools import lru_cache

# tiktoken encodings (pure-Python BPE, no model download)
_TIKTOKEN = {"general": "cl100k_base", "multilingual": "o200k_base"}

# Hugging Face tokenizers (downloaded once, then cached locally)
_HF = {"code": "bigcode/starcoder2-3b", "old": "gpt2"}

CHOICES = sorted(_TIKTOKEN) + sorted(_HF)


@lru_cache(maxsize=None)
def _load(name):
    """Return ``(kind, tokenizer)`` for a named tokenizer, loading it once."""
    if name in _TIKTOKEN:
        import tiktoken
        return "tiktoken", tiktoken.get_encoding(_TIKTOKEN[name])
    if name in _HF:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        return "hf", AutoTokenizer.from_pretrained(_HF[name])
    raise ValueError(f"unknown tokenizer {name!r}; choose from {CHOICES}")


def _encode(text, name):
    kind, tok = _load(name)
    if kind == "hf":
        return tok, tok.encode(text, add_special_tokens=False)
    # disallowed_special=() so a literal special-token string in the text
    # (e.g. "<|endoftext|>" when counting the book itself, which discusses
    # them) is counted as plain text rather than raising; no ordinary string
    # is affected.
    return tok, tok.encode(text, disallowed_special=())


def count_tokens(text: str, tokenizer: str = "general") -> int:
    """Number of tokens *text* becomes under the chosen tokenizer."""
    _, ids = _encode(text, tokenizer)
    return len(ids)


def tokenize(text: str, tokenizer: str = "general") -> list:
    """The actual pieces *text* breaks into, decoded back to readable text.

    e.g. ``tokenize("tokenization")`` -> ``['token', 'ization']``.
    """
    tok, ids = _encode(text, tokenizer)
    return [tok.decode([i]) for i in ids]


def token_ids(text: str, tokenizer: str = "general") -> list:
    """The integer IDs the model actually consumes. Text is never seen directly.

    e.g. ``token_ids("hello")`` -> ``[15339]``.
    """
    _, ids = _encode(text, tokenizer)
    return ids


def special_tokens(tokenizer: str = "code") -> list:
    """The reserved control tokens a model uses (end-of-text, fill-in-middle, ...).

    These are not words; they are structural signals baked into the vocabulary.
    """
    kind, tok = _load(tokenizer)
    if kind == "hf":
        return list(dict.fromkeys(tok.all_special_tokens))
    return sorted(tok.special_tokens_set)


def _apply_merge(symbols: tuple, a: str, b: str) -> tuple:
    """Replace every adjacent (a, b) in *symbols* with the merged token a+b."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def learn_bpe(words: list, num_merges: int = 10):
    """Run Byte Pair Encoding on a tiny word list, the real training procedure.

    Starts from single characters and repeatedly merges the most frequent
    adjacent pair. Returns ``(merges, splits)``: the ordered list of merges
    learned, and how each word ends up tokenized.
    """
    from collections import Counter
    splits = {w: tuple(w) for w in words}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym in splits.values():
            for pair in zip(sym, sym[1:]):
                pairs[pair] += 1
        if not pairs or pairs.most_common(1)[0][1] < 2:
            break  # nothing repeats, so further merges would be arbitrary
        (a, b), _ = pairs.most_common(1)[0]
        merges.append(a + b)
        splits = {w: _apply_merge(sym, a, b) for w, sym in splits.items()}
    return merges, splits


# --- display helpers -------------------------------------------------------
# Each keeps a demo cell down to a single readable call; the formatting,
# slicing, and tokenizer plumbing live here instead of on the page.

_DOT = "·"  # stand-in for a space, so indentation runs are visible


def show_special_tokens(tokenizer: str = "code", n: int = 6) -> None:
    """List the first ``n`` reserved control tokens of a code model."""
    toks = special_tokens(tokenizer)
    print(f"{len(toks)} special tokens, including:")
    for tok in toks[:n]:
        print("   ", tok)


def show_indent_tokens(line: str = "            return value") -> None:
    """Show how the 2019 GPT-2 tokenizer spends one token per indent space, while
    both a modern general tokenizer and a code-first one collapse the whole run
    into a single token. Spaces are drawn as ``·`` so the indentation is visible,
    and the first row counts the raw characters for comparison."""
    dot = lambda s: s.replace(" ", _DOT)
    print(f"{'line':8} {len(line):2d} chars   {dot(line)}")
    for kind in ["old", "general", "code"]:
        pieces = tokenize(line, kind)
        print(f"{kind:8} {len(pieces):2d} tokens  {[dot(p) for p in pieces]}")


def show_chars_per_token(samples: dict) -> None:
    """Characters per token for each labelled sample (the rule-of-thumb check)."""
    for label, text in samples.items():
        ratio = len(text) / count_tokens(text)
        print(f"{label:8} {len(text):3d} chars / {count_tokens(text):2d} tokens "
              f"= {ratio:.1f} chars/token")


def compare_tokenizers(samples: dict, other: str = "code") -> None:
    """Token counts under the general tokenizer vs ``other``, per labelled sample."""
    width = max(len(k) for k in samples) + 1
    for label, text in samples.items():
        print(f"{label:{width}} general:{count_tokens(text):3d}  "
              f"{other}:{count_tokens(text, other):3d}")


# Words built to hide a repeated letter inside chunks the tokenizer never splits
# to the character level: (word, the letter to count, the true count).
SPELLING_SURVEY = [
    ("strawberry",   "R", 3),
    ("raspberry",    "R", 3),
    ("mississippi",  "S", 4),
    ("parallel",     "L", 3),
    ("hippopotamus", "P", 3),
    ("possesses",    "S", 5),
]


def score_spelling_survey(items=SPELLING_SURVEY,
                          models=("llama3.2:1b", "llama3.1:latest",
                                  "qwen3:latest", "gemma4:latest")) -> dict:
    """Ask each model to count the repeated letter in each tricky word, once at
    temperature 0, and gather the answers into a study dict for plotting. Every
    model sees the word only as a few chunks (``str``/``aw``/``berry``), so the
    repeated letters stay sealed inside and the answers scatter. Real replies,
    reduced to the single integer the prompt asks for. Regenerates SPELLING_STUDY
    (``scripts/_tok_spelling_probe.py``)."""
    import re
    from genai import ask
    rows = []
    for word, letter, true in items:
        q = f"How many letter {letter}'s are in '{word}'? Reply with only the number."
        answers = []
        for m in models:
            raw = ask(q, model=m, think=False,
                      options={"temperature": 0}, max_tokens=8).strip()
            n = re.search(r"\d+", raw)
            answers.append(int(n.group()) if n else None)
        rows.append({"word": word, "letter": letter, "true": true,
                     "answers": answers})
    return {"models": list(models), "rows": rows}


# Baked from a real run of score_spelling_survey (scripts/_tok_spelling_probe.py),
# every model asked once at temperature 0. None of the four counts better than
# half: capability buys confident wrong answers, not the ability to see letters.
SPELLING_STUDY = {
    "models": ["llama3.2:1b", "llama3.1:latest", "qwen3:latest", "gemma4:latest"],
    "rows": [
        {"word": "strawberry",   "letter": "R", "true": 3, "answers": [5, 4, 3, 2]},
        {"word": "raspberry",    "letter": "R", "true": 3, "answers": [5, 4, 2, 2]},
        {"word": "mississippi",  "letter": "S", "true": 4, "answers": [6, 4, 4, 4]},
        {"word": "parallel",     "letter": "L", "true": 3, "answers": [2, 4, 2, 3]},
        {"word": "hippopotamus", "letter": "P", "true": 3, "answers": [6, 4, 3, 2]},
        {"word": "possesses",    "letter": "S", "true": 5, "answers": [5, 4, 3, 4]},
    ],
}


def show_indent_validity(model: str = "gemma4:latest") -> None:
    """Ask the model whether indentation is required for valid Python."""
    from genai import ask
    cases = [("indented", "def greet(n):\n    print(n)"),
             ("flat",     "def greet(n):\nprint(n)")]
    for label, code in cases:
        verdict = ask(f"Is this valid Python? Reply Yes or No only.\n{code}",
                      model=model, max_tokens=12).splitlines()[0]
        print(f"{label:9} -> {verdict}")


def show_digit_product(a: int = 3947, b: int = 6281,
                       model: str = "gemma4:latest") -> None:
    """A capable model still misses the product of digits it reads in chunks.
    Shown as a transcript: the full prompt, then the model's real reply. The
    exact product and how each number chunked stay in the prose walkthrough
    below, not in the speaker box."""
    from genai import ask
    from genai.agent import show_turn
    prompt = f"What is {a} * {b}? Reply with only the number."
    show_turn("USER", prompt)
    show_turn(model.split(":")[0], ask(prompt, model=model, max_tokens=16).strip())

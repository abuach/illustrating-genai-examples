"""Core LLM calls — ask() and chat().

Both functions use the Ollama chat endpoint internally so that top-level
parameters like `think` are honoured correctly.

Default behaviour enforces concise output via a system prompt and token cap.
Override per-call when you need something different:

    ask("Write a 500-word essay...", max_tokens=800, system=None)
    chat(messages, max_tokens=500)

For thinking models (gemma4, qwen3, deepseek-r1, etc.), disable thinking with
the `think=False` top-level parameter, or reserve a token budget with
`thinking_budget` if you want thinking enabled but need to account for its cost:

    ask = partial(_ask, model="gemma4:e2b", think=False)
    ask = partial(_ask, model="gemma4:e2b", thinking_budget=1000)

Every call goes to the Ollama server on this machine unless you point the
library elsewhere, either with the OLLAMA_HOST environment variable (the same
one Ollama's own tools read) or at runtime:

    from genai import set_host
    set_host("http://class-server.example.edu:11434")
"""
import os
import sys
from urllib.parse import urlsplit

import ollama


def _normalize_host(host: str) -> str:
    """Accept 'myserver', 'myserver:11434', or 'http://myserver:11434/' and
    return the 'http://myserver:11434' form every caller in the library uses."""
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    parts = urlsplit(host)
    if parts.scheme == "http" and parts.port is None:
        host = f"http://{parts.hostname}:11434"
    return host


SERVER        = _normalize_host(os.environ.get("OLLAMA_HOST") or "localhost:11434")
DEFAULT_MODEL = "gemma4:latest"
CODING_MODEL  = "qwen2.5-coder:latest"


class _SharedClient:
    """The one ollama.Client every genai module talks through. Modules import
    this object instead of building their own, so ``set_host`` can repoint the
    whole library by swapping the client inside it."""

    def __init__(self, host: str):
        self.connect(host)

    def connect(self, host: str) -> None:
        self._inner = ollama.Client(host=host)

    def __getattr__(self, name):
        return getattr(self._inner, name)


_client = _SharedClient(SERVER)


def set_host(host: str) -> str:
    """Point every genai call at the Ollama server at ``host`` and return its
    normalized URL. Takes a bare hostname, ``host:port``, or a full URL."""
    global SERVER
    SERVER = _normalize_host(host)
    _client.connect(SERVER)
    # Some modules call ollama.chat(...) directly; repoint those functions too.
    default = ollama.Client(host=SERVER)
    for name in ("chat", "generate", "embed", "embeddings", "show",
                 "list", "ps", "pull"):
        if hasattr(ollama, name):
            setattr(ollama, name, getattr(default, name))
    # Modules that copied the URL at import read it as a global when they run,
    # so rebinding it in each loaded module is enough.
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("genai.") and mod is not None:
            for attr in ("SERVER", "_SERVER"):
                if hasattr(mod, attr):
                    setattr(mod, attr, SERVER)
    return SERVER


def get_host() -> str:
    """The Ollama server URL genai is currently talking to."""
    return SERVER

# ── Default system prompt ────────────────────────────────────────────────────
BRIEF = (
    "You are a helpful assistant in a programming textbook. "
    "Keep every response short: 1–3 sentences for explanations, "
    "≤8 lines for code. Skip preamble, filler, and repeated context."
)

DEFAULT_MAX_TOKENS = 200


def ask(prompt: str,
        model:           str  = DEFAULT_MODEL,
        system:          str  = BRIEF,
        max_tokens:      int  = DEFAULT_MAX_TOKENS,
        thinking_budget: int  = 0,
        think:           bool = False,
        **kw) -> str:
    """Send a single prompt; return the trimmed response string.

    Uses the chat endpoint internally so that `think=False` is honoured.

    Args:
        prompt:          The user-facing text.
        model:           Any model available on the Ollama server.
        system:          System-level instruction. Pass system="" to disable.
        max_tokens:      Desired visible output length in tokens.
        thinking_budget: Extra tokens reserved for internal reasoning.
                         Set to ~1000 for thinking models when think=True.
        think:           Pass False to disable thinking on supported models.
                         Must be a top-level param (not inside options={}).
        **kw:            Forwarded to ollama.Client.chat() — e.g.
                         options={"temperature": 0.0, "top_k": 5}.
    """
    opts = {**kw.pop("options", {}), "num_predict": max_tokens + thinking_budget}
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": prompt})
    if think is not None:
        kw["think"] = think
    resp = _client.chat(model=model, messages=msgs, options=opts, **kw)
    return resp["message"]["content"].strip()


def chat(messages:       list,
         model:           str  = DEFAULT_MODEL,
         system:          str  = BRIEF,
         max_tokens:      int  = DEFAULT_MAX_TOKENS,
         thinking_budget: int  = 0,
         think:           bool = False,
         **kw) -> str:
    """Send a conversation; return the assistant's reply string.

    Args:
        messages:        List of {"role": "user"|"assistant", "content": "..."}.
        system:          Injected as the first system message if non-empty.
        max_tokens:      Desired visible output length in tokens.
        thinking_budget: Extra tokens reserved for internal reasoning.
        think:           Pass False to disable thinking on supported models.
    """
    opts = {**kw.pop("options", {}), "num_predict": max_tokens + thinking_budget}
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    if think is not None:
        kw["think"] = think
    resp = _client.chat(model=model, messages=msgs, options=opts, **kw)
    return resp["message"]["content"].strip()


def next_token_distribution(prompt: str,
                            model: str = DEFAULT_MODEL,
                            top_k: int = 8,
                            reply_start: str = None) -> list:
    """Reveal the model's probability distribution over the next single token.

    Returns a list of (token, probability) pairs sorted from most to least
    likely. This is the raw distribution a generative model samples from at
    every step. We use the Ollama logprobs API (not exposed through ask/chat)
    and temperature 0 so the snapshot is reproducible.

    With ``reply_start``, the model's answer is pre-filled with that text and
    the distribution is over the token it writes next in its own reply. That
    differs from sending ``prompt + reply_start`` as a new message, which the
    model may answer (or correct) instead of continuing.
    """
    import json, math, urllib.request
    request = {"model": model, "stream": False,
               "options": {"num_predict": 1, "temperature": 0},
               "logprobs": True, "top_logprobs": top_k}
    if reply_start is None:
        endpoint, request["prompt"] = "generate", prompt
    else:   # Ollama continues a trailing assistant message
        endpoint, request["messages"] = "chat", [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply_start}]
    req = urllib.request.Request(f"{SERVER}/api/{endpoint}",
                                 data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    cands = json.loads(urllib.request.urlopen(req).read())["logprobs"][0]["top_logprobs"]
    return [(c["token"], math.exp(c["logprob"])) for c in cands]

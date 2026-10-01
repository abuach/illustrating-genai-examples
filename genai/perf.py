"""LLM performance measurement helpers."""
import subprocess
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def _metrics(r: dict, model: str) -> dict:
    """Package an Ollama response's timing fields into one small dict."""
    gen_tokens = r.get("eval_count", 0)
    gen_ns = r.get("eval_duration", 1)
    prompt_ns = r.get("prompt_eval_duration", 0)
    return {
        "model": model,
        "prompt_tokens":  r.get("prompt_eval_count", 0),
        "gen_tokens":     gen_tokens,
        "tokens_per_sec": round(gen_tokens / (gen_ns / 1e9), 1),
        "prompt_ms":      round(prompt_ns / 1e6, 1),
        "total_ms":       round(r.get("total_duration", 0) / 1e6, 1),
    }


def time_call(prompt: str, model: str = "llama3.2:latest") -> dict:
    """Run a single generate call and return timing metrics."""
    resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False})
    return _metrics(resp.json(), model)


def time_chat(messages: list, model: str = "llama3.2:latest",
              fresh: bool = False, options: dict = None) -> tuple:
    """Send a chat history, return (reply text, timing metrics).

    ``fresh=True`` evicts the model from memory first, so it arrives with
    no cached memory of the conversation and must read the transcript
    from scratch. ``options`` passes Ollama sampler/runtime settings
    through, e.g. ``{"num_ctx": 8192}`` to widen the context window past
    the 4,096-token default before a long transcript gets truncated.
    """
    if fresh:
        subprocess.run(["ollama", "stop", model], capture_output=True)
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    resp = requests.post(OLLAMA_CHAT_URL, json=payload)
    r = resp.json()
    return r["message"]["content"], _metrics(r, model)


def compare_models(prompt: str, models: list) -> list:
    """Return a sorted list of timing dicts for each model on the same prompt."""
    return sorted([time_call(prompt, m) for m in models], key=lambda x: -x["tokens_per_sec"])


def compare_task_ladder(ladder, small: str = "llama3.2:1b",
                        large: str = "llama3.2:latest") -> list:
    """Time a small vs large model on each task; print t/s and return chart rows."""
    rows = []
    for task, prompt in ladder:
        s = time_call(prompt, model=small)
        l = time_call(prompt, model=large)
        rows.append({"task": task, "small": s["tokens_per_sec"],
                     "large": l["tokens_per_sec"]})
        print(f"{task:9s} 1B {s['tokens_per_sec']:3.0f} t/s   "
              f"3B {l['tokens_per_sec']:3.0f} t/s")
    return rows


def show_batching_savings(ctx, qs) -> None:
    """Compare resending the context per question vs bundling it into one call."""
    import time
    t0, ptok = time.time(), 0
    for q in qs:
        ptok += time_call(ctx + "\n" + q)["prompt_tokens"]
    unbatched = time.time() - t0
    t0 = time.time()
    bat_tok = time_call(ctx + "\n" + " ".join(qs))["prompt_tokens"]
    print(f"unbatched {unbatched:5.1f}s  {ptok:3d} prompt tokens")
    print(f"batched   {time.time()-t0:5.1f}s  {bat_tok:3d} prompt tokens")


# Measured by scripts/_efficiency_moe_probe.py on an idle Mac (decode speed is the
# fastest of three runs per model, since GPU contention only ever slows a call).
# total_b/size_gb come from the Ollama API; active_b is the per-token count from
# each model's card (LFM2-24B-A2B fires ~2B of its 23.8B; Gemma 3n E4B keeps ~4B
# live; the two dense models activate everything). Sorted by active params, decode
# speed is monotonic — 1.1B->136, 2.0B->56, 3.2B->47, 4.0B->28 t/s — so the
# mixture-of-experts model runs at its 2B active speed, not its 23.8B stored size.
# fmt_answer is each model's real reply to a strict-format request; the dense floor
# ignores it and pads with chatter, the heavier models comply.
SPARSITY_STUDY = [
    {"model": "tinyllama:1.1b", "kind": "dense floor", "total_b": 1.1,
     "active_b": 1.1, "size_gb": 0.6, "tps": 136.2,
     "fmt_answer": "Sure! Here are the three fruits I mentioned earlier:\n\n"
                   "1. Apple\n2. Pear\n3. Plum"},
    {"model": "lfm2:24b", "kind": "mixture-of-experts", "total_b": 23.8,
     "active_b": 2.0, "size_gb": 14.4, "tps": 56.2,
     "fmt_answer": "apple,banana,orange"},
    {"model": "llama3.2:latest", "kind": "dense workhorse", "total_b": 3.2,
     "active_b": 3.2, "size_gb": 2.0, "tps": 46.7,
     "fmt_answer": "apple, banana, orange"},
    {"model": "gemma3n:e4b", "kind": "selective (PLE)", "total_b": 6.9,
     "active_b": 4.0, "size_gb": 7.5, "tps": 28.1,
     "fmt_answer": "apple, banana, orange"},
]


# Measured by scripts/_efficiency_effort_probe.py: gpt-oss:20b at reasoning effort
# low/medium/high over six tasks (two easy facts + the four NOVEL_PROBLEMS), temp 0.
# `bill` is the mean token count generated per answer (eval_count, reasoning trace
# plus visible answer); `gen_s` the mean generation time, which drifts with machine
# load while the token bill is stable. The finding: turning the dial up ~doubles
# the bill and the time but leaves accuracy flat at 6/6 — gpt-oss solves every task
# at low effort, so higher effort is pure cost. `example` is the snail-in-the-well
# problem, which returns "4" at every setting while its bill climbs 170 -> 321
# (the answer is one token; the rest is hidden reasoning you pay for but never read).
# A frontier hunt (10 reasoning traps, see the probe's `frontier` mode) found no
# clean task where low effort fails and high succeeds: gpt-oss is strong enough that
# effort almost never changes the answer, only the price.
EFFORT_STUDY = {
    "efforts": ["low", "medium", "high"],
    "bill":    [80, 127, 168],
    "gen_s":   [3.2, 5.2, 7.6],
    "correct": [6, 6, 6],
    "n_tasks": 6,
    "example": {
        "task": "snail-in-the-well",
        "question": "A snail is at the bottom of a 7-metre well. It climbs 4 metres "
                    "each day and slides back 3 metres each night. How many days does "
                    "it take to reach the top?",
        "answer": "4",
        "bill":  [170, 241, 321],
        "gen_s": [6.9, 9.9, 14.2],
    },
}

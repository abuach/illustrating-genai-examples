"""Zero-shot classification with typed answers and calibrated confidence.

A local take on "System One" classifiers like TypeSafe's Jev: hand over some
text and the labels you care about (no training), and get back one of those
labels plus a probability for every label, ready for code to branch on.

    from genai import classify

    c = classify("The export button crashes the app",
                 ["billing", "bug", "feature_request", "other"])
    c.label          # 'bug'
    c.confidence     # 0.98
    c.probs          # {'bug': 0.98, 'billing': 0.01, ...}

Two Ollama features do the work in a single call:
  * structured outputs: a JSON schema whose `enum` is the label list, so the
    model can only ever answer with one of your labels;
  * logprobs: the model's probability for each candidate token where the
    label is written, which becomes the probability of each label.

Ollama reports logprobs from the model's raw distribution, before the schema
constrains it, so the candidates at a step can include text that is not a
label at all. We keep only the candidates that spell (a prefix of) a label and
renormalize over the labels. Labels that share a prefix ("feature" and
"feature_request") are told apart at the token where they diverge.
"""
import json
import math
import re
import urllib.request
from dataclasses import dataclass

from genai import llm as _llm

CLASSIFY_MODEL = "qwen3.5:4b"
TOP_LOGPROBS   = 20


@dataclass
class Classification:
    """One typed decision: the chosen label, its probability, and the full
    distribution over every label (sums to 1)."""
    label:      str
    confidence: float
    probs:      dict

    def __str__(self) -> str:
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])
        return f"{self.label} ({self.confidence:.2f})  " + "  ".join(
            f"{k}={v:.2f}" for k, v in ranked)


def _chat(request: dict) -> dict:
    """POST to /api/chat on the current server (set_host() applies)."""
    req = urllib.request.Request(f"{_llm.SERVER}/api/chat",
                                 data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def _matches(text: str, label: str) -> bool:
    """Is ``text`` (the label value written so far, maybe including the closing
    quote) consistent with ``label``?"""
    if '"' in text:
        return text[:text.index('"')] == label
    return label.startswith(text)


def _label_probs(content: str, logprobs: list, labels: list,
                 chosen: str) -> dict:
    """Turn the per-token logprobs of the chosen answer into a probability for
    every label. Walk the tokens of the chosen label; at each step, any label
    that agreed with the path so far collects the probability of the
    candidates that spell it, times the probability of the path up to there."""
    m = re.search(r'"label"\s*:\s*"', content)
    if m is None:
        return {l: float(l == chosen) for l in labels}
    vstart = m.end()

    mass = dict.fromkeys(labels, 0.0)
    path_logp, pos = 0.0, 0
    for step in logprobs:
        tok = step["token"]
        start, pos = pos, pos + len(tok)
        if pos <= vstart:
            continue                         # still before the label value
        pre    = content[start:vstart] if start < vstart else ""
        prefix = content[vstart:start] if start > vstart else ""
        live   = [l for l in labels if l != chosen and l.startswith(prefix)
                  and mass[l] == 0.0]
        for cand in step.get("top_logprobs", []):
            ctok = cand["token"]
            if ctok == tok or not ctok.startswith(pre):
                continue
            cont = ctok[len(pre):]
            if not prefix:
                cont = cont.lstrip()         # ' bug' and 'bug' both count
            hits = [l for l in live if _matches(prefix + cont, l)]
            for l in hits:                   # indistinguishable here: share it
                mass[l] += math.exp(path_logp + cand["logprob"]) / len(hits)
        path_logp += step["logprob"]
        if '"' in content[max(start, vstart):pos]:
            break                            # the label's closing quote
    mass[chosen] = math.exp(path_logp)

    total = sum(mass.values())
    return {l: mass[l] / total for l in labels}


def classify(text:        str,
             labels:      list,
             model:       str = CLASSIFY_MODEL,
             instruction: str = "Classify the text.") -> Classification:
    """Pick one of ``labels`` for ``text`` and report how sure the model is.

    Args:
        text:        The input to classify.
        labels:      The answer space, chosen at call time (no training).
        model:       Any model on the Ollama server; small ones are fast
                     enough for routing and guardrail checks.
        instruction: What the decision is about, e.g. "Is this message spam?"
                     with labels ["spam", "not_spam"].
    """
    labels = list(dict.fromkeys(labels))
    system = (f"{instruction} Answer with JSON {{\"label\": ...}} using "
              f"exactly one of these labels: {', '.join(labels)}")
    resp = _chat({
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
        "format": {"type": "object",
                   "properties": {"label": {"type": "string", "enum": labels}},
                   "required": ["label"]},
        "options": {"temperature": 0},
        "logprobs": True, "top_logprobs": TOP_LOGPROBS,
    })
    content = resp["message"]["content"]
    chosen  = json.loads(content)["label"]
    probs   = _label_probs(content, resp.get("logprobs", []), labels, chosen)
    return Classification(chosen, probs[chosen], probs)

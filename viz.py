"""Visualization helpers for Programming Generative AI."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# ── Shared palette ────────────────────────────────────────────────────────────
# The book prints in four-color process, and screen colors do not all survive
# the trip to ink. Most of these only lose a little vividness, which is fine.
# PURPLE is the exception: the old #8B5CF6 violet converted to almost exactly
# the same ink as BLUE (their separation fell from 103 to 37), so any chart
# using both lost a series. This deep plum stays 108 away from every other
# color here after conversion. Check with: python scripts/check_cmyk.py --viz
BLUE, GREEN, ORANGE, RED = "#2563EB", "#10B981", "#F97316", "#EF4444"
LGRAY, MGRAY, DGRAY = "#F3F4F6", "#9CA3AF", "#111827"
PURPLE, TEAL, PINK, AMBER = "#821E6E", "#14B8A6", "#EC4899", "#F59E0B"
SLATE = "#475569"


# ── Model zoo: one stable color per contestant across every bake-off ──────────
# Keyed by family substring, checked in order so the specific name wins over the
# general one (functiongemma before gemma, qwen3.5 before qwen, ministral before
# mistral). The weak function-tuned tail is gray on purpose; gpt-oss, the heavy
# reasoning newcomer, gets a sober dark slate to set it apart from the family hues.
_MODEL_COLOR_RULES = [
    ("gpt-oss", SLATE),
    ("functiongemma", MGRAY), ("gemma4", GREEN), ("gemma3", AMBER),
    ("qwen3.5", TEAL), ("qwen2.5", TEAL), ("qwen", BLUE),
    ("ministral", PURPLE), ("mistral", ORANGE),
    ("llama3.2", PINK), ("llama", RED),
]


def short_model(name: str) -> str:
    """Drop the ``:latest`` tag for an axis label but keep size tags like ``:1b``."""
    return name[:-7] if name.endswith(":latest") else name


def model_color(name: str) -> str:
    """The stable bake-off color for a model, by family."""
    low = name.lower()
    for key, col in _MODEL_COLOR_RULES:
        if key in low:
            return col
    return DGRAY


# ── Internal helpers ──────────────────────────────────────────────────────────

def _arr(ax, x1, y1, x2, y2, col=MGRAY, pct=0.28, ls="solid"):
    dx, dy = x2 - x1, y2 - y1
    ax.annotate("", xy=(x2 - pct*dx, y2 - pct*dy), xytext=(x1 + pct*dx, y1 + pct*dy),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                mutation_scale=10, shrinkA=0, shrinkB=0, linestyle=ls))


def _fbox(ax, cx, cy, hw, hh, fc, ec="none", lw=1.0):
    ax.add_patch(mp.FancyBboxPatch((cx - hw, cy - hh), hw*2, hh*2,
                                   boxstyle="round,pad=0.07", facecolor=fc,
                                   edgecolor=ec, linewidth=lw, zorder=3))


def _label(ax, cx, cy, text, tc=DGRAY, fs=10.5, fw="700"):
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.2,
            fontsize=fs, fontweight=fw, color=tc, zorder=4)


def _grid_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=11)
    ax.yaxis.grid(True, color=LGRAY, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def _place_labels(ax, coords, labels, fontsize=10.5, color=None):
    """Place point labels using adjustText so they don't overlap.

    coords: (N, 2) array of point positions.
    labels: list of N strings.
    """
    from adjustText import adjust_text
    if color is None:
        color = DGRAY
    arr = np.asarray(coords)
    texts = [ax.text(arr[i, 0], arr[i, 1], lbl,
                     fontsize=fontsize, color=color)
             for i, lbl in enumerate(labels)]
    adjust_text(texts, ax=ax,
                expand=(1.15, 1.4),
                arrowprops=dict(arrowstyle='-', color=LGRAY, lw=0.6))
    return texts


# ── Existing helpers ──────────────────────────────────────────────────────────

def _attention_from(sentence, pronoun, layer, head):
    """Read one attention head: how much the ``pronoun`` token looks back at each
    word of ``sentence``. Returns (words, weights) with scaffold tokens dropped."""
    import torch
    from transformers import BertTokenizer, BertModel

    tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True,
                                      local_files_only=True).eval()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    row = out.attentions[layer][0, head, toks.index(pronoun)].numpy()
    words = [(t, w) for t, w in zip(toks, row) if t not in ("[CLS]", "[SEP]")]
    return [t for t, _ in words], np.array([w for _, w in words])


def plot_attention(sentences, pronoun: str = "it", layer: int = 9, head: int = 1,
                   path: str = None, figsize=(8.2, 4.4)):
    """Show where a pronoun looks back, for two sentences that differ by one word.

    For each sentence, read one attention head and plot how much the ``pronoun``
    token attends to every earlier word. The word it leans on hardest is the one
    the model ties the pronoun to, and swapping a single word flips it.
    """
    fig, axes = plt.subplots(len(sentences), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sentence in zip(axes, sentences):
        words, weights = _attention_from(sentence, pronoun, layer, head)
        top = int(np.argmax(weights))
        colors = [GREEN if i == top else LGRAY for i in range(len(words))]
        y = range(len(words))
        ax.barh(y, weights, color=colors, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(sentence, fontsize=10.5, fontweight="600", loc="left", color=DGRAY)
        ax.text(weights[top], top, f"  {words[top]}", va="center",
                fontsize=10, fontweight="700", color=GREEN)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0, colors=MGRAY)
    axes[-1].set_xlabel(f'attention from "{pronoun}" to each word', fontsize=10.5)
    plt.tight_layout()
    _save(fig, path)
    plt.show()


def attention_votes(sentence: str, word: str) -> tuple:
    """Let every BERT attention head vote for the word that ``word`` attends to
    most. Returns (words, votes) in sentence order.

    One head is one specialist; asking all 12 layers x 12 heads gives the
    consensus. BERT's [CLS]/[SEP] markers and punctuation are set aside, as is
    the word itself: heads park a lot of attention there, and none of it says
    which other word matters.
    """
    import torch
    from transformers import BertTokenizer, BertModel

    tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True,
                                      local_files_only=True).eval()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        att = model(**enc).attentions
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    me = toks.index(word)
    keep = [j for j, t in enumerate(toks)
            if j != me and t not in ("[CLS]", "[SEP]") and any(ch.isalnum() for ch in t)]
    votes = {j: 0 for j in keep}
    for layer in att:
        for head in layer[0]:
            votes[max(keep, key=lambda j: head[me, j])] += 1
    return [toks[j] for j in keep], np.array([votes[j] for j in keep])


def plot_attention_votes(sentence: str, word: str, path: str = None, figsize=(8.2, 3.2)):
    """Bar chart of attention_votes: how many of BERT's 144 heads look hardest
    at each word when reading ``word``. The winner is highlighted."""
    words, votes = attention_votes(sentence, word)
    top = int(np.argmax(votes))
    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(words))
    ax.barh(y, votes, color=[GREEN if i == top else LGRAY for i in y], zorder=3)
    for i, v in enumerate(votes):
        ax.text(v + 0.8, i, str(v), va="center", fontsize=10,
                color=GREEN if i == top else DGRAY, fontweight="700" if i == top else "normal")
    ax.set_yticks(y); ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(sentence, fontsize=10.5, fontweight="600", loc="left", color=DGRAY)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, colors=MGRAY)
    ax.set_xlabel(f'heads (of {int(votes.sum())}) that attend most to each word from "{word}"',
                  fontsize=10.5)
    plt.tight_layout()
    _save(fig, path)
    plt.show()


def plot_embeddings_2d(words: list, vecs,
                       figsize=(6, 6), title: str = "Word Embeddings (PCA)"):
    """Project high-dimensional embeddings to 2D with PCA and scatter-plot them."""
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(vecs)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], s=80)
    for word, (x, y) in zip(words, coords):
        ax.annotate(word, (x, y), fontsize=14, textcoords="offset points", xytext=(4, 4))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ── Agentic chapter ───────────────────────────────────────────────────────────

def plot_agentic_loop(save_path="images/agentic/agentic_loop.png"):
    """Architecture diagram of the sense-plan-act agentic loop."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0.8, 5.2); ax.axis("off")

    nodes = {
        "goal":  (1.4, 3.0, "Goal",            BLUE,  "white"),
        "plan":  (3.6, 4.4, "Plan\n(Planner)", LGRAY, DGRAY),
        "tool":  (6.4, 4.4, "Act\n(Executor)", LGRAY, DGRAY),
        "world": (8.6, 3.0, "World\n(State)",  GREEN, "white"),
        "obs":   (6.4, 1.6, "Observe",          LGRAY, DGRAY),
        "check": (3.6, 1.6, "Goal\nmet?",      LGRAY, DGRAY),
    }
    for key, (x, y, label, fc, tc) in nodes.items():
        hw, hh = (0.90, 0.55) if "\n" in label else (0.82, 0.38)
        _fbox(ax, x, y, hw, hh, fc,
              ec=fc if fc in (BLUE, GREEN) else BLUE,
              lw=0 if fc in (BLUE, GREEN) else 1.0)
        _label(ax, x, y, label, tc=tc)

    g = nodes
    _arr(ax, *g["goal"][:2],  *g["plan"][:2])
    _arr(ax, *g["plan"][:2],  *g["tool"][:2])
    _arr(ax, *g["tool"][:2],  *g["world"][:2])
    _arr(ax, *g["world"][:2], *g["obs"][:2])
    _arr(ax, *g["obs"][:2],   *g["check"][:2])
    _arr(ax, *g["check"][:2], *g["plan"][:2], col=ORANGE, ls="dashed")

    ax.text(2.8, 2.50, "not yet", color=ORANGE, fontsize=9.5,
            style="italic", ha="center")
    ax.set_title("The Agentic Loop", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_system_anatomy(save_path="images/architecture/system_anatomy.png"):
    """The model as one component inside a larger, deterministic system.

    A single coloured box (the model) sits in a ring of grey boxes (memory,
    tools, guardrails, orchestrator) that are ordinary code the engineer owns.
    The first figure of Engineering Intelligent Systems.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    # the system boundary
    ax.add_patch(mp.FancyBboxPatch((0.5, 0.95), 9.0, 4.95,
                 boxstyle="round,pad=0.10", facecolor="none", edgecolor=MGRAY,
                 linewidth=1.0, linestyle=(0, (5, 4)), zorder=1))
    ax.text(1.0, 5.55, "the system", color=MGRAY, fontsize=9.5,
            style="italic", ha="left")

    # the model: the one non-deterministic component
    _fbox(ax, 5.0, 3.05, 1.25, 0.62, ORANGE)
    _label(ax, 5.0, 3.05, "Model\n(reasoning)", tc="white")

    # deterministic components: ordinary code the engineer owns
    det = {"orch": (5.0, 4.65, "Orchestrator\n(control flow)"),
           "mem":  (2.1, 3.05, "Memory"),
           "tools": (7.9, 3.05, "Tools"),
           "guard": (5.0, 1.55, "Guardrails\n(in / out)")}
    for x, y, lab in det.values():
        _fbox(ax, x, y, 1.05, 0.55 if "\n" in lab else 0.40, LGRAY, ec=BLUE, lw=1.0)
        _label(ax, x, y, lab, tc=DGRAY)

    _arr(ax, 5.0, 4.10, 5.0, 3.67)        # orchestrator -> model
    _arr(ax, 3.15, 3.05, 3.75, 3.05)      # memory -> model
    _arr(ax, 6.25, 3.05, 6.85, 3.05)      # model -> tools
    _arr(ax, 5.0, 2.43, 5.0, 2.10)        # model -> guardrails
    _arr(ax, 3.75, 2.80, 3.15, 2.80, ls="dashed")   # model -> memory (write back)
    _arr(ax, 6.85, 3.30, 6.25, 3.30, ls="dashed")   # tools -> model (results)

    ax.add_patch(mp.Rectangle((1.05, 0.30), 0.26, 0.26, color=ORANGE))
    ax.text(1.42, 0.43, "the model (non-deterministic)", fontsize=8.5,
            va="center", color=DGRAY)
    ax.add_patch(mp.Rectangle((5.55, 0.30), 0.26, 0.26, color=LGRAY,
                              ec=BLUE, lw=1.0))
    ax.text(5.92, 0.43, "code you own (deterministic)", fontsize=8.5,
            va="center", color=DGRAY)

    ax.set_title("A model is one component", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_neurosymbolic(save_path="images/reasoning/neurosymbolic.png"):
    """The neuro-symbolic split: the model translates, the solver deduces."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    ax.text(0.7, 3.4, "problem\n(in words)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.7, 3.4, 2.55, 3.4)

    _fbox(ax, 4.0, 3.4, 1.35, 0.78, ORANGE)
    _label(ax, 4.0, 3.4, "Model\ntranslates", tc="white", fs=10.5)
    ax.text(4.0, 1.75, "fluent, can hallucinate", ha="center", fontsize=8.8,
            style="italic", color=MGRAY)

    _arr(ax, 5.35, 3.4, 6.55, 3.4)
    ax.text(5.95, 3.78, "constraints", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 8.0, 3.4, 1.35, 0.78, TEAL)
    _label(ax, 8.0, 3.4, "Solver\ndeduces", tc="white", fs=10.5)
    ax.text(8.0, 1.75, "exact, can't", ha="center", fontsize=8.8,
            style="italic", color=MGRAY)

    _arr(ax, 9.35, 3.4, 10.2, 3.4)
    ax.text(11.2, 3.4, "answer\n(grounded)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)

    ax.set_title("The model translates, the solver reasons", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_orchestration_models(save_path="images/frameworks/orchestration_models.png"):
    """Three ways to wire agents: a graph of steps, a manager with roles, a chain."""
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_facecolor("white")

    def box(ax, x, y, label):
        _fbox(ax, x, y, 1.2, 0.62, LGRAY, ec=BLUE, lw=1.0)
        _label(ax, x, y, label, tc=DGRAY, fs=10)

    def sub(ax, title, caption):
        ax.set_title(title, fontsize=12.5, fontweight="700", color=DGRAY, pad=2)
        ax.text(5, 0.5, caption, ha="center", fontsize=8.8, color=MGRAY, linespacing=1.3)

    g = axes[0]
    box(g, 5, 8.4, "step"); box(g, 5, 5.4, "step"); box(g, 5, 2.4, "step")
    box(g, 8.3, 5.4, "branch")
    _arr(g, 5, 7.7, 5, 6.1); _arr(g, 5, 4.7, 5, 3.1)
    _arr(g, 6.2, 5.4, 7.0, 5.4)
    _arr(g, 8.3, 4.7, 5.9, 3.0, ls="dashed")
    sub(g, "graph", "steps and the edges\nbetween them")

    r = axes[1]
    box(r, 5, 8.4, "manager")
    box(r, 2.6, 3.6, "agent"); box(r, 7.4, 3.6, "agent")
    _arr(r, 4.1, 7.8, 3.1, 4.3); _arr(r, 5.9, 7.8, 6.9, 4.3)
    sub(r, "role", "a manager hands work\nto specialists")

    h = axes[2]
    box(h, 2.1, 5.4, "agent"); box(h, 5, 5.4, "agent"); box(h, 7.9, 5.4, "agent")
    _arr(h, 3.35, 5.4, 3.75, 5.4); _arr(h, 6.25, 5.4, 6.65, 5.4)
    sub(h, "handoff", "each agent passes\ncontrol to the next")

    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_mcp_anatomy(save_path="images/mcp/mcp_anatomy.png"):
    """Client/server shape of MCP: an app's client reaches a server's primitives."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # the application (host) holds the model and an MCP client
    ax.add_patch(mp.FancyBboxPatch((0.4, 1.0), 3.6, 3.9, boxstyle="round,pad=0.10",
                 facecolor="none", edgecolor=MGRAY, linewidth=1.0,
                 linestyle=(0, (5, 4)), zorder=1))
    ax.text(0.7, 4.55, "your application", color=MGRAY, fontsize=9.5, style="italic")
    _fbox(ax, 2.2, 3.55, 1.15, 0.5, ORANGE); _label(ax, 2.2, 3.55, "Model", tc="white")
    _fbox(ax, 2.2, 2.05, 1.2, 0.5, LGRAY, ec=BLUE, lw=1.0)
    _label(ax, 2.2, 2.05, "MCP Client", tc=DGRAY, fs=10)
    _arr(ax, 2.55, 3.15, 2.55, 2.55)
    _arr(ax, 1.85, 2.55, 1.85, 3.15)

    # the server advertises three kinds of thing
    ax.add_patch(mp.FancyBboxPatch((6.0, 0.8), 3.5, 4.3, boxstyle="round,pad=0.10",
                 facecolor="none", edgecolor=BLUE, linewidth=1.3, zorder=1))
    ax.text(7.75, 4.65, "MCP Server", color=DGRAY, fontsize=11.5,
            fontweight="700", ha="center")
    for i, lab in enumerate(["Tools", "Resources", "Prompts"]):
        _fbox(ax, 7.75, 3.65 - i * 1.05, 1.35, 0.37, LGRAY, ec=BLUE, lw=1.0)
        _label(ax, 7.75, 3.65 - i * 1.05, lab, tc=DGRAY, fs=10.5)

    # the protocol: one wire, spoken by every framework
    _arr(ax, 3.5, 2.25, 6.0, 2.6, col=ORANGE)
    _arr(ax, 6.0, 1.95, 3.5, 1.7, col=ORANGE, ls="dashed")
    ax.text(4.75, 2.62, "MCP", color=ORANGE, fontsize=10.5, fontweight="700", ha="center")

    ax.set_title("One protocol, any tool", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_control_ownership(study, save_path="images/architecture/control_ownership.png"):
    """Reliability and latency when the model owns the loop vs when code does."""
    md, cd = study["model_driven"], study["code_driven"]
    labels = ["model owns\nthe loop", "code owns\nthe loop"]
    colors = [ORANGE, BLUE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.0))
    fig.patch.set_facecolor("white")

    accs = [md["accuracy"] * 100, cd["accuracy"] * 100]
    ax1.bar(labels, accs, color=colors, width=0.62, zorder=3)
    for i, v in enumerate(accs):
        ax1.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=12,
                 fontweight="700", color=DGRAY)
    ax1.set_ylim(0, 112); ax1.set_ylabel("orders totalled correctly")
    ax1.set_title("Reliability", fontsize=12, fontweight="600", color=DGRAY)

    lats = [md["mean_secs"], cd["mean_secs"]]
    ax2.bar(labels, lats, color=colors, width=0.62, zorder=3)
    for i, v in enumerate(lats):
        ax2.text(i, v + max(lats) * 0.03, f"{v:.1f}s", ha="center", fontsize=12,
                 fontweight="700", color=DGRAY)
    ax2.set_ylim(0, max(lats) * 1.18); ax2.set_ylabel("seconds per order")
    ax2.set_title("Latency", fontsize=12, fontweight="600", color=DGRAY)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=LGRAY, zorder=0)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_quality_attributes(study, save_path="images/architecture/quality_attributes.png"):
    """Three quality attributes read off the same two register designs:
    reliability (runs totalled correctly), cost (tokens per order), and
    consistency (distinct totals the same order produced)."""
    md, cd = study["model_driven"], study["code_driven"]
    labels = ["model owns\nthe loop", "code owns\nthe loop"]
    colors = [ORANGE, BLUE]
    trials = study.get("trials", 12)
    panels = [
        ("Reliability", "runs totalled correctly",
         [md["reliability"] * 100, cd["reliability"] * 100], "{:.0f}%", 112),
        ("Cost", "tokens per order",
         [md["tokens_per_order"], cd["tokens_per_order"]], "{:,.0f}", None),
        ("Consistency", f"different results in {trials} runs",
         [md["distinct_answers"], cd["distinct_answers"]], "{:.0f}", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    fig.patch.set_facecolor("white")
    for ax, (title, ylabel, vals, fmt, ylim) in zip(axes, panels):
        ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
        top = ylim or max(vals) * 1.18
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.025, fmt.format(v), ha="center",
                    fontsize=12, fontweight="700", color=DGRAY)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="600", color=DGRAY)
        ax.set_facecolor("white")
        if title == "Consistency":     # a count of outcomes: whole ticks only
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=LGRAY, zorder=0)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_pattern_scorecard(cascade_study,
                           save_path="images/architecture/pattern_scorecard.png"):
    """Well-formed vs. right, read off the cascade study's three arms. Each design
    reads the same fifteen orders; the slate bar counts the receipts that pass the
    schema (well-formed), the teal bar the receipts that price to the true total
    (right). The cheap model is mostly neither; escalating on a rejection buys
    almost all of the big model's correctness at half its token bill; the gap the
    cascade still leaves, valid but wrong, is the one only the checker closes."""
    arms = cascade_study["arms"]
    n = cascade_study["n"]
    tok = cascade_study["big_tokens_per_order"]
    groups = [("cheap alone\n(gemma3:1b)", arms["cheap"]),
              ("cascade\n(cheap → big)", arms["cascade"]),
              ("big alone\n(gemma4)", arms["big"])]
    x = np.arange(len(groups))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    valid = [g[1]["valid"] for g in groups]
    right = [g[1]["correct"] for g in groups]
    b1 = ax.bar(x - w/2, valid, w, color=SLATE, zorder=3,
                label="well-formed (passes the schema)")
    b2 = ax.bar(x + w/2, right, w, color=TEAL, zorder=3,
                label="right (prices to the true total)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                    f"{int(bar.get_height())}", ha="center", va="bottom",
                    fontsize=11.5, fontweight="700", color=DGRAY)
    # Name the gap the cascade still leaves: valid but wrong, what the checker is for.
    gap_top = arms["cascade"]["valid"]
    ax.annotate("", xy=(1 - w/2, gap_top), xytext=(1 + w/2, arms["cascade"]["correct"]),
                arrowprops=dict(arrowstyle="-", color=MGRAY, lw=1.0, ls=(0, (2, 2))))
    ax.text(1, gap_top + 1.05, "valid but wrong\n→ the checker's job", ha="center",
            va="bottom", fontsize=9.5, fontweight="600", color=SLATE, linespacing=1.15)
    # The economy line: escalate-on-rejection vs always-big token bill.
    ax.text(0.5, -0.22, f"big-model tokens per order:  cascade {tok['cascade']}  "
            f"vs  always-big {tok['always_big']}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.5, color=DGRAY)
    ax.set_ylabel(f"receipts (of {n})", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, n + 3)
    ax.set_title("Well-Formed vs. Right", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()


def plot_tool_bench(bench: dict, save_path="images/agentic/tool_bench.png"):
    """Dual-bar chart: tool-call accuracy and latency across models.

    bench = {"model_name": {"score_pct": int, "latency_s": float}, ...}
    """
    keys   = list(bench.keys())
    labels = [short_model(k) for k in keys]
    scores = [bench[k]["score_pct"] for k in keys]
    lats   = [bench[k]["latency_s"] for k in keys]
    colors = [model_color(k) for k in keys]
    x      = np.arange(len(keys))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)

    for bars, vals, ax, ylabel, title, ylim, fmt in [
        (ax1.bar(x, scores, color=colors, width=0.55, zorder=3), scores, ax1,
         "accuracy (%)", "Tool-Call Accuracy", 115, "{:.0f}%"),
        (ax2.bar(x, lats,   color=colors, width=0.55, zorder=3), lats,   ax2,
         "avg latency (s)", "Latency per Call", None, "{:.1f}s"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (1.5 if "%" in ylabel else 0.1),
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=11, fontweight="600", color=DGRAY)
        if ylim:
            ax.set_ylim(0, ylim)

    fig.suptitle("Tool-Calling Model Comparison", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_bakeoff(study, solid_key, faded_key, solid_label, faded_label,
                 title, ylabel, save_path, ylim=128, fmt="{:.0f}",
                 full_labels=False, solid_first=True):
    """Reusable model-zoo bake-off: per-model paired bars comparing two
    conditions. The solid bar is the headline/better condition; the faded,
    hatched bar is the baseline. Each model keeps its zoo color. By default the
    solid bar is drawn left; pass ``solid_first=False`` to draw the faded
    baseline left and the solid bar right, so a before-then-after pair reads
    left to right. With ``full_labels`` the axis keeps the ``:latest`` tag, so a
    roster that mixes size variants of one family (``gemma4:latest`` vs
    ``gemma4:e2b``) reads unambiguously.

    study = {"model": {solid_key: pct, faded_key: pct}, ...}
    """
    keys   = list(study.keys())
    labels = [k if full_labels else short_model(k) for k in keys]
    solid  = [study[k][solid_key] for k in keys]
    faded  = [study[k][faded_key] for k in keys]
    colors = [model_color(k) for k in keys]
    x = np.arange(len(keys))
    n = len(keys)
    # A big roster (the reflexion field grew to fourteen models) needs more
    # canvas, steeper labels, and smaller value text or the pairs collide. Small
    # boards (<=10) keep the original geometry, so their output is unchanged.
    crowded = n > 10
    w = 0.34 if crowded else 0.38
    fig_w = max(9.5, 0.92 * n)
    vfont = 8.0 if crowded else 9.5
    rot   = 28 if crowded else 15
    solid_off = -w/2 if solid_first else w/2     # which side each bar sits on
    faded_off = -solid_off

    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    b1 = ax.bar(x + solid_off, solid, width=w, color=colors, zorder=3)
    b2 = ax.bar(x + faded_off, faded, width=w, color=colors, zorder=3,
                alpha=0.4, hatch="////", edgecolor="white")
    for bars, vals in [(b1, solid), (b2, faded)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ylim*0.012,
                    fmt.format(v), ha="center", va="bottom", fontsize=vfont,
                    fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=10)
    sp = mp.Patch(facecolor=DGRAY, label=solid_label)
    fp = mp.Patch(facecolor=DGRAY, alpha=0.4, hatch="////", label=faded_label)
    # Sit the legend in the headroom band above the tallest bar, centered, so it
    # never lands on a bar regardless of which model is tallest. Order the legend
    # to match the on-page bar order (left entry = left bar).
    handles = [sp, fp] if solid_first else [fp, sp]
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
              fontsize=10, columnspacing=1.4, handlelength=1.4)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_tool_identity(study: dict, save_path="images/agentic/tool_identity.png"):
    """Per-model tool call-rate across the four label conditions, as a heatmap.
    ``study`` is ``{model: {"clear_clear", "clear_vague", "vague_clear",
    "vague_vague"}}`` keyed name_desc. Columns run both labels clear, name only,
    description only, both vague. Rows are sorted by lean (name-only minus
    description-only), so models that read the name sit at the top and models that
    read the description at the bottom; the two middle columns are where the field
    splits, with the same crossover the showdown then shows on a single request."""
    cols    = ["clear_clear", "clear_vague", "vague_clear", "vague_vague"]
    headers = ["both\nclear", "name\nonly", "description\nonly", "both\nvague"]
    models  = sorted(study, reverse=True,
                     key=lambda m: study[m]["clear_vague"] - study[m]["vague_clear"])
    grid = np.array([[study[m][c] for c in cols] for m in models])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("RdYlGn").copy()
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(len(cols)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                    fontweight="700", color="white" if v < 28 else DGRAY)

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(headers, fontsize=10.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_model(m) for m in models], fontsize=10.5)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for x in (0.5, 1.5, 2.5):                       # thin gutters between columns
        ax.axvline(x, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which Label Does Each Model Read?  (tool call-rate, %)",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=24)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_tool_identity_summary(study: dict,
                               save_path="images/agentic/tool_identity_summary.png"):
    """The per-model heatmap rolled into one 2x2: mean tool call-rate across models,
    tool name (clear/vague) on the rows against description (clear/vague) on the
    columns. Both-clear is hot, both-vague is the cliff, and the two off-diagonal
    cells stay warm: either label clear on its own keeps the tool alive."""
    keys = ["clear_clear", "clear_vague", "vague_clear", "vague_vague"]
    mean = {k: sum(s[k] for s in study.values()) / len(study) for k in keys}
    # rows = name (clear, vague); cols = description (clear, vague)
    grid = np.array([[mean["clear_clear"], mean["clear_vague"]],
                     [mean["vague_clear"], mean["vague_vague"]]])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("RdYlGn").copy()
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=30,
                    fontweight="800", color="white" if v < 28 else DGRAY)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["clear\n“…in the book’s glossary”",
                        "vague\n“returns a string”"], fontsize=10.5)
    ax.set_yticklabels(["clear\nlookup(term)", "vague\nfn(x)"],
                       fontsize=10.5, rotation=90, va="center")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("tool description", fontsize=12, fontweight="700",
                  color=DGRAY, labelpad=10)
    ax.set_ylabel("tool name", fontsize=12, fontweight="700", color=DGRAY,
                  labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Both Labels Matter: How Often the Model Reaches for the Tool",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=42)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_reflexion_bakeoff(study, save_path="images/agentic/reflexion_bakeoff.png"):
    """Final gotcha-suite pass-rate, reflexion vs blind retry, per model. The spread
    up from the faded blind bar to the solid reflexion bar is the lift a model
    gets from writing itself a lesson; a flat pair means no lift (already aces it,
    or too weak to write a useful lesson)."""
    plot_bakeoff(study, "reflexion", "blind", "reflexion", "blind retry",
                 "Who Learns From Their Own Mistakes?",
                 "gotcha tasks solved (%)", save_path, solid_first=False)


def plot_confidence(study, save_path="images/agentic/confidence.png"):
    """Mean self-confidence (1-5) on answerable vs impossible questions, per model.
    The spread from the faded impossible bar up to the solid answerable bar is
    calibration: a wide spread means the model knows when it is guessing."""
    plot_bakeoff(study, "answerable", "impossible",
                 "answerable questions", "impossible questions",
                 "Who Knows What They Don't Know?",
                 "mean self-confidence (1-5)", save_path, ylim=6.4, fmt="{:.1f}")


# ── Lost-in-the-middle (reuse plot_bakeoff) ──────────────────────────────────

def plot_context(study, save_path="images/augmentation/context.png"):
    """Per model: recall of a fact planted at the start, middle and end of a long
    context, one bar each.

    Three bars rather than two on purpose. Averaging start and end into a single
    "edges" bar reported a flat 50% for nearly every model, which read as
    "sometimes finds it near either edge" and was really "always at the end,
    never at the start". Keeping the positions apart is the whole finding.
    """
    models = list(study)
    x = np.arange(len(models))
    w = 0.26
    bars = [("start", -w, BLUE, None), ("middle", 0.0, MGRAY, "///"),
            ("end", w, GREEN, None)]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    fig.patch.set_facecolor("white"); _grid_ax(ax)
    for key, off, col, hatch in bars:
        vals = [study[m][key] for m in models]
        ax.bar(x + off, vals, w, color=col, hatch=hatch, zorder=3,
               label=f"fact at the {key}" if key != "middle" else "fact in the middle")
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 3, f"{v:.0f}", ha="center", fontsize=9,
                    fontweight="600", color=DGRAY if v else MGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], fontsize=10, color=DGRAY,
                       rotation=18, ha="right")
    ax.set_ylabel("recall (%)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
    ax.set_title("Only the End of a Long Context Survives",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_index_speed(rows, save_path="images/augmentation/index_speed.png"):
    """Time for one search as the corpus grows: the naive linear scan (orange)
    climbs with the passage count while Chroma's HNSW index (blue) stays flat.
    Corpus sizes are spaced evenly on the x-axis so they read as a ladder."""
    sizes  = [r["n"] for r in rows]
    naive  = [r["naive_ms"] for r in rows]
    chroma = [r["chroma_ms"] for r in rows]
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(x, naive, "-o", color=ORANGE, lw=2.4, ms=8, zorder=3,
            label="naive linear scan (DocumentStore)")
    ax.plot(x, chroma, "-o", color=BLUE, lw=2.4, ms=8, zorder=3,
            label="Chroma (HNSW index)")
    for xi, v in zip(x, naive):
        ax.annotate(f"{v:.0f} ms", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10,
                    fontweight="600", color=ORANGE)
    ax.annotate(f"{rows[-1]['speedup']:.0f}× faster", (x[-1], chroma[-1]),
                textcoords="offset points", xytext=(-10, 18), ha="right",
                fontsize=11, fontweight="700", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("passages in the corpus", fontsize=11.5, color=DGRAY)
    ax.set_ylabel("time for one search (ms)", fontsize=11.5, color=DGRAY)
    ax.set_ylim(0, max(naive) * 1.2)
    ax.set_title("Linear Scan vs. Indexed Search",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_scoreboard(study, value_key, title, ylabel, save_path,
                    ylim=115, fmt="{:.0f}"):
    """One bar per model on a single metric, each in its zoo color. Pass a study
    of ``{model: {value_key: number}}`` (or ``{model: number}``); the caller
    controls ordering, so sort the dict for a ranked board."""
    keys   = list(study.keys())
    labels = [short_model(k) for k in keys]
    vals   = [study[k][value_key] if isinstance(study[k], dict) else study[k]
              for k in keys]
    colors = [model_color(k) for k in keys]
    x = np.arange(len(keys))
    # A big roster (the full fourteen-model field) needs more canvas and steeper
    # labels; boards of ten or fewer keep the original geometry, so their output
    # is unchanged.
    crowded = len(keys) > 10
    fig_w = max(8.5, 0.92 * len(keys))
    rot   = 28 if crowded else 15

    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ylim*0.013,
                fmt.format(v), ha="center", va="bottom", fontsize=10.5,
                fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=11, color=MGRAY)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_agent_landscape(save_path="images/agentic/agent_landscape.png"):
    """Timeline of agent research milestones, one per capability theme.

    Each marker is a real, dated body of work; the years are publication dates,
    not measured quantities. This is the survey spine for the chapter intro.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # (year, name, capability, theme color)
    milestones = [
        ("2022", "ReAct",              "reason + act in a loop", BLUE),
        ("2023", "Toolformer · Gorilla", "tool use and APIs",    GREEN),
        ("2023", "Generative Agents",  "human-like memory",      PURPLE),
        ("2023", "AutoGen · MetaGPT",  "teams of agents",        ORANGE),
        ("2024", "SWE-agent · Computer Use", "acting in the world", BLUE),
        ("2025", "MCP · research agents", "standardize and scale", GREEN),
    ]
    n = len(milestones)
    xs = np.linspace(0.9, 9.1, n)
    y0 = 2.5

    # baseline arrow of the timeline
    ax.annotate("", xy=(9.6, y0), xytext=(0.4, y0),
                arrowprops=dict(arrowstyle="-|>", color=MGRAY, lw=1.6,
                                mutation_scale=14, shrinkA=0, shrinkB=0))

    for i, (x, (year, name, cap, col)) in enumerate(zip(xs, milestones)):
        above = i % 2 == 0
        ly = y0 + 1.45 if above else y0 - 1.45
        # connector + dot on the timeline
        ax.plot([x, x], [y0, ly + (-0.42 if above else 0.42)],
                color=col, lw=1.2, zorder=2)
        ax.scatter([x], [y0], s=120, color=col, zorder=4,
                   edgecolors="white", linewidths=1.5)
        # label block
        _label(ax, x, ly + (0.12 if above else 0.12), name, tc=DGRAY, fs=9.5)
        ax.text(x, ly + (-0.22 if above else -0.22), cap, ha="center", va="center",
                fontsize=8, color=MGRAY, style="italic", zorder=4)
        ax.text(x + 0.34, y0 + (0.30 if above else -0.30), year, ha="left", va="center",
                fontsize=9, fontweight="700", color=col, zorder=4)

    ax.set_title("Five Years of Agent Research, One Capability at a Time",
                 fontsize=12, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_skill_retrieval(study: list, save_path="images/agentic/skill_retrieval.png"):
    """Per new task, the embedding similarity to the skill it should reuse vs the
    nearest distractor in the library. Where the right skill stands clear the
    library pays off and reuse is reliable; where a lexical neighbour crowds in,
    retrieval grabs the wrong skill (the "main idea" task lands on preview, not
    summarize). ``study`` is the row list from ``skill_retrieval_study``."""
    from matplotlib.patches import Patch
    labels = [r["label"]     for r in study]
    right  = [r["want_sim"]  for r in study]
    other  = [r["other_sim"] for r in study]
    x = np.arange(len(study)); w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.bar(x - w/2, right, w, color=GREEN, zorder=3)
    ax.bar(x + w/2, other, w, zorder=3,
           color=[MGRAY if r["hit"] else RED for r in study])
    for xi, r, o in zip(x, right, other):
        ax.text(xi - w/2, r + 0.015, f"{r:.2f}", ha="center", va="bottom",
                fontsize=9, color=DGRAY)
        ax.text(xi + w/2, o + 0.015, f"{o:.2f}", ha="center", va="bottom",
                fontsize=9, color=DGRAY)
    ax.legend(handles=[Patch(color=GREEN, label="skill to reuse"),
                       Patch(color=MGRAY, label="nearest distractor"),
                       Patch(color=RED,   label="distractor wins (miss)")],
              fontsize=10, framealpha=0, loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY)
    ax.set_ylabel("similarity to the new task", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 1.0)
    ax.set_title("A Growing Skill Library: Does the Right Skill Come Back?",
                 fontsize=13, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


# ── Augmentation chapter ──────────────────────────────────────────────────────

def plot_rag_pipeline(save_path="images/augmentation/rag_pipeline.png"):
    """Two-row RAG pipeline: indexing phase and query phase."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")

    def box(cx, cy, label, fc, tc, hw=0.90, hh=0.50):
        _fbox(ax, cx, cy, hw, hh, fc,
              ec=fc if fc in (BLUE, GREEN) else BLUE,
              lw=0 if fc in (BLUE, GREEN) else 1.0)
        _label(ax, cx, cy, label, tc=tc)

    index_items = [
        (1.3, 3.9, "Raw\nDocs",     LGRAY, DGRAY),
        (3.2, 3.9, "Clean &\nChunk", LGRAY, DGRAY),
        (5.1, 3.9, "Embed",          LGRAY, DGRAY),
        (7.0, 3.9, "Vector\nIndex",  BLUE,  "white"),
    ]
    query_items = [
        (1.3, 1.7, "User\nQuery",    GREEN,  "white"),
        (3.2, 1.7, "Retrieve\nTop-K",LGRAY, DGRAY),
        (5.1, 1.7, "Context +\nPrompt", LGRAY, DGRAY),
        (7.0, 1.7, "Generate",        LGRAY, DGRAY),
        (9.0, 1.7, "Cited\nAnswer",  GREEN,  "white"),
    ]

    for cx, cy, lbl, fc, tc in index_items + query_items:
        box(cx, cy, lbl, fc, tc)

    for items in (index_items, query_items):
        for i in range(len(items) - 1):
            _arr(ax, items[i][0], items[i][1], items[i+1][0], items[i+1][1], pct=0.30)

    _arr(ax, 7.0, 3.9 - 0.50, 3.2, 1.7 + 0.50, col=ORANGE, pct=0.12)
    # Caption tucked into the clear space below the diagonal arrow (the arrow runs
    # ~y3.0 across this x-span, the query boxes top out at y2.2).
    ax.text(5.7, 2.45, "retrieves from", color=ORANGE, fontsize=9.5,
            ha="center", style="italic")
    ax.text(4.15, 4.78, "Indexing Phase",
            fontsize=11, fontweight="700", color=BLUE, ha="center")
    ax.text(5.15, 0.88, "Query Phase",
            fontsize=11, fontweight="700", color=GREEN, ha="center")
    ax.set_title("Retrieval-Augmented Generation Pipeline",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_chunk_size(sizes: list, n_chunks: list, sims: list,
                    save_path="images/augmentation/chunk_size.png"):
    """Side-by-side bars: chunk count and top-1 similarity vs chunk size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)

    labels = [str(s) for s in sizes]
    ax1.bar(labels, n_chunks, color=BLUE, width=0.5, zorder=3)
    ax1.set_xlabel("chunk size (words)", fontsize=11, color=MGRAY)
    ax1.set_ylabel("chunks produced", fontsize=11, color=MGRAY)
    ax1.set_title("Chunks Produced", fontsize=14, fontweight="600", color=DGRAY, pad=6)
    max_chunks = max(n_chunks) if n_chunks else 1
    ax1.set_ylim(0, max_chunks * 1.25)
    for i, v in enumerate(n_chunks):
        ax1.text(i, v + max_chunks * 0.04, str(v),
                 ha="center", fontsize=11, fontweight="600", color=DGRAY)

    ax2.plot(labels, sims, color=GREEN, marker="o", linewidth=2, markersize=7, zorder=3)
    ax2.set_xlabel("chunk size (words)", fontsize=11, color=MGRAY)
    ax2.set_ylabel("top-1 similarity to query", fontsize=11, color=MGRAY)
    ax2.set_title("Retrieval Quality vs. Chunk Size",
                  fontsize=14, fontweight="600", color=DGRAY, pad=6)
    ax2.set_ylim(0, 1.05)

    plt.suptitle("Chunk Size Tradeoffs", fontsize=14, fontweight="600", color=DGRAY)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_embedding_space(labels: list, vecs, colors: list,
                         group_names: list = ("Prompting", "Multimodal"),
                         title: str = "Book Passage Embeddings (PCA to 2D)",
                         save_path="images/augmentation/embedding_space.png"):
    """PCA scatter of embedding vectors, colored by group.
    Labels for nearby points are stacked vertically so they don't overlap.
    """
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    coords = PCA(n_components=2).fit_transform(np.array(vecs))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10)

    for (x, y), col in zip(coords, colors):
        ax.scatter(x, y, color=col, s=90, zorder=4)
    _place_labels(ax, coords, labels, fontsize=10.5)

    unique_cols = list(dict.fromkeys(colors))
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               ms=9, label=n)
        for c, n in zip(unique_cols, list(group_names)[:len(unique_cols)])
    ], fontsize=11, framealpha=0)
    ax.set_title(title, fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_retrieval_comparison(labels: list, dense: list, sparse: list, hybrid: list,
                              k: int = 3,
                              save_path="images/augmentation/retrieval_comparison.png"):
    """Grouped bar chart comparing Dense, Sparse, and Hybrid retrieval precision."""
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)

    for i, (vals, lbl, col) in enumerate([(dense, "Dense", BLUE),
                                           (sparse, "Sparse", ORANGE),
                                           (hybrid, "Hybrid", GREEN)]):
        bars = ax.bar(x + (i - 1)*w, vals, w*0.9, label=lbl, color=col, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=10, color=DGRAY)

    # Mark groups where every retriever returned nothing with a single label.
    for xi, d, s, h in zip(x, dense, sparse, hybrid):
        if d == 0 and s == 0 and h == 0:
            ax.text(xi, 0.06, "no matches", ha="center", va="bottom",
                    fontsize=10, color=MGRAY, style="italic")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(f"Precision@{k}", fontsize=11, color=MGRAY)
    ax.set_title(f"Dense vs. Sparse vs. Hybrid Retrieval: Precision@{k}",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    ax.legend(fontsize=11, framealpha=0)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_precision_recall(k_vals: list, p_curve: list, r_curve: list,
                          save_path="images/augmentation/precision_recall.png"):
    """Precision and recall curves vs retrieval depth k."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)

    ax.plot(k_vals, p_curve, color=BLUE,  marker="o", lw=2, ms=7, label="Precision@k")
    ax.plot(k_vals, r_curve, color=GREEN, marker="s", lw=2, ms=7, label="Recall@k")

    diffs   = [abs(p - r) for p, r in zip(p_curve, r_curve)]
    ci      = diffs.index(min(diffs))
    cross_k = k_vals[ci]
    cross_y = (p_curve[ci] + r_curve[ci]) / 2
    ax.axvline(cross_k, color=ORANGE, ls="--", lw=1.2)
    ax.text(cross_k + 0.15, max(0.05, cross_y - 0.22),
            f"crossover\n(k={cross_k})", color=ORANGE, fontsize=10.5, style="italic")

    ax.set_xlabel("k (documents retrieved)", fontsize=11, color=MGRAY)
    ax.set_ylabel("score", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=11, framealpha=0)
    ax.set_title("Precision vs. Recall Tradeoff by Retrieval Depth",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_rag_architecture(save_path="images/augmentation/rag_architecture.png"):
    """Three-column full RAG system architecture diagram."""
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")

    SOLID = (BLUE, GREEN, ORANGE, RED)

    def lbox(cx, cy, label, fc, tc, hw=1.10, hh=0.45, sub=None):
        _fbox(ax, cx, cy, hw, hh, fc,
              ec=fc if fc in SOLID else BLUE,
              lw=0 if fc in SOLID else 0.8)
        if sub:
            _label(ax, cx, cy + 0.15, label, tc=tc, fs=10)
            ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                    fontsize=8.5, color=tc if fc in SOLID else MGRAY,
                    style="italic", zorder=4)
        else:
            _label(ax, cx, cy, label, tc=tc, fs=10)

    # Column 1: Ingestion
    for y1, lbl, fc, tc in [(5.4, "Raw Docs", LGRAY, DGRAY),
                              (4.2, "Clean +\nExtract", LGRAY, DGRAY),
                              (3.0, "Chunk", LGRAY, DGRAY),
                              (1.8, "Embed", LGRAY, DGRAY),
                              (0.7, "Vector\nIndex", BLUE, "white")]:
        lbox(1.3, y1, lbl, fc, tc, hw=0.95)
    for y1, y2 in [(5.4, 4.2), (4.2, 3.0), (3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 1.3, y1 - 0.45, 1.3, y2 + 0.45, pct=0.06)

    # Column 2: Query handling
    for y1, lbl, fc, tc in [(5.4, "User Query", GREEN, "white"),
                              (4.2, "Contextualize", LGRAY, DGRAY),
                              (3.0, "Hybrid\nRetrieve", LGRAY, DGRAY),
                              (1.8, "Rerank +\nDedup", LGRAY, DGRAY),
                              (0.7, "Compress\nContext", LGRAY, DGRAY)]:
        lbox(5.5, y1, lbl, fc, tc, hw=1.05)
    for y1, y2 in [(5.4, 4.2), (4.2, 3.0), (3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 5.5, y1 - 0.45, 5.5, y2 + 0.45, pct=0.06)

    # Column 3: Generation
    for y1, lbl, fc, tc in [(3.0, "Sanitize\nInput", LGRAY, DGRAY),
                              (1.8, "Ground +\nPrompt", LGRAY, DGRAY),
                              (0.7, "Cited\nAnswer", GREEN, "white")]:
        lbox(9.4, y1, lbl, fc, tc, hw=1.05)
    for y1, y2 in [(3.0, 1.8), (1.8, 0.7)]:
        _arr(ax, 9.4, y1 - 0.45, 9.4, y2 + 0.45, pct=0.08)

    _arr(ax, 1.3 + 0.95, 0.7, 5.5 - 1.05, 3.0, col=ORANGE, pct=0.08)
    _arr(ax, 5.5 + 1.05, 0.7, 9.4 - 1.05, 1.8, col=ORANGE, pct=0.10)
    _arr(ax, 5.5 + 1.05, 5.4, 9.4 - 1.05, 3.0, col=MGRAY, pct=0.10, ls="dashed")

    for x, title in [(1.3, "Indexing"), (5.5, "Retrieval"), (9.4, "Generation")]:
        ax.text(x, 6.1, title, ha="center", fontsize=11.5, fontweight="700", color=DGRAY)
        ax.plot([x - 1.1, x + 1.1], [5.85, 5.85], color=LGRAY, lw=1.2)

    ax.set_title("Full RAG System Architecture",
                 fontsize=13.5, fontweight="700", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


# ── Metacoding chapter ────────────────────────────────────────────────────────

def render_code_listing(snippets, font_size=30, pad=24, scale=2, save_path=None):
    """Render Python snippets as a syntax-highlighted image.

    Lets a chapter *show* a small code dataset as a figure (Pygments ``friendly``
    style — the same one the PDF's ``minted`` uses — set in IBM Plex Mono, the
    book's code font) instead of pasting it inline as string literals. The real
    snippets live as named constants in ``genai.code``; this just paints them.

    ``snippets`` is a single code string, or a list whose items are code strings
    or ``(name, code)`` pairs (the name is ignored — the ``def`` line already
    carries it). Multi-line snippets are spaced apart by a blank line. Returns a
    PIL image, which Jupyter displays as the cell's figure.
    """
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    from pygments.lexers import PythonLexer
    from pygments.styles import get_style_by_name

    if isinstance(snippets, str):
        blocks = [snippets]
    else:
        blocks = [s[1] if isinstance(s, (tuple, list)) else s for s in snippets]
    sep = "\n\n" if any("\n" in b for b in blocks) else "\n"
    code = sep.join(b.rstrip("\n") for b in blocks)

    # Shared fonts/ live at the monorepo root, i.e. one level above this package
    # (genai/viz.py -> parents[1] == repo root). resolve() follows the per-book
    # chapters/genai symlink back to the real package location first.
    fonts = Path(__file__).resolve().parents[1] / "fonts"
    fs, p = font_size * scale, pad * scale
    reg = ImageFont.truetype(str(fonts / "IBMPlexMono-Regular.ttf"), fs)
    bold = ImageFont.truetype(str(fonts / "IBMPlexMono-Bold.ttf"), fs)
    ital = ImageFont.truetype(str(fonts / "IBMPlexMono-Italic.ttf"), fs)
    cw = reg.getlength("M")                       # monospace cell width
    asc, desc = reg.getmetrics()
    lh = int((asc + desc) * 1.34)                 # line height

    style = get_style_by_name("friendly")
    rows = code.split("\n")
    W = int(p * 2 + cw * max((len(r) for r in rows), default=1))
    H = int(p * 2 + lh * len(rows))
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    x, y = p, p
    for ttype, value in PythonLexer().get_tokens(code):
        st = style.style_for_token(ttype)
        color = "#" + st["color"] if st["color"] else DGRAY
        font = bold if st["bold"] else (ital if st["italic"] else reg)
        for i, part in enumerate(value.split("\n")):
            if i:                                 # token spanned a line break
                x, y = p, y + lh
            if part:
                draw.text((x, y), part, font=font, fill=color)
                x += cw * len(part)

    img = img.resize((W // scale, H // scale), Image.LANCZOS)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    return img


def render_code_columns(columns, spacing=44, save_path=None, **kwargs):
    """Render several code blocks as side-by-side syntax-highlighted columns.

    ``columns`` is a list; each item is anything ``render_code_listing`` accepts.
    Use this when a single listing would be too tall for the page. Columns are
    top-aligned on a white canvas with a ``spacing`` of whitespace between them.
    """
    from PIL import Image
    imgs = [render_code_listing(col, **kwargs) for col in columns]
    width = sum(im.width for im in imgs) + spacing * (len(imgs) - 1)
    height = max(im.height for im in imgs)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + spacing
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
    return canvas


def plot_code_embeddings(save_path="images/metacoding/embeddings_2d.png"):
    """PCA projection of code function embeddings, colored by category."""
    from sklearn.decomposition import PCA
    from genai.code import embed_code

    CATEGORIES = {
        "arithmetic": [
            ("add",       "def add(a, b): return a + b"),
            ("multiply",  "def multiply(a, b): return a * b"),
            ("power",     "def power(base, exp): return base ** exp"),
            ("factorial", "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"),
        ],
        "strings": [
            ("reverse",    "def reverse(s): return s[::-1]"),
            ("uppercase",  "def uppercase(s): return s.upper()"),
            ("palindrome", "def is_palindrome(s): return s == s[::-1]"),
            ("word_count", "def word_count(s): return len(s.split())"),
        ],
        "lists": [
            ("find_max",  "def find_max(lst): return max(lst)"),
            ("sort_desc", "def sort_desc(lst): return sorted(lst, reverse=True)"),
            ("flatten",   "def flatten(lst): return [x for sub in lst for x in sub]"),
            ("evens",     "def evens(lst): return [x for x in lst if x % 2 == 0]"),
        ],
        "file I/O": [
            ("read_file",  "def read_file(path): return open(path).read()"),
            ("write_file", "def write_file(path, data): open(path, 'w').write(data)"),
            ("log_error",  "def log_error(msg): open('error.log','a').write(msg + '\\n')"),
        ],
    }
    COLORS = {"arithmetic": BLUE, "strings": GREEN, "lists": ORANGE, "file I/O": PURPLE}

    vecs, labels, colors, cats = [], [], [], []
    for cat, items in CATEGORIES.items():
        for lbl, code in items:
            vecs.append(embed_code(code))
            labels.append(lbl)
            colors.append(COLORS[cat])
            cats.append(cat)

    coords = PCA(n_components=2).fit_transform(np.stack(vecs))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=COLORS[cat], s=100, label=cat, zorder=3,
                   edgecolors="white", lw=0.8)
    _place_labels(ax, coords, labels, fontsize=10.5)

    ax.set_title("Code Embeddings in 2D  (15 functions, PCA projection)",
                 fontsize=14, color=DGRAY, pad=12)
    ax.set_xlabel("First principal component", fontsize=11, color=MGRAY)
    ax.set_ylabel("Second principal component", fontsize=11, color=MGRAY)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.9, title="category")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10, colors=MGRAY)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_search_comparison(search_results: list,
                           save_path="images/metacoding/model_comparison.png"):
    """Grouped bar chart comparing two embedding models on code search results.

    search_results = [(name, unixcoder_score, nomic_score), ...]
    """
    names  = [r[0] for r in search_results]
    uni_sc = [r[1] for r in search_results]
    nom_sc = [r[2] for r in search_results]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w/2, uni_sc, w, color=BLUE,  label="UniXcoder",       zorder=2)
    ax.bar(x + w/2, nom_sc, w, color=GREEN, label="nomic-embed-text", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Similarity to query", fontsize=12.5, color=DGRAY)
    ax.set_title('Search scores for "combine two numbers"',
                 fontsize=14, color=DGRAY, pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_embedder_comparison(results,
                             save_path="images/metacoding/embedder_comparison.png"):
    """Grouped bar chart comparing three code embedders on one search query.

    results = [(name, unixcoder, codebert, codesearch), ...]. The y-axis dips
    below zero because CodeSearch can score an irrelevant function negative.
    """
    names = [r[0] for r in results]
    uni   = [r[1] for r in results]
    cb    = [r[2] for r in results]
    csn   = [r[3] for r in results]
    x = np.arange(len(names))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, uni, w, color=BLUE,  label="UniXcoder",  zorder=2)
    ax.bar(x,     cb,  w, color=AMBER, label="CodeBERT",   zorder=2)
    ax.bar(x + w, csn, w, color=TEAL,  label="CodeSearch", zorder=2)
    ax.axhline(0, color=MGRAY, lw=0.9, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("Similarity to query", fontsize=12.5, color=DGRAY)
    ax.set_title('Three code embedders score "combine two numbers"',
                 fontsize=14, color=DGRAY, pad=10)
    ax.set_ylim(-0.2, 1.12)
    ax.legend(fontsize=10.5, ncol=3, loc="upper center", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_model_speed(bench: list, save_path="images/metacoding/speed.png"):
    """Horizontal bar chart of tokens/sec for each model.

    bench = [{"label": str, "tps": float, "params_b": float}, ...]
    Color bins are only shown in the legend if they appear in the data.
    """
    labels = [b["label"] for b in bench]
    tps    = [b["tps"]   for b in bench]

    def bin_for(t):
        if t > 60: return ("fast",   BLUE,   "> 60 tok/s")
        if t > 35: return ("medium", GREEN,  "35–60 tok/s")
        return         ("slow",   ORANGE, "< 35 tok/s")

    bin_info = [bin_for(t) for t in tps]
    colors = [c for _, c, _ in bin_info]

    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, tps, color=colors, height=0.52, zorder=2)
    for bar, val in zip(bars, tps):
        ax.text(val + 1.8, bar.get_y() + bar.get_height()/2,
                f"{val:.0f} tok/s", va="center", fontsize=11, color=DGRAY)

    ax.set_xlabel("Tokens per second", fontsize=12.5, color=DGRAY)
    # Extra right-side headroom so value labels never collide with the legend.
    ax.set_xlim(0, max(tps) * 1.35)
    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("Coding Model Speed (local, Apple Silicon)",
                 fontsize=14, fontweight="700", color=DGRAY, pad=10)

    seen = {}
    for name, col, lbl in bin_info:
        seen.setdefault(name, (col, lbl))
    if len(seen) > 1:
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in seen.values()],
                  loc="upper right",
                  bbox_to_anchor=(1.0, 1.0),
                  fontsize=10, framealpha=0.9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_size_vs_speed(bench: list, save_path="images/metacoding/size_vs_speed.png"):
    """Scatter: model parameter count vs tokens/sec.

    bench = [{"label": str, "tps": float, "params_b": float}, ...]
    Labels for points within a small neighborhood get stacked vertically
    so they don't collide.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for b in bench:
        ax.scatter(b["params_b"], b["tps"], s=140, color=BLUE, zorder=3,
                   edgecolors="white", linewidths=0.8)

    tps_range = max(b["tps"] for b in bench) - min(b["tps"] for b in bench) or 1.0
    par_range = max(b["params_b"] for b in bench) - min(b["params_b"] for b in bench) or 1.0

    for i, b in enumerate(bench):
        # Count how many earlier points are in this point's cluster.
        stack_idx = sum(
            1 for j, other in enumerate(bench)
            if j < i
            and abs(b["params_b"] - other["params_b"]) < par_range * 0.10
            and abs(b["tps"] - other["tps"]) < tps_range * 0.04
        )
        ax.annotate(b["label"],
                    (b["params_b"], b["tps"]),
                    xytext=(10, 4 + stack_idx * 14),
                    textcoords="offset points",
                    fontsize=11, color=DGRAY, va="center")

    ax.set_xlabel("Model size (billions of parameters)", fontsize=12.5, color=DGRAY)
    ax.set_ylabel("Tokens per second", fontsize=12.5, color=DGRAY)
    ax.set_title("Smaller Models Generate Faster",
                 fontsize=14, fontweight="700", color=DGRAY, pad=10)
    ax.grid(alpha=0.18, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    # A touch of right-padding so labels don't run off-canvas.
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.05)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ── Efficiency chapter ────────────────────────────────────────────────────────

def _short(model: str) -> str:
    """Trim the ':latest' / 'instruct' noise off a model tag for axis labels."""
    return model.replace(":latest", "").replace("-instruct", "")


def plot_throughput(rows: list, save_path="images/efficiency/throughput.png"):
    """Two panels on one short question across models: per-token speed (left)
    and total time to answer split into prompt-reading vs generation (right).

    rows = [{"model": str, "tokens_per_sec"|"tps": float, "prompt_ms": float,
             "gen_tokens": int}, ...]
    The pairing reveals two things at once: the fastest model per token is not
    the fastest to finish, and almost the whole wait is generation. The
    prompt-reading sliver barely registers while a reasoning model that thinks
    for thousands of tokens balloons the generation slab. Generation time is
    derived as gen_tokens / tokens_per_sec, the pure writing time, so a one-off
    model load never leaks into the slab.
    """
    labels = [_short(r["model"]) for r in rows]
    tps    = [r.get("tps", r.get("tokens_per_sec")) for r in rows]
    toks   = [r.get("gen_tokens", 0) for r in rows]
    read_s = [r["prompt_ms"] / 1000 for r in rows]
    gen_s  = [t / v if v else 0.0 for t, v in zip(toks, tps)]
    secs   = [r + g for r, g in zip(read_s, gen_s)]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY, rotation=18, ha="right")

    bars1 = ax1.bar(x, tps, color=GREEN, width=0.55, zorder=3)
    ax1.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax1.set_title("Speed per Token", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_ylim(0, max(tps) * 1.18)
    for bar, v in zip(bars1, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tps)*0.02,
                 f"{v:.0f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    ax2.bar(x, read_s, color=BLUE, width=0.55, zorder=3, label="reading")
    ax2.bar(x, gen_s, bottom=read_s, color=ORANGE, width=0.55, zorder=3,
            label="generation")
    ax2.set_ylabel("time to answer (s)", fontsize=11, color=MGRAY)
    ax2.set_title("Reading vs Generation Time", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax2.set_ylim(0, max(secs) * 1.22)
    ax2.legend(fontsize=9.5, frameon=False, loc="upper left")
    for xi, s, t in zip(x, secs, toks):
        ax2.text(xi, s + max(secs)*0.02, f"{s:.0f}s\n{t} tok", ha="center",
                 va="bottom", fontsize=9.5, fontweight="600", color=DGRAY,
                 linespacing=1.1)

    fig.suptitle("Same One-Sentence Question, Four Models", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_context_cost(rows: list, save_path="images/efficiency/context_cost.png"):
    """Line chart: prompt-processing time grows linearly with prompt length.

    rows = [{"prompt_tokens": int, "prompt_ms": float}, ...]
    """
    toks = [r["prompt_tokens"] for r in rows]
    ms   = [r["prompt_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(toks, ms, color=BLUE, marker="o", lw=2, ms=8, zorder=3)
    for t, m in zip(toks, ms):
        ax.text(t, m + max(ms) * 0.03, f"{m:.0f}", ha="center",
                fontsize=10, color=DGRAY)
    ax.set_xlabel("prompt length (tokens)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("prompt-processing time (ms)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(ms) * 1.18)
    ax.set_title("Every Extra Token in the Prompt Costs Time",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_conversation_cache(warm: list, cold: list,
                            save_path="images/efficiency/conversation_cache.png"):
    """Two panels: transcript size per turn, and reading time warm vs cold.

    warm/cold = per-turn timing dicts from time_chat. The transcript grows
    every turn; a fresh model pays the full reading bill while the ongoing
    chat, served from the KV cache, reads only the newest question.
    """
    turns = np.arange(1, len(cold) + 1)
    toks = [r["prompt_tokens"] for r in cold]
    cms = [r["prompt_ms"] for r in cold]
    wms = [r["prompt_ms"] for r in warm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        _grid_ax(ax)
        ax.set_xticks(turns)
        ax.set_xlabel("turn", fontsize=11, color=MGRAY)

    ax1.bar(turns, toks, color=BLUE, width=0.6, zorder=3)
    for t, v in zip(turns, toks):
        ax1.text(t, v + max(toks) * 0.03, f"{v}", ha="center",
                 fontsize=10, fontweight="600", color=DGRAY)
    ax1.set_ylim(0, max(toks) * 1.18)
    ax1.set_ylabel("transcript (tokens)", fontsize=11, color=MGRAY)
    ax1.set_title("What the App Sends Each Turn", fontsize=13,
                  fontweight="600", color=DGRAY, pad=6)

    ax2.plot(turns, cms, color=RED, marker="o", lw=2, ms=7, zorder=3,
             label="fresh model (no cache)")
    ax2.plot(turns, wms, color=GREEN, marker="o", lw=2, ms=7, zorder=3,
             label="ongoing chat (cached)")
    for t, v in zip(turns, cms):
        ax2.text(t, v + max(cms) * 0.04, f"{v:.0f}", ha="center",
                 fontsize=9.5, color=DGRAY)
    ax2.text(turns[-1], wms[-1] + max(cms) * 0.04, f"{wms[-1]:.0f}",
             ha="center", fontsize=9.5, color=DGRAY)
    ax2.set_ylim(0, max(cms) * 1.22)
    ax2.set_ylabel("reading time (ms)", fontsize=11, color=MGRAY)
    ax2.legend(fontsize=10, frameon=False, loc="upper left")
    ax2.set_title("Time the Model Spends Reading", fontsize=13,
                  fontweight="600", color=DGRAY, pad=6)

    fig.suptitle("Six Turns, One Growing Transcript", fontsize=14,
                 fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_generation_sag(rows: list,
                        save_path="images/efficiency/generation_sag.png"):
    """Line: generation speed vs transcript length across one long chat.

    rows = per-turn timing dicts from time_chat. The KV cache spares the
    re-reading, but every new token still attends over the whole cache,
    so tokens/sec sags as the transcript grows.
    """
    toks = [r["prompt_tokens"] for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(toks, tps, color=ORANGE, marker="o", lw=2, ms=6, zorder=3)
    for i in (0, len(rows) - 1):
        ax.annotate(f"{tps[i]:.1f}", (toks[i], tps[i]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xlabel("transcript (tokens)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("generation speed (tokens / sec)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(tps) * 1.18)
    ax.set_title("Writing Slows as the Transcript Grows",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_prompt_compression(rows: list,
                            save_path="images/efficiency/prompt_compression.png"):
    """Paired horizontal bars: prompt size and reading time per phrasing.

    rows = [{"label": str, "prompt_tokens": int, "prompt_ms": float}, ...]
    The same question dressed three ways; the padding, not the question,
    sets the cost. Rows are sorted largest-first so the most padded
    phrasing lands on top.
    """
    rows = sorted(rows, key=lambda r: -r["prompt_tokens"])
    labels = [r["label"] for r in rows]
    toks = [r["prompt_tokens"] for r in rows]
    ms = [r["prompt_ms"] for r in rows]
    colors = [RED, ORANGE, GREEN, BLUE, PURPLE][:len(rows)]
    y = np.arange(len(rows))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, vals, unit, title in [
            (ax1, toks, "tokens", "What the Model Reads"),
            (ax2, ms, "milliseconds", "Time Before Response")]:
        _grid_ax(ax)
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.yaxis.grid(False)
        ax.barh(y, vals, color=colors, height=0.58, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.0f}", va="center",
                    fontsize=10.5, fontweight="600", color=DGRAY)
        ax.set_xlim(0, max(vals) * 1.16)
        ax.set_xlabel(unit, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=11, color=DGRAY)
    fig.suptitle("Price of Politeness", fontsize=14, fontweight="600",
                 color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_task_ladder(rows: list, save_path="images/efficiency/task_ladder.png"):
    """Grouped bars: generation speed for a small vs large model up a task ladder.

    rows = [{"task": str, "small": float, "large": float}, ...]
    The flat tops carry the lesson: per-token speed barely moves as the task
    gets harder, because speed is a property of the model, not the question.
    """
    tasks = [r["task"] for r in rows]
    small = [r["small"] for r in rows]
    large = [r["large"] for r in rows]
    x = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    top = max(small + large)
    for offs, vals, color, label in [(-w/2, small, GREEN, "1B model"),
                                     (+w/2, large, BLUE, "3B model")]:
        bars = ax.bar(x + offs, vals, w, color=color, zorder=3, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top*0.02,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11.5, color=DGRAY)
    ax.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax.set_ylim(0, top * 1.38)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center", ncols=2)
    ax.set_title("Per-Token Speed Belongs to the Model, Not the Task",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_quantization(rows: list, save_path="images/efficiency/quantization.png"):
    """Dual bars: disk footprint and generation speed across quantization levels.

    rows = [{"model": str, "size_gb": float, "tps": float}, ...]
    """
    labels = [_short(r["model"]).split(":")[-1] for r in rows]
    size   = [r["size_gb"] for r in rows]
    tps    = [r["tps"] for r in rows]
    colors = [GREEN, BLUE, ORANGE][:len(rows)]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, color=DGRAY)

    for ax, vals, title, ylab, fmt in [
        (ax1, size, "Disk Footprint", "gigabytes", "{:.2f}"),
        (ax2, tps,  "Generation Speed", "tokens / sec", "{:.0f}"),
    ]:
        bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
        ax.set_ylabel(ylab, fontsize=11, color=MGRAY)
        ax.set_title(title, fontsize=13, fontweight="600", color=DGRAY, pad=6)
        ax.set_ylim(0, max(vals) * 1.18)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    fig.suptitle("One Model, Three Quantizations (Llama 3.2 1B)",
                 fontsize=13.5, fontweight="600", color=DGRAY, y=1.03)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


# Active-parameter colors: dense pays for everything, a mixture-of-experts
# model lights up only a sliver, and an on-device model sits between the two.
_SPARSITY_COLOR = {"dense": BLUE, "selective": TEAL, "expert": ORANGE}


def _sparsity_kind(kind: str) -> str:
    """Bucket a model's description into dense / selective / expert."""
    if "expert" in kind:
        return "expert"
    if "selective" in kind or "PLE" in kind:
        return "selective"
    return "dense"


def plot_sparsity(rows: list, save_path="images/efficiency/sparsity.png"):
    """Two panels: parameters spent per token, and generation speed.

    rows = [{"model": str, "total_b": float, "active_b": float, "tps": float,
             "kind": str}, ...]
    The left panel draws each model's stored parameters as a light bar with the
    parameters it actually activates per token filled solid on top, so a
    mixture-of-experts model shows a tall light bar over a short solid core.
    The right panel shows decode speed tracks that active core, not the stored
    total: the MoE model decodes as fast as dense models a fraction of its size.
    """
    import matplotlib.patches as mpatches
    labels = [_short(r["model"]).split(":")[0] for r in rows]
    total  = [r["total_b"] for r in rows]
    active = [r["active_b"] for r in rows]
    tps    = [r["tps"] for r in rows]
    colors = [_SPARSITY_COLOR[_sparsity_kind(r["kind"])] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, color=DGRAY)

    # Left: stored parameters (light) with the active core (solid) on top.
    ax1.bar(x, total, width=0.62, color=LGRAY, zorder=2)
    ax1.bar(x, active, width=0.62, color=colors, zorder=3)
    ax1.set_ylabel("parameters (billions)", fontsize=11, color=MGRAY)
    ax1.set_title("Stored vs Active per Token", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax1.set_ylim(0, max(total) * 1.20)
    for xi, t, a in zip(x, total, active):
        ax1.text(xi, t + max(total) * 0.02, f"{t:.1f}", ha="center", va="bottom",
                 fontsize=9.5, color=MGRAY)
        if a < t * 0.85:   # label the active core only when it is visibly smaller
            ax1.text(xi, a + max(total) * 0.02, f"{a:.0f}", ha="center",
                     va="bottom", fontsize=9.5, fontweight="700", color=DGRAY)
    present = [k for k in ("dense", "selective", "expert")
               if k in {_sparsity_kind(r["kind"]) for r in rows}]
    legend_label = {"dense": "active (dense)", "selective": "active (on-device)",
                    "expert": "active (mixture-of-experts)"}
    handles = [mpatches.Patch(color=LGRAY, label="stored")] + \
              [mpatches.Patch(color=_SPARSITY_COLOR[k], label=legend_label[k])
               for k in present]
    ax1.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left",
               handlelength=1.1, labelspacing=0.3)

    # Right: decode speed, colored to match the active core.
    bars = ax2.bar(x, tps, width=0.58, color=colors, zorder=3)
    ax2.set_ylabel("tokens / sec", fontsize=11, color=MGRAY)
    ax2.set_title("Generation Speed", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax2.set_ylim(0, max(tps) * 1.20)
    for bar, v in zip(bars, tps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tps)*0.02,
                 f"{v:.0f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    fig.suptitle("Store a Big Model, Run a Small One",
                 fontsize=14, fontweight="600", color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_effort_cost(study: dict, save_path="images/efficiency/effort_cost.png"):
    """Two panels: what a reasoning-effort dial costs vs what it buys.

    study = {"efforts": [...], "bill": [...], "gen_s": [...], "correct": [...],
             "n_tasks": int}
    Left, the mean token bill per effort level, rising green to red with the
    answer time labelled on each bar. Right, accuracy over the same tasks, which
    stays flat: turning the dial up spends more and returns the same answers.
    """
    efforts = [e.capitalize() for e in study["efforts"]]
    bill = study["bill"]
    gen_s = study["gen_s"]
    n = study["n_tasks"]
    acc = [100.0 * c / n for c in study["correct"]]
    x = np.arange(len(efforts))
    cost_colors = [GREEN, AMBER, RED][:len(efforts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.7))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _grid_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(efforts, fontsize=10.5, color=DGRAY)
        ax.set_xlabel("reasoning effort", fontsize=10.5, color=MGRAY)

    # Left: token bill (rising), with answer time labelled on each bar.
    bars = ax1.bar(x, bill, width=0.6, color=cost_colors, zorder=3)
    ax1.set_ylabel("tokens per answer", fontsize=11, color=MGRAY)
    ax1.set_title("What You Pay", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax1.set_ylim(0, max(bill) * 1.22)
    for bar, b, s in zip(bars, bill, gen_s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bill)*0.02,
                 f"{b} tok\n{s:.1f}s", ha="center", va="bottom",
                 fontsize=10, fontweight="600", color=DGRAY, linespacing=1.1)

    # Right: accuracy (flat) over the same tasks.
    abars = ax2.bar(x, acc, width=0.6, color=GREEN, zorder=3)
    ax2.set_ylabel("tasks correct (%)", fontsize=11, color=MGRAY)
    ax2.set_title("What You Get", fontsize=13, fontweight="600", color=DGRAY, pad=6)
    ax2.set_ylim(0, 119)
    for bar, c in zip(abars, study["correct"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{c}/{n}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="600", color=DGRAY)

    fig.suptitle("The Price of Thinking", fontsize=14, fontweight="600",
                 color=DGRAY, y=1.04)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_embed_vs_gen(n: int, embed_s: float, gen_s: float,
                      save_path="images/efficiency/embed_vs_gen.png"):
    """Two bars on a log axis: embedding vs generation throughput (items/sec)."""
    embed_rate = n / embed_s
    gen_rate   = n / gen_s
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(["embedding", "generation"], [embed_rate, gen_rate],
                  color=[GREEN, ORANGE], width=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("items per second (log scale)", fontsize=11.5, color=MGRAY)
    for bar, v in zip(bars, [embed_rate, gen_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.12,
                f"{v:.1f}/s", ha="center", va="bottom",
                fontsize=11, fontweight="600", color=DGRAY)
    ax.set_title(f"Embedding Is ~{embed_rate/gen_rate:.0f}x Faster Than Generation",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Responsible chapter ───────────────────────────────────────────────────────

def plot_privacy_budget(rows, save_path="images/responsible/privacy_budget.png"):
    """The differential-privacy tradeoff curve: error in the released statistic
    against the privacy budget epsilon.

    rows = [(epsilon, private_mean, abs_error, strength), ...] as returned by
    genai.security.dp_budget_table. The error collapses as epsilon grows, which
    is precisely the privacy you spend to buy that accuracy back.
    """
    eps = [r[0] for r in rows]
    err = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.set_xscale("log")
    ax.plot(eps, err, color=RED, marker="o", lw=2.2, ms=9, zorder=3)
    ax.fill_between(eps, err, color=RED, alpha=0.08, zorder=1)
    for e, v in zip(eps, err):
        ax.text(e, v + max(err) * 0.04, f"{v:.1f}", ha="center",
                fontsize=10, color=DGRAY, zorder=4)
    ax.set_xlabel("privacy budget  ε   (log scale)", fontsize=11.5, color=MGRAY)
    ax.set_ylabel("error in released average (mmHg)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(err) * 1.25)
    ax.text(eps[0], max(err) * 1.12, "strong privacy\nlots of noise", fontsize=9.5,
            color=GREEN, ha="left", va="top", fontweight="700")
    ax.text(eps[-1], max(err) * 0.5, "weak privacy\nlittle noise", fontsize=9.5,
            color=ORANGE, ha="right", va="center", fontweight="700")
    ax.set_title("The Privacy Budget: No Free Lunch", fontsize=14,
                 fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_watermark_detection(watermarked_z, human_z,
                             save_path="images/responsible/watermark_detection.png"):
    """Strip plot of green-list z-scores for watermarked vs. human text.

    Human text scatters around zero (about half its tokens are 'green' by
    chance); watermarked text spikes well past the usual z=4 detection
    threshold. The watermark is invisible to a reader but obvious to the score.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.axvline(4.0, color=MGRAY, ls="--", lw=1.3, zorder=2)
    ax.text(4.0, 1.62, "detection\nthreshold (z=4)", fontsize=9, color=MGRAY,
            ha="center", va="bottom", linespacing=1.1)
    for z, y, col, lbl in [(human_z, 1.0, BLUE, "human"),
                           (watermarked_z, 0.0, RED, "watermarked")]:
        jitter = y + rng.uniform(-0.13, 0.13, len(z))
        ax.scatter(z, jitter, s=120, color=col, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.text(min(z) - 0.4, y, lbl, fontsize=11.5, fontweight="700",
                color=col, ha="right", va="center")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(min(min(human_z), 0) - 3.0, max(watermarked_z) + 1.2)
    ax.set_xlabel("watermark z-score", fontsize=11.5, color=MGRAY)
    ax.set_title("A Watermark the Eye Can't See, the Math Can", fontsize=14,
                 fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Responsible chapter screenings (reuse plot_bakeoff) ───────────────────────

def plot_red_team_zoo(study, save_path="images/responsible/red_team_zoo.png"):
    """Per candidate model: percent of the red-team suite resisted wearing the
    hardened system prompt vs the naive one. The distance up from the faded naive
    bar is what the written security policy buys that model; the height of the
    solid bar is who you would trust with the production job."""
    plot_bakeoff(study, "hardened", "naive",
                 "hardened (security policy)", "naive (no policy)",
                 "Who Holds the Line Under Attack?",
                 "attacks resisted (%)", save_path, full_labels=True)


def plot_poison_zoo(study, save_path="images/responsible/poison_zoo.png"):
    """Per model: how completely it follows the in-context examples it is
    handed, the honest few-shot set vs the label-flipped one. Both bars rise
    together; a model that follows the honest examples to the letter follows the
    poisoned ones just as completely, and a shorter poisoned bar is a model
    leaning on its own prior instead of the flipped labels."""
    plot_bakeoff(study, "clean", "poisoned",
                 "follows honest examples", "follows poisoned examples",
                 "Who Swallows the Poisoned Labels?",
                 "follows the examples (%)", save_path)


def plot_pii_screen(study, save_path="images/responsible/pii_screen.png"):
    """Per model: percent of summaries with zero PII regex hits when politely
    asked to omit identifiers vs when not asked. The faded bar is the default
    behavior (a faithful summarizer parrots the identifiers); the distance up to the
    solid bar is how much a polite request buys, and any solid bar short of 100
    is a model ignoring the request."""
    plot_bakeoff(study, "instructed", "unprompted",
                 "asked to omit identifiers", "not asked",
                 "Can You Just Ask the Model to Redact?",
                 "identifier-free summaries (%)", save_path)


def plot_safety_judges(study, save_path="images/responsible/safety_judges.png"):
    """Per candidate judge: percent of quietly dangerous drafts flagged vs
    percent of safe drafts cleared. A usable judge is tall on both; tall only
    on flags is a paranoid judge that blocks everything, tall only on clears
    is a rubber stamp."""
    plot_bakeoff(study, "flags", "clears",
                 "flags the dangerous", "clears the safe",
                 "Who Can Sit in the Judge's Chair?",
                 "rate (%)", save_path)


# ── Semantics chapter ─────────────────────────────────────────────────────────

def plot_word_vs_embed_similarity(pairs: list, pair_labels: list, save_path=None):
    """Grouped bar chart: word-overlap similarity vs embedding similarity.

    pairs = [(sentence_a, sentence_b), ...]  — similarity is computed internally.
    """
    from genai.embed import similarity as _sim

    words_set  = lambda s: set(s.lower().split())
    word_sims  = [len(words_set(a) & words_set(b)) / len(words_set(a) | words_set(b))
                  for a, b in pairs]
    embed_sims = [_sim(a, b) for a, b in pairs]

    x = np.arange(len(pair_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, word_sims,  w, label="Word overlap",         color="#e07b54", alpha=0.9)
    ax.bar(x + w/2, embed_sims, w, label="Embedding similarity", color="#4a90d9", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Similarity score", fontsize=11)
    ax.set_title("Word Overlap vs. Embedding Similarity", fontsize=13, fontweight="600")
    ax.legend(fontsize=10.5)
    ax.tick_params(labelsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_crosslingual(study: dict, save_path="images/semantics/crosslingual.png"):
    """Line chart: cosine of an English sentence to its translation in each
    language, traced by two multilingual embedders, nomic-embed-text-v2-moe and
    embeddinggemma. The x-axis runs across languages from the best-carried down
    to Swahili; the unrelated control is drawn as a shaded floor band so each
    language reads against the baseline it would hit by chance.

    Both lines stay high across the well-resourced languages, embeddinggemma above,
    then plunge together at Swahili toward the unrelated floor: meaning transfers
    where the training data was, and runs out at the low-resource edge.
    """
    labels, nomic_v2, gemma = study["labels"], study["nomic_v2"], study["gemma"]
    # The last entry is the unrelated control, drawn as a floor band, not a language.
    langs = labels[:-1]
    nomic_lang, gemma_lang = nomic_v2[:-1], gemma[:-1]
    floor_lo, floor_hi = sorted((nomic_v2[-1], gemma[-1]))
    x = np.arange(len(langs))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    # Unrelated-sentence floor: the two controls bracket a thin band; shade the
    # basement below it so anything sinking in is "no better than random."
    ax.axhspan(0, floor_hi, color=MGRAY, alpha=0.13, zorder=0)
    ax.axhline(floor_hi, color=MGRAY, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.text(len(langs) - 1.45, floor_hi + 0.015, "unrelated-sentence floor",
            ha="left", va="bottom", fontsize=9, style="italic", color=DGRAY)
    ax.plot(x, gemma_lang, "-o", color=ORANGE, lw=2.4, ms=8, zorder=3, label="embeddinggemma")
    ax.plot(x, nomic_lang, "-o", color=BLUE,   lw=2.4, ms=8, zorder=3,
            label="nomic-embed-text-v2-moe")
    for xi, v in zip(x, gemma_lang):
        ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="600", color=ORANGE)
    for xi, v in zip(x, nomic_lang):
        ax.text(xi, v - 0.04, f"{v:.2f}", ha="center", va="top",
                fontsize=9, fontweight="600", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=10.5, color=DGRAY)
    ax.set_ylabel("cosine to the English sentence", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=10, framealpha=0, loc="upper right")
    ax.set_title("Does Meaning Survive Translation?",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_toy_embeddings(pts: dict, save_path=None):
    """Scatter of hand-crafted 2D embeddings showing word clustering."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for word, (x, y) in pts.items():
        ax.scatter(x, y, s=110)
        ax.annotate(word, (x, y), fontsize=11.5,
                    textcoords="offset points", xytext=(5, 4))
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.tick_params(labelsize=10)
    ax.set(xlim=(-0.3, 1.2), ylim=(-0.55, 1.2))
    ax.set_title("Word Embeddings in 2D", fontsize=13, fontweight="600")
    ax.set_xlabel("Dim 1", fontsize=11); ax.set_ylabel("Dim 2", fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


_TSNE_PALETTE = {
    "animals":  "#4a90d9", "vehicles": "#e07b54",
    "fruits":   "#27ae60", "code":     "#8e44ad",
}


def plot_tsne_embeddings(word_groups: dict, palette: dict = None, save_path=None):
    """t-SNE projection of word embeddings, colored by semantic group.

    word_groups = {"group_name": [word, ...], ...}  — embeddings computed internally.
    """
    from sklearn.manifold import TSNE
    from genai.embed import embed as _embed

    if palette is None:
        palette = _TSNE_PALETTE

    words        = [w for ws in word_groups.values() for w in ws]
    group_labels = [g for g, ws in word_groups.items() for _ in ws]
    vecs         = np.array([_embed(w) for w in words])
    coords       = TSNE(n_components=2, random_state=42,
                        perplexity=5).fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(8, 6))
    for grp, color in palette.items():
        mask = [i for i, g in enumerate(group_labels) if g == grp]
        ax.scatter(coords[mask, 0], coords[mask, 1], color=color, s=110, label=grp)
    for i, word in enumerate(words):
        ax.annotate(word, coords[i], fontsize=11,
                    textcoords="offset points", xytext=(4, 4))
    ax.legend(title="Group", framealpha=0.9, fontsize=10.5, title_fontsize=11)
    ax.tick_params(labelsize=10.5)
    ax.set_title("Real Word Embeddings: t-SNE Projection", fontsize=13, fontweight="600")
    ax.set_xlabel("t-SNE dim 1", fontsize=11); ax.set_ylabel("t-SNE dim 2", fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cosine_angles(save_path=None):
    """Three arrows-from-origin panels illustrating cosine = 1, 0, -1.

    A friendly companion to the formula: two vectors that point the same way
    score 1, that sit at a right angle score 0, and that point in opposite
    directions score -1. The first panel deliberately draws two arrows of
    different lengths to show that only direction matters, not magnitude.
    """
    A, B = BLUE, ORANGE
    panels = [
        ("Same direction", "+1", [(0.92, 0.66, A), (0.58, 0.42, B)]),
        ("Right angle",     "0", [(0.95, 0.00, A), (0.00, 0.95, B)]),
        ("Opposite",       "-1", [(0.80, 0.60, A), (-0.80, -0.60, B)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3))
    for ax, (caption, cos, vecs) in zip(axes, panels):
        for x, y, col in vecs:
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6,
                                        mutation_scale=20, shrinkA=0, shrinkB=0))
        ax.scatter([0], [0], s=16, color=DGRAY, zorder=5)
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(caption, fontsize=12.5, fontweight="700", color=DGRAY, pad=4)
        ax.text(0, -1.08, f"cosine = {cos}", ha="center", va="center",
                fontsize=12, fontweight="600", color=MGRAY)
    fig.suptitle("Cosine similarity is the angle between two vectors",
                 fontsize=13, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_analogy_offsets(save_path=None):
    """Why an analogy carries from one pair to another, drawn as arrows.

    Each panel measures the step ``a -> b`` for two pairs and draws them from a
    common origin at their TRUE angle apart (arccos of their cosine). When the
    arrows point the same way the analogy transfers; gender keeps a tight cone,
    opposites splay toward a right angle, so 'the opposite of' is a different
    arrow for every word. Angles are read from real embeddings, not posed.
    """
    from genai.embed import embed, similarity
    step = lambda a, b: embed(b) - embed(a)

    # Each arrow: word_a, word_b, color, (label_x, label_y, ha, va).
    # The first arrow in each panel is the reference, drawn along +x.
    panels = [
        ("Gender", [
            ("man", "woman", BLUE,     (0.55, -0.17, "center", "top")),
            ("actor", "actress", GREEN, (0.74, 0.73, "left",   "center")),
            ("king", "queen", ORANGE,  (0.40, 0.99, "center", "bottom")),
        ]),
        ("Comparative", [
            ("big", "bigger", BLUE,      (0.55, -0.17, "center", "top")),
            ("tall", "taller", GREEN,    (0.70, 0.80, "left",   "center")),
            ("small", "smaller", ORANGE, (0.18, 1.04, "center", "bottom")),
        ]),
        ("Opposite", [
            ("happy", "sad", BLUE,   (0.55, -0.17, "center", "top")),
            ("rich", "poor", ORANGE, (0.38,  1.00, "left",  "bottom")),
            ("hot", "cold", GREEN,   (-0.05, 1.15, "right", "bottom")),
        ]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9))
    for ax, (title, arrows) in zip(axes, panels):
        ref = step(arrows[0][0], arrows[0][1])
        drawn = []
        for i, (a, b, col, lab) in enumerate(arrows):
            cos = 1.0 if i == 0 else similarity(ref, step(a, b))
            drawn.append((np.degrees(np.arccos(np.clip(cos, -1, 1))), a, b, col, lab))

        spread = max(ang for ang, *_ in drawn)
        ax.add_patch(mp.Wedge((0, 0), 1.0, 0, spread, color=LGRAY, zorder=0))
        ax.plot([0, 0], [0, 1.18], ls=(0, (2, 3)), color=MGRAY, lw=1, zorder=1)
        ax.text(0.03, 1.2, "90°", fontsize=8.5, color=MGRAY, ha="left", va="bottom")

        for ang, a, b, col, (lx, ly, ha, va) in drawn:
            r = np.radians(ang)
            ax.annotate("", xy=(np.cos(r), np.sin(r)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                        mutation_scale=15, shrinkA=0, shrinkB=0))
            ax.text(lx, ly, f"{a}→{b}", fontsize=9.5, fontweight="700",
                    color=col, ha=ha, va=va)
        ax.scatter([0], [0], s=14, color=DGRAY, zorder=5)
        ax.set(xlim=(-0.5, 1.5), ylim=(-0.34, 1.32))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="700", color=DGRAY, pad=6)

    fig.suptitle("An analogy carries only when both pairs move along the same arrow",
                 fontsize=12.5, fontweight="600", color=DGRAY, y=1.0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_spread_comparison(save_path=None):
    """Dumbbell chart of the related-vs-unrelated similarity spread, two ways.

    For each pair, a line runs from the unrelated score (orange) to the related
    score (blue); its length is the spread. Averaging crams every pair into a short
    span high on the scale, while the transformer pulls related and unrelated
    apart into long spans, so its lines are visibly longer. Real scores from
    genai.embed.CONTRAST_PAIRS, sorted by the transformer spread.
    """
    from genai import similarity
    from genai.embed import averaged_similarity, CONTRAST_PAIRS
    rows = [(lab, averaged_similarity(a, rel), averaged_similarity(a, unr),
             similarity(a, rel), similarity(a, unr)) for lab, a, rel, unr in CONTRAST_PAIRS]
    rows.sort(key=lambda r: r[3] - r[4])
    labels = [r[0] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    panels = [("Averaging word vectors", [(r[2], r[1]) for r in rows]),
              ("Transformer (nomic-embed-text)", [(r[4], r[3]) for r in rows])]
    for ax, (title, spans) in zip(axes, panels):
        for i, (unr, rel) in enumerate(spans):
            ax.plot([unr, rel], [i, i], color=LGRAY, lw=3.5, zorder=1,
                    solid_capstyle="round")
            ax.scatter([unr], [i], color=ORANGE, s=60, zorder=3)
            ax.scatter([rel], [i], color=BLUE, s=60, zorder=3)
        mean_spread = np.mean([rel - unr for unr, rel in spans])
        ax.set_title(f"{title}\nmean spread {mean_spread:.2f}",
                     fontsize=11.5, fontweight="700", color=DGRAY)
        ax.set_xlim(0, 1)
        ax.set_xlabel("cosine similarity", fontsize=10.5)
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=10.5)
    axes[0].set_ylim(-0.6, len(labels) - 0.4)

    from matplotlib.lines import Line2D
    dot = lambda c, l: Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                              markersize=9, label=l)
    axes[0].legend(handles=[dot(ORANGE, "unrelated"), dot(BLUE, "related")],
                   loc="lower left", bbox_to_anchor=(-0.5, 1.0), ncol=2,
                   fontsize=10, frameon=False, columnspacing=1.3, handletextpad=0.4)
    fig.suptitle("The transformer opens a wider spread between related and unrelated text",
                 fontsize=12.5, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def _show_token(t: str) -> str:
    """A token as a chart label, with its leading space made visible.

    Tokenizers fold the space before a word into the token, so "Washington" and
    " Washington" are different tokens that would otherwise print identically.
    A leading space shows as ␣; an all-whitespace token keeps its repr.
    """
    if not t.strip():
        return repr(t)
    return "␣" + t[1:] if t.startswith(" ") else t


def plot_next_token(prompt: str, dist: list, save_path=None):
    """Horizontal bar chart of a next-token probability distribution.

    dist = [(token, probability), ...] from genai.next_token_distribution,
    most likely first. The top candidate is highlighted; the rest fade gray.
    """
    labels = [_show_token(t) for t, _ in dist][::-1]
    probs  = [p * 100 for _, p in dist][::-1]
    colors = [MGRAY] * (len(probs) - 1) + [BLUE]   # winner sits at the top

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(range(len(probs)), probs, color=colors, zorder=3)
    for i, p in enumerate(probs):
        ax.text(p + 1.5, i, f"{p:.1f}%", va="center", fontsize=11, color=DGRAY)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability of being the next token (%)", fontsize=11)
    ax.set_title(f'"{prompt} ___"', fontsize=13, color=DGRAY, loc="left", pad=10)
    _grid_ax(ax)
    ax.xaxis.grid(True, color=LGRAY, zorder=0)
    ax.yaxis.grid(False)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_two_step(prompt, top_word, step1, step2, save_path=None):
    """Two next-token distributions stacked, showing generation one step at a time.

    Top panel: the model's distribution over the FIRST word after `prompt`, an
    open, spread-out field. Bottom panel: once we commit to its top pick,
    `top_word`, the model forecasts the SECOND word, and the field can snap
    sharply onto a single front-runner. The committed word is the blue bar in
    the top panel and the carried-over word in the bottom panel's title. It's
    usually the favorite, but any step-one pick can be committed, even a long
    shot, to show where it leads.

    step1, step2 = [(token, probability), ...] from genai.next_token_distribution.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6.8))
    panels = [
        (ax_top, step1, f'Step 1   "{prompt} ___"', "probability of the first word (%)"),
        (ax_bot, step2, f'Step 2   "{prompt} {top_word} ___"', "probability of the second word (%)"),
    ]
    picked = next((i for i, (tok, _) in enumerate(step1)
                   if tok.strip() == top_word.strip()), 0)
    for ax, dist, title, xlabel in panels:
        labels = [_show_token(tok) for tok, _ in dist][::-1]
        probs  = [p * 100 for _, p in dist][::-1]
        blue   = picked if dist is step1 else 0      # committed word; winner in step 2
        colors = [BLUE if i == len(probs) - 1 - blue else MGRAY for i in range(len(probs))]
        ax.barh(range(len(probs)), probs, color=colors, zorder=3)
        for i, p in enumerate(probs):
            label = f"{p:.0f}%" if p >= 1 else f"{p:.1f}%"   # a long shot still reads as nonzero
            ax.text(p + 1.5, i, label, va="center", fontsize=11, color=DGRAY)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=12.5, color=DGRAY, loc="left", pad=10)
        _grid_ax(ax)
        mark = ax.get_yticklabels()[len(probs) - 1 - blue]   # a long shot's bar can be a hairline (after _grid_ax, which recolors ticks)
        mark.set_color(BLUE); mark.set_fontweight("bold")
        ax.xaxis.grid(True, color=LGRAY, zorder=0)
        ax.yaxis.grid(False)
    plt.tight_layout(h_pad=2.2)
    _save(fig, save_path)
    plt.show()


# ── Prompting chapter ─────────────────────────────────────────────────────────

def plot_temperature_lifecycle(save_path="images/prompting/temperature_lifecycle.png"):
    """Recommended sampling temperature across the software development lifecycle.

    A high temperature suits open-ended phases (brainstorming, design); a low one
    suits phases that demand consistency (implementation, debugging). Test-case
    generation sits in the middle — you want some variety in the cases.
    """
    phases = ["Brainstorm", "Design", "Implement", "Generate\nTests", "Debug"]
    temps  = [0.9, 0.7, 0.2, 0.6, 0.1]
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    # Shade the "explore" (high) and "be consistent" (low) bands.
    ax.axhspan(0.55, 1.08, color=ORANGE, alpha=0.07, zorder=0)
    ax.axhspan(-0.05, 0.40, color=BLUE,  alpha=0.07, zorder=0)
    ax.text(len(phases) - 0.4, 1.0, "explore", color=ORANGE,
            fontsize=12, style="italic", ha="right", va="top")
    ax.text(len(phases) - 0.4, 0.02, "be consistent", color=BLUE,
            fontsize=12, style="italic", ha="right", va="bottom")

    pt_cols = [ORANGE if t >= 0.5 else BLUE for t in temps]
    ax.plot(x, temps, color=MGRAY, lw=2, zorder=2)
    ax.scatter(x, temps, c=pt_cols, s=130, zorder=3,
               edgecolors="white", linewidths=1.2)
    for xi, t in zip(x, temps):
        ax.text(xi, t + 0.07, f"{t:.1f}", ha="center", fontsize=12,
                fontweight="600", color=DGRAY, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=12, color=DGRAY)
    ax.set_ylim(-0.05, 1.14)
    ax.set_ylabel("recommended temperature", fontsize=12, color=MGRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(LGRAY); ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=11)
    ax.set_title("Temperature Across the Development Lifecycle",
                 fontsize=14, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_sampling_controls(dist, k=4, p=0.9, n=7, temps=(0.5, 1.0, 1.5),
                           save_path="images/prompting/sampling_controls.png"):
    """The three control-panel dials acting on one real next-token distribution.

    `dist` is a list of (token, probability) pairs from
    genai.next_token_distribution -- the model's own ranked guesses for the word
    that comes next. We keep the top `n`, renormalize, and show the same numbers
    two ways, mirroring the chapter's split between weighing and counting.

    Left: temperature reshapes the odds. A cool setting (T < 1) sharpens the
    distribution toward the front-runner; a hot one (T > 1) flattens it so the
    long-shots get a real say. Same candidates, different boldness.

    Right: top-k and top-p decide which candidates are even on the table. Top-k
    keeps a fixed count (the k tallest); top-p keeps the smallest group whose
    probabilities clear p, so it widens when the model is unsure. They are set
    here to disagree on purpose, which is the whole point of the panel.
    """
    toks = [t if t.strip() else repr(t) for t, _ in dist[:n]]
    base = np.array([pr for _, pr in dist[:n]], dtype=float)
    base = base / base.sum()                       # renormalize over shown candidates
    x = np.arange(n)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    # ── Left: one distribution, reshaped by temperature ──────────────────────
    logits = np.log(base)
    for T in temps:
        z = np.exp((logits - logits.max()) / T)
        probs = z / z.sum()
        col = BLUE if T < 1 else ORANGE if T > 1 else DGRAY
        axL.plot(x, probs * 100, color=col, lw=2.6 if T != 1 else 1.8,
                 marker="o", ms=6, zorder=3, label=f"T = {T:.1f}")
    _grid_ax(axL)
    axL.set_xticks(x); axL.set_xticklabels(toks, rotation=30, ha="right", fontsize=10)
    axL.set_ylabel("probability (%)", fontsize=11, color=MGRAY)
    axL.set_title("Temperature reshapes the odds", fontsize=12.5,
                  fontweight="600", color=DGRAY, pad=8)
    axL.legend(fontsize=10.5, frameon=False, loc="upper right")

    # ── Right: the same distribution, trimmed by top-k and top-p ─────────────
    m = int(np.searchsorted(np.cumsum(base), p) + 1)   # smallest nucleus clearing p
    axR.bar(x, base * 100, color=[BLUE if i < m else MGRAY for i in range(n)], zorder=3)
    axR.axvline(k - 0.5, color=ORANGE, lw=2.2, ls="--", zorder=4)
    _grid_ax(axR)
    top = base.max() * 100
    axR.set_ylim(0, top * 1.24)
    # Keep both labels off the dashed line at x = k - 0.5: top-k to its right,
    # top-p in the upper-left over the blue nucleus (a vertical line crosses any
    # text sitting at its x, whatever the height).
    axR.text(k - 0.5 + 0.12, top * 1.15, f"top-k = {k}", color=ORANGE, fontsize=11,
             fontweight="700", ha="left")
    axR.text(-0.4, top * 1.15, f"top-p = {p:g}  keeps {m}", color=BLUE,
             fontsize=11, fontweight="700", ha="left")
    axR.set_xticks(x); axR.set_xticklabels(toks, rotation=30, ha="right", fontsize=10)
    axR.set_ylabel("probability (%)", fontsize=11, color=MGRAY)
    axR.set_title("Top-k and top-p trim the field", fontsize=12.5,
                  fontweight="600", color=DGRAY, pad=8)

    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()


def plot_token_boundaries(text=None, lines=None, tokenizer="general",
                          title="Same Code, Two Tokenizers",
                          save_path="images/tokens/token_boundaries.png"):
    """Each token as its own colored chip, so "tokens are not words" is visible.

    Renders one or more strings as a left-to-right strip of chips, one chip per
    token, with chip width tracking the token's length. A leading space inside a
    token is drawn as a faint dot, so the word-boundary tokens that tiktoken
    emits (like " return") stay legible instead of looking clipped.

    Pass ``text`` to run one line through both the general-purpose and the code
    tokenizer, the side-by-side comparison the chapter leans on. For full control,
    pass ``lines`` instead: each row's value is either a string (tokenized with the
    function-level ``tokenizer``) or a ``(text, tokenizer)`` pair.
    """
    from genai.tokens import tokenize
    if lines is None:
        if text is None:
            text = "def get_user_by_id(user_id):"
        lines = {
            "general": (text, "general"),
            "code":    (text, "code"),
        }
    # Normalize each row to (label, text, tokenizer).
    rows = [(label, *(v if isinstance(v, tuple) else (v, tokenizer)))
            for label, v in lines.items()]
    palette = [BLUE, GREEN, PURPLE, ORANGE]
    cw, pad, sep = 0.34, 0.22, 0.12          # char width, chip padding, inter-chip sep
    show = lambda t: t.replace(" ", "·").replace("\n", "⏎") or "·"

    end = []
    for _, text, tok in rows:
        x = 0.0
        for piece in tokenize(text, tok):
            x += len(show(piece)) * cw + 2 * pad + sep
        end.append(x)
    span = max(end)

    fig, ax = plt.subplots(figsize=(9.6, 0.62 + 0.92 * len(rows)))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-2.0, span + 2.7); ax.axis("off")
    ax.set_ylim(0, len(rows))

    for r, (label, text, tok) in enumerate(rows):
        y = len(rows) - r - 0.5
        ax.text(-1.9, y, label, ha="left", va="center", fontsize=11.5,
                fontweight="700", color=DGRAY, family="monospace")
        x = 0.0
        for i, piece in enumerate(tokenize(text, tok)):
            disp = show(piece)
            w = len(disp) * cw + 2 * pad
            _fbox(ax, x + w / 2, y, w / 2, 0.30, palette[i % len(palette)])
            ax.text(x + w / 2, y, disp, ha="center", va="center", fontsize=11.5,
                    color="white", family="monospace", zorder=4)
            x += w + sep
        ax.text(x + 0.15, y, f"{i + 1} tokens", ha="left", va="center",
                fontsize=9.5, color=MGRAY, style="italic")

    ax.set_title(title, fontsize=13, fontweight="700",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_special_tokens(tokenizer="code",
                        save_path="images/tokens/special_tokens.png"):
    """Every reserved control token, grouped by the repository part it marks.

    Sorting the vocabulary's special tokens into families turns a flat list into
    a map: the code tokenizer carries dedicated markers for issue threads, pull
    requests, Jupyter notebooks, fill-in-the-middle, and even redacted secrets.
    Rules run in order, first match wins, and anything unmatched lands in "Other"
    so the figure stays honest to whatever the tokenizer actually ships.
    """
    from genai.tokens import special_tokens
    toks = special_tokens(tokenizer)
    groups = [
        ("Document boundaries", BLUE,   lambda t: t in ("<|endoftext|>", "<empty_output>")),
        ("Fill-in-the-middle",  GREEN,  lambda t: t.startswith("<fim_")),
        ("Repository layout",   ORANGE, lambda t: t in ("<repo_name>", "<file_sep>")),
        ("Issue threads",       PURPLE, lambda t: t.startswith("<issue_")),
        ("Jupyter notebooks",   TEAL,   lambda t: t.startswith("<jupyter")),
        ("Code transforms",     AMBER,  lambda t: "intermediate" in t),
        ("Pull requests",       PINK,   lambda t: t.startswith("<pr")),
        ("Redacted secrets",    RED,    lambda t: t in ("<NAME>", "<EMAIL>", "<KEY>", "<PASSWORD>")),
    ]
    buckets = {name: [] for name, _, _ in groups}
    order = list(groups)
    for t in toks:
        for name, _, rule in groups:
            if rule(t):
                buckets[name].append(t); break
        else:
            buckets.setdefault("Other", []).append(t)
    if buckets.get("Other"):
        order.append(("Other", MGRAY, None))

    # Flow layout: heading, then chips wrapping at a fixed content width.
    cw, pad, sep = 0.70, 0.55, 0.55           # char width, chip padding, inter-chip sep
    line_h, head_h, cat_gap = 1.35, 1.45, 0.6
    W = 60.0
    chips, heads = [], []                     # (cx, cy, hw, text, color) / (y, text, color, n)
    y = 0.0
    for name, color, _ in order:
        items = buckets[name]
        if not items:
            continue
        heads.append((y, f"{name}  ({len(items)})", color))
        y -= head_h
        x = 0.0
        for t in items:
            w = len(t) * cw + 2 * pad
            if x > 0 and x + w > W:
                x = 0.0; y -= line_h
            chips.append((x + w / 2, y, w / 2, t, color))
            x += w + sep
        y -= line_h + cat_gap
    total_h = -y + 0.4

    fig_w = 9.2
    fig, ax = plt.subplots(figsize=(fig_w, total_h * fig_w / (W + 6)))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.0, W + 5.0); ax.set_ylim(-total_h + 0.2, head_h + 0.4)
    ax.axis("off")
    for y0, text, color in heads:
        ax.text(0.0, y0, text, ha="left", va="center", fontsize=11.5,
                fontweight="700", color=color)
    for cx, cy, hw, text, color in chips:
        _fbox(ax, cx, cy, hw, 0.44, color)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", family="monospace", zorder=4)
    ax.set_title(f"What {len(toks)} Special Tokens Reveal",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_prompt_anatomy(save_path="images/prompting/prompt_anatomy.png"):
    """One good prompt, broken into the four parts you can tighten one at a time.

    Instruction (what to do), context (what the model cannot read from your mind),
    output format (the shape you want back), and a few examples. The worked example
    is a small support-ticket urgency classifier. Conceptual diagram, no model call.
    """
    parts = [
        ("INSTRUCTION",   "Sort each support ticket by urgency.",              BLUE,   "#DBEAFE"),
        ("CONTEXT",       "Small team; P1 means a customer is fully blocked.", GREEN,  "#DCFCE7"),
        ("OUTPUT FORMAT", "Reply with one word: high, medium, or low.",        ORANGE, "#FFEDD5"),
        ("EXAMPLES",      '"The checkout page is down"  →  high',              PURPLE, "#EDE9FE"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.3); ax.axis("off")
    ax.text(7, 4.06, "The Anatomy of a Prompt", ha="center",
            fontsize=14, fontweight="700", color=DGRAY)
    for (name, text, col, bg), y in zip(parts, [3.25, 2.45, 1.65, 0.85]):
        ax.text(0.2, y, name, ha="left", va="center",
                fontsize=11.5, fontweight="700", color=col)
        _fbox(ax, 8.7, y, 4.9, 0.33, bg, ec=col, lw=1.4)
        ax.text(4.0, y, text, ha="left", va="center", fontsize=11, color=DGRAY)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_token_pipeline(word="strawberry",
                        save_path="images/tokens/token_pipeline.png"):
    """Text -> subword tokens -> integer IDs -> model. The model only sees numbers."""
    from genai.tokens import tokenize, token_ids
    pieces, ids = tokenize(word), token_ids(word)

    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

    y, xs = 2.35, [1.9, 5.3, 8.7, 12.1]
    stages = [
        (xs[0], f'"{word}"',            LGRAY,  DGRAY,   "raw text"),
        (xs[1], "  ".join(pieces),      BLUE,   "white", "subword tokens"),
        (xs[2], str(ids),               PURPLE, "white", "integer IDs"),
        (xs[3], "Language\nModel",      GREEN,  "white", "sees only numbers"),
    ]
    for cx, lbl, fc, tc, sub in stages:
        _fbox(ax, cx, y, 1.30, 0.58, fc)
        _label(ax, cx, y, lbl, tc=tc, fs=11)
        ax.text(cx, y - 1.02, sub, ha="center", fontsize=9.5,
                color=MGRAY, style="italic")
    for i in range(len(xs) - 1):
        _arr(ax, xs[i], y, xs[i + 1], y, pct=0.41)

    ax.set_title("From Text to Tokens to Numbers",
                 fontsize=13, fontweight="700", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_multilingual_tokens(save_path="images/tokens/multilingual_tokens.png"):
    """Grouped bars: the same sentence costs more tokens in some languages."""
    from genai.tokens import count_tokens
    sentences = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Spanish": "El veloz zorro marrón salta sobre el perro perezoso.",
        "Hindi":   "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    }
    langs   = list(sentences)
    general = [count_tokens(s) for s in sentences.values()]
    multi   = [count_tokens(s, "multilingual") for s in sentences.values()]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor("white"); _grid_ax(ax)
    x, w = np.arange(len(langs)), 0.38
    bars = [ax.bar(x - w/2, general, w, color=MGRAY, label="general (GPT-4)", zorder=3),
            ax.bar(x + w/2, multi,   w, color=BLUE,  label="multilingual (GPT-4o)", zorder=3)]
    top = max(general)
    for group in bars:
        for r in group:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + top*0.02,
                    f"{int(r.get_height())}", ha="center",
                    fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=12, color=DGRAY)
    ax.set_ylabel("tokens for the same sentence", fontsize=11, color=MGRAY)
    ax.set_ylim(0, top * 1.18)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("The Same Sentence Costs More in Some Languages",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_thinking_tokens(study, save_path="images/thinking/thinking_tokens.png"):
    """Where a thinking model's generation goes: hidden reasoning against the
    visible answer, with the standard model's whole reply beside it for scale.

    Bars are stacked so the full height is everything the model generated and
    the sliver on top is the only part the reader sees. The standard model has
    no hidden segment at all, which is the comparison the chapter is making.
    """
    labels = study["labels"]
    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor("white"); _grid_ax(ax)

    ax.bar(x - w / 2, study["fast_answer"], w, color=MGRAY, zorder=3,
           label=f"{short_model(study['fast'])}  (answer only)")
    ax.bar(x + w / 2, study["thinking"], w, color=PURPLE, zorder=3,
           label=f"{short_model(study['deep'])}  reasoning, never shown")
    ax.bar(x + w / 2, study["answer"], w, bottom=study["thinking"],
           color=GREEN, zorder=3, label=f"{short_model(study['deep'])}  answer")

    top = max(t + a for t, a in zip(study["thinking"], study["answer"]))
    for xi, (hidden, shown, snap) in enumerate(
            zip(study["thinking"], study["answer"], study["fast_answer"])):
        ax.text(xi + w / 2, hidden + shown + top * 0.025,
                f"{hidden:,} hidden + {shown} shown", ha="center",
                fontsize=9.5, fontweight="600", color=DGRAY)
        ax.text(xi - w / 2, snap + top * 0.025, f"{snap}", ha="center",
                fontsize=9.5, fontweight="600", color=MGRAY)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5, color=DGRAY)
    ax.set_ylabel("tokens generated", fontsize=11, color=MGRAY)
    ax.set_ylim(0, top * 1.20)
    ax.set_xlim(-0.62, len(labels) - 1 + 0.78)   # room for the wide bar labels
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("The Tokens You Pay For and Never Read", fontsize=13,
                 fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_book_map(save_path="images/book_map.png"):
    """The eleven chapters as one path, grouped into the book's three parts.

    Each chapter carries a short phrase for what it adds, echoing that
    chapter's own Abstract. The part headings match the divider pages
    (scripts/build_pdf.py PART_DIVIDERS) and the table of contents.
    """
    parts = [
        ("I", "Foundations", BLUE, [
            (1, "Introduction", "guessing the\nnext word"),
            (2, "Prompting", "the controls,\nand how to ask"),
            (3, "Tokens", "what it really\nreads"),
            (4, "Semantics", "meaning as a\nplace in space"),
        ]),
        ("II", "Engineering", TEAL, [
            (5, "Metacoding", "pointing it\nat code"),
            (6, "Augmentation", "something to\nlook up"),
            (7, "Agentic", "giving it\nhands"),
            (8, "Thinking", "letting it\nreason first"),
        ]),
        ("III", "Pragmatism", ORANGE, [
            (9, "Multimodal", "giving it eyes\nand ears"),
            (10, "Efficiency", "fast enough\nto use"),
            (11, "Responsible", "what goes wrong,\nand what to do"),
        ]),
    ]
    dx, hw, hh, pitch = 2.95, 1.30, 0.46, 3.55

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-0.5, 12.2); ax.set_ylim(-1.35, 2 * pitch + 2.3); ax.axis("off")

    for r, (roman, name, col, stops) in enumerate(parts):
        y = (len(parts) - 1 - r) * pitch
        # Two-line heading, matching the divider pages: small grey part number
        # over the part's name.
        ax.text(0.05, y + 1.82, f"PART {roman}", ha="left", va="center",
                fontsize=9, color=MGRAY)
        ax.text(0.05, y + 1.32, name, ha="left", va="center", fontsize=12.5,
                fontweight="700", color=col)
        xs = [0.05 + hw + i * dx for i in range(len(stops))]
        for i in range(len(stops) - 1):
            _arr(ax, xs[i] + hw, y, xs[i + 1] - hw, y, col=MGRAY, pct=0.10)
        for (num, chapter, gain), cx in zip(stops, xs):
            _fbox(ax, cx, y, hw, hh, col)
            _label(ax, cx, y, f"{num}.  {chapter}", tc="white", fs=11)
            ax.text(cx, y - 1.02, gain, ha="center", va="center", fontsize=9.5,
                    color=MGRAY, linespacing=1.45)

    ax.set_title("The Route Through This Book", fontsize=13,
                 fontweight="700", color=DGRAY, pad=4)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_document_clusters(docs, labels, vecs, width: int = 18,
                           save_path="images/semantics/document_clusters.png"):
    """The clustered documents as points, flattened to two dimensions with PCA.

    Colour comes from the cluster number k-means assigned, and the legend uses
    those same bare numbers, since nothing in the pipeline ever named a group.
    What the picture adds to the printed labels is the distance: the three
    clumps sit far apart, which is why the split came out clean.
    """
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    coords = PCA(n_components=2).fit_transform(np.array(vecs))
    palette = [BLUE, ORANGE, TEAL, PURPLE, GREEN]
    order = sorted(set(int(v) for v in labels))

    def short(text):
        out = ""
        for word in text.split():
            if len(out) + len(word) + 1 > width:
                return out + "…"
            out += (" " if out else "") + word
        return out

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.set_xticks([]); ax.set_yticks([])

    # Room for the point labels, which sit outside the cloud.
    (x0, y0), (x1, y1) = coords.min(0), coords.max(0)
    padx, pady = (x1 - x0) * 0.46, (y1 - y0) * 0.22
    ax.set_xlim(x0 - padx, x1 + padx); ax.set_ylim(y0 - pady, y1 + pady * 1.4)

    # A soft ellipse around each cluster's own members, so the grouping the
    # numbers describe is visible as a grouping.
    for k in order:
        pts = coords[np.array([int(v) == k for v in labels])]
        cx, cy = pts.mean(0)
        w = 2 * (np.abs(pts[:, 0] - cx).max() + padx * 0.42)
        h = 2 * (np.abs(pts[:, 1] - cy).max() + pady * 0.42)
        ax.add_patch(mp.Ellipse((cx, cy), w, h, facecolor=palette[k % len(palette)],
                                alpha=0.09, edgecolor="none", zorder=1))

    cols = [palette[int(v) % len(palette)] for v in labels]
    for (x, y), col in zip(coords, cols):
        ax.scatter(x, y, color=col, s=110, zorder=4)
    _place_labels(ax, coords, [short(d) for d in docs], fontsize=9.5)

    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", ms=10,
                              markerfacecolor=palette[k % len(palette)],
                              label=f"[{k}]") for k in order],
              fontsize=11, framealpha=0, loc="best", ncol=len(order))
    ax.set_title("Three Clusters  (PCA to 2D)",
                 fontsize=13, fontweight="700", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_fewshot_sweep(study, save_path="images/prompting/fewshot_sweep.png"):
    """Accuracy against the number of examples in the prompt, run twice: once
    with label names that explain themselves, once with in-house codes whose
    meaning lives only in the examples.

    The x axis carries a second row giving what the prompt costs in tokens at
    each shot count, so the flat stretch of a curve can be read against the
    tokens still being spent to get there.
    """
    shots, toks = study["shots"], study["tokens"]
    x = np.arange(len(shots))

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    fig.patch.set_facecolor("white"); _grid_ax(ax)

    # The two curves sit on top of each other once the taught one tops out, so
    # only the moving series carries value labels and the flat one goes dashed,
    # which keeps it readable underneath.
    series = [("known", "LOGISTICS / CONCEPTUAL / DEBUGGING", BLUE, "o", (0, (5, 2))),
              ("taught", "Q1 / Q2 / Q3", ORANGE, "s", "solid")]
    for key, label, col, mark, style in series:
        ys = [v * 100 for v in study[key]]
        ax.plot(x, ys, color=col, lw=2.2, marker=mark, ms=7, ls=style,
                label=label, zorder=3, clip_on=False)
        if key != "taught":
            continue
        for xi, y in zip(x, ys):
            # A label just under the flat line would collide with it; drop those
            # below their own marker instead.
            dy = -7.0 if 90 < y < 100 else 4.5
            ax.text(xi, y + dy, f"{y:.0f}", ha="center", fontsize=9.5,
                    fontweight="600", color=col, zorder=4)
    ax.text(-0.05, 104.5, "flat at 100 with no examples at all", ha="left",
            va="center", fontsize=9.5, color=BLUE, style="italic", zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in shots])
    ax.set_xlabel("examples in the prompt", fontsize=11, color=MGRAY, labelpad=18)
    ax.set_ylabel("share of 15 questions correct (%)", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.35, len(shots) - 0.65)

    # Second tick row: what those examples cost.
    for xi, t in zip(x, toks):
        ax.text(xi, -13.5, str(t), ha="center", va="center", fontsize=9.5,
                color=MGRAY, clip_on=False)
    ax.text(-0.62, -13.5, "tokens", ha="right", va="center", fontsize=9.5,
            color=MGRAY, style="italic", clip_on=False)

    ax.legend(frameon=False, fontsize=10, loc="lower right",
              bbox_to_anchor=(1.0, 0.06))
    ax.set_title("What Examples Are Actually For",
                 fontsize=13, fontweight="700", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_ban_pressure(study, save_path="images/prompting/ban_pressure.png"):
    """One panel per model: the banned sentence flowing downward, token by token,
    with a bar for how much probability sat on "elephant" at each step.

    Reading a panel top to bottom is reading the sentence as the model wrote it,
    so the step where a ban breaks is the step where the bar appears. The bar for
    a token the model actually emitted is drawn in red. Each panel header carries
    the same model's peak with no ban in the prompt, which is what the ban had to
    push down. A panel that stays flat never had the word in the running at all.
    """
    rows = study["models"]
    fig, axes = plt.subplots(1, len(rows), figsize=(8.0, 4.4))
    fig.patch.set_facecolor("white")

    for ax, row in zip(axes, rows):
        toks, press = row["tokens"], row["pressure"]
        said, n = row["said_at"], len(row["tokens"])
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot([0, 0], [-0.6, n - 0.4], color=LGRAY, lw=1.0, zorder=0)

        for i, (tok, p) in enumerate(zip(toks, press)):
            hit = i == said
            ax.barh(i, p, height=0.62, color=RED if hit else SLATE, zorder=3)
            ax.text(-0.04, i, tok.strip() or "·", ha="right", va="center",
                    fontsize=10.5, family="monospace", zorder=4,
                    color=RED if hit else DGRAY,
                    fontweight="bold" if hit else "normal")
            if p >= 0.02:
                ax.text(p + 0.04, i, f"{p:.2f}", ha="left", va="center",
                        fontsize=10.5, fontweight="700",
                        color=RED if hit else MGRAY, zorder=4)

        if max(press) < 0.01:
            ax.text(0.66, n / 2 - 0.5, "flat: the word never\nenters the running",
                    ha="center", va="center", fontsize=10, color=MGRAY,
                    style="italic", linespacing=1.5)

        ax.set_xlim(-0.85, 1.30); ax.set_ylim(n - 0.4, -0.9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{short_model(row['model'])}\nno ban: {row['free_peak']:.2f}",
                     fontsize=12, fontweight="700", linespacing=1.5,
                     color=model_color(row["model"]), pad=8)

    fig.suptitle("Where the Ban Breaks", fontsize=13, fontweight="700",
                 color=DGRAY, y=1.02)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_digit_chunking(a: int = 3947, b: int = 6281,
                        save_path="images/tokens/digit_chunking.png"):
    """Place-value columns above, real token chunks below, drawn on one grid.

    Column multiplication needs every digit reachable on its own and works from
    the right. The tokenizer chunks three digits at a time from the left, so the
    boundaries the algorithm depends on fall inside a token. Those swallowed
    boundaries are drawn as dashed red lines through the chunk that ate them.
    Chunks come from the real tokenizer, so the picture tracks whatever it does.
    """
    from genai.tokens import tokenize
    top, bot = str(a), str(b)
    n = max(len(top), len(bot))
    chunks = {top: tokenize(top), bot: tokenize(bot)}

    dw, hw, hh = 1.40, 0.52, 0.36            # column pitch, chip half-width/height
    xs = [0.9 + i * dw for i in range(n)]    # one x per digit column, left to right
    col = lambda s, i: xs[i + n - len(s)]    # right-align a short number under a long one
    right = xs[-1] + 1.5                     # where the annotation column starts
    palette = [BLUE, GREEN, PURPLE, ORANGE]

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    edge = right + 4.3          # annotations run past this; tight bbox crops to them
    ax.set_xlim(-0.9, edge); ax.set_ylim(-0.55, 7.05); ax.axis("off")

    def act(y_head, head, y_note, note, color=MGRAY):
        if head:
            ax.text(-0.7, y_head, head, ha="left", va="center", fontsize=12,
                    fontweight="700", color=DGRAY)
        ax.text(right, y_note, note, ha="left", va="center",
                fontsize=10.5, color=color, linespacing=1.6)

    # ── Act 1: the digits column multiplication needs ────────────────────────
    y1 = (5.35, 4.35)
    act(6.35, "What column multiplication needs", 4.85,
        "every digit reachable on its own,\nworking right to left")
    for place, lbl in enumerate(["1000s", "100s", "10s", "1s"][-n:]):
        ax.text(xs[place], 5.98, lbl, ha="center", va="center",
                fontsize=9, color=MGRAY, style="italic")
    for s, y in zip((top, bot), y1):
        for i, d in enumerate(s):
            _fbox(ax, col(s, i), y, hw, hh, "white", ec=MGRAY, lw=1.1)
            _label(ax, col(s, i), y, d, fs=14)
    ax.text(0.0, y1[1], "×", ha="center", va="center", fontsize=15, color=MGRAY)
    _arr(ax, xs[-1] + 0.55, 3.72, xs[0] - 0.55, 3.72, col=MGRAY, pct=0.02)
    ax.text((xs[0] + xs[-1]) / 2, 3.36, "start at the ones column",
            ha="center", va="center", fontsize=9.5, color=MGRAY, style="italic")

    ax.plot([-0.7, edge - 0.4], [2.85, 2.85], color=LGRAY, lw=1.2, zorder=0)

    # ── Act 2: the chunks the tokenizer actually hands over ──────────────────
    y2 = (1.55, 0.55)
    act(2.35, "What the model receives", 1.72,
        f"{chunks[top][0]} arrives fused: no way\nto ask it for the "
        f"{top[1]} or the {top[2]}")
    for s, y in zip((top, bot), y2):
        i = n - len(s)                       # first column this number occupies
        for j, piece in enumerate(chunks[s]):
            k = len(piece)                   # digits in this chunk
            cx = (xs[i] + xs[i + k - 1]) / 2
            _fbox(ax, cx, y, (k - 1) * dw / 2 + hw, hh, palette[j % len(palette)])
            _label(ax, cx, y, piece, tc="white", fs=14)
            for split in range(i + 1, i + k):  # place-value splits inside the chunk
                ax.plot([xs[split] - dw / 2] * 2, [y - hh - 0.2, y + hh + 0.2],
                        color=RED, lw=1.1, ls=(0, (2.4, 2.4)), zorder=5)
            i += k
    ax.text(0.0, y2[1], "×", ha="center", va="center", fontsize=15, color=MGRAY)

    act(None, None, 0.42, "dashed red marks a column boundary\nthe model never gets to use",
        color=RED)
    ax.set_title("The Digits the Model Never Sees",
                 fontsize=13, fontweight="700", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_code_scorecard(grid, save_path="images/metacoding/code_scorecard.png"):
    """Heatmap of per-category scores: model (rows) against task column, by mode.

    grid = [(model_label, [(solved, total), ...]), ...] in CODE_COLUMNS order, each
    cell the count of small tasks in that category the model fully solved. Cells run
    red (none) through orange and yellow to green (all); a category a model has no
    mode for (fill-in-the-middle on an instruction-only model) is gray and dashed.
    The point is the texture: different models light up different columns, so
    capability reads as a profile, not a single verdict.
    """
    from genai.code import CODE_COLUMNS
    labels = [g[0] for g in grid]
    cols   = [c for c, _m, _k in CODE_COLUMNS]
    modes  = [m for _c, m, _k in CODE_COLUMNS]
    n_rows, n_cols = len(labels), len(cols)

    frac  = np.full((n_rows, n_cols), np.nan)
    annot = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for i, (_, row) in enumerate(grid):
        for j, (solved, total) in enumerate(row):
            if solved is None:
                annot[i][j] = "—"
            else:
                frac[i, j]  = solved / total
                annot[i][j] = f"{solved}/{total}"

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(LGRAY)
    ax.imshow(np.ma.masked_invalid(frac), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = frac[i, j]
            tc = "white" if (val == val and val < 0.2) else DGRAY
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    fontsize=11, color=tc, fontweight="600")

    ax.set_xticks(range(n_cols)); ax.set_xticklabels(cols, fontsize=10.5)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    n_instr = modes.count("instruction")
    ax.axvline(n_instr - 0.5, color="white", lw=5)
    ax.text((n_instr - 1) / 2, -0.82, "Instruction (write the function)",
            ha="center", fontsize=10.5, fontweight="700", color=DGRAY)
    ax.text(n_instr + (n_cols - n_instr - 1) / 2, -0.82, "Fill-in-the-Middle",
            ha="center", fontsize=10.5, fontweight="700", color=DGRAY)
    ax.set_ylim(n_rows - 0.5, -1.35)

    ax.set_title("Code zoo scorecard: small tasks solved per category, by model",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=24)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_three_pass(save_path="images/research/three_pass.png"):
    """The three-pass reading method as a narrowing funnel.

    Each pass costs more time and admits fewer papers: every paper gets a
    10-minute first pass, the survivors earn a 30-minute second pass, and only
    the few you build on get the deep third pass. Boxes shrink left to right to
    make the funnel felt rather than stated.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 5); ax.axis("off")

    y = 2.6
    passes = [
        (4.0,  1.45, 1.02, BLUE,   "First Pass\n10 min",
         "title · abstract\nintro · conclusion"),
        (8.0,  1.40, 0.80, ORANGE, "Second Pass\n30 min",
         "figures · tables\nrelated work"),
        (12.0, 1.35, 0.56, GREEN,  "Third Pass\n1-3 hrs",
         "re-implement\nthe key ideas"),
    ]
    y_sub = 1.15
    for cx, hw, hh, fc, lbl, sub in passes:
        _fbox(ax, cx, y, hw, hh, fc)
        _label(ax, cx, y, lbl, tc="white", fs=11.5)
        ax.text(cx, y_sub, sub, ha="center", va="top", fontsize=9.0,
                color=MGRAY, style="italic", linespacing=1.25)

    # "every paper" feeds the first pass
    ax.text(1.0, y, "every\npaper", ha="center", va="center", fontsize=9.0,
            color=MGRAY, style="italic", linespacing=1.2)
    _arr(ax, 1.65, y, 2.55, y, pct=0.10)

    # narrowing arrows between passes, captioned with what survives each cut
    for x1, x2, cx, cap in [(5.45, 6.60, 6.02, "worth a\ncloser look"),
                            (9.40, 10.65, 10.02, "worth a\ndeep read")]:
        _arr(ax, x1, y, x2, y, pct=0.12)
        ax.text(cx, 4.10, cap, ha="center", va="center", fontsize=8.5,
                color=DGRAY, style="italic", linespacing=1.15)

    ax.set_title("The Three-Pass Method: Each Pass Costs More, Admits Fewer",
                 fontsize=12.5, fontweight="700", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


# ── Thinking chapter ──────────────────────────────────────────────────────────

def plot_thinking_latency(rows: list, save_path="images/thinking/latency.png"):
    """Grouped bars: time to answer for a standard vs a thinking model, per task.

    rows = [{"label": str, "std_s": float, "thk_s": float}, ...]
    The distance widens with difficulty: thinking is nearly free on a fact lookup and
    expensive on a multi-step puzzle, because the reasoning chain grows with it.
    """
    labels = [r["label"] for r in rows]
    std    = [r["std_s"] for r in rows]
    thk    = [r["thk_s"] for r in rows]
    x = np.arange(len(rows)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, std, w, color=BLUE,   zorder=3, label="standard model"),
            ax.bar(x + w/2, thk, w, color=ORANGE, zorder=3, label="thinking model")]
    top = max(std + thk)
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top*0.02,
                    f"{bar.get_height():.1f}s", ha="center", va="bottom",
                    fontsize=10, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, color=DGRAY)
    ax.set_ylabel("time to answer (s)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, top * 1.20)
    ax.legend(fontsize=10.5, framealpha=0, loc="upper left")
    ax.set_title("Thinking Is Far Slower, and Grows With the Problem",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_thinking_accuracy(labels: list, std_acc: list, thk_acc: list,
                           runs: int, save_path="images/thinking/accuracy.png",
                           series=("snap answer", "thinking"),
                           title="Novel Multi-Step Problems: Snap Answers vs Thinking"):
    """Grouped bars: accuracy (% correct over `runs` trials) on novel problems.

    std_acc / thk_acc are fractions in [0, 1], one per problem in `labels`.
    The standard model answers in a single snap; the thinking model reasons
    internally first. The distance shows up only where a snap answer is unsafe.
    """
    std = [100 * a for a in std_acc]
    thk = [100 * a for a in thk_acc]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, std, w, color=BLUE,   zorder=3, label=series[0]),
            ax.bar(x + w/2, thk, w, color=ORANGE, zorder=3, label=series[1])]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {runs} runs (%)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title(title,
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_self_consistency(study: dict,
                          save_path="images/thinking/self_consistency.png"):
    """Accuracy as a majority vote widens from 1 to N sampled reasoning chains.

    ``study`` carries n_values and a {problem: [accuracy per n]} curve map. A
    vote climbs where the model wobbles toward the right answer (the curve starts
    above the noise and rises), is redundant where the model is already sure (flat
    along the top), and powerless where it has no real signal (flat near the floor).
    """
    from matplotlib.ticker import PercentFormatter
    n = study["n_values"]
    colors = {"Trains": BLUE, "Ages": GREEN, "Handshakes": PURPLE, "Well": RED}
    cycle = [GREEN, PURPLE, BLUE, RED, ORANGE]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    for i, (name, curve) in enumerate(study["curves"].items()):
        col = colors.get(name, cycle[i % len(cycle)])
        ax.plot(n, curve, color=col, marker="o", lw=2.2, ms=7, zorder=3)
        ax.text(n[-1] + 0.15, curve[-1], f" {name}", color=col,
                va="center", ha="left", fontsize=11, fontweight="600")

    ax.set_xlabel("reasoning chains sampled, then voted", fontsize=11, color=MGRAY)
    ax.set_ylabel("accuracy", fontsize=11, color=MGRAY)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.6, n[-1] + 1.9)
    ax.set_xticks(n)
    ax.set_title("Thinking Wider: Voting Helps Only Where the Model Wobbles",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_step_back(study: dict, save_path="images/prompting/step_back.png"):
    """Grouped bars: accuracy of a cold answer vs a step-back answer per problem.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]).
    Step-back climbs where the model knows the principle but its snap answer
    follows the wrong intuition (Half, Pendulum), helps only partway where the
    arithmetic still trips it (Pressure), and is redundant where the cold answer
    is already right (Inverse).
    """
    labels = study["labels"]
    direct = [100 * a for a in study["direct"]]
    step = [100 * a for a in study["step_back"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, direct, w, color=BLUE,   zorder=3, label="cold answer"),
            ax.bar(x + w/2, step,   w, color=ORANGE, zorder=3, label="step-back")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {study['samples']} tries (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Step-Back Prompting: Naming the Principle Before Answering",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_pal(study: dict, save_path="images/prompting/pal.png"):
    """Grouped bars: accuracy reasoning in prose vs writing and running a program.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]). The
    program-aided bar climbs where the problem is easy to translate to code but
    the arithmetic is gnarly enough that prose reasoning slips (the wins), ties at
    the ceiling on problems the model can already grind out in its head, and stays
    stuck where the model mis-models the problem so the program runs to a confident
    wrong number (the honest limit).
    """
    labels = study["labels"]
    cot = [100 * a for a in study["cot"]]
    pal = [100 * a for a in study["pal"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, cot, w, color=BLUE,   zorder=3, label="in prose"),
            ax.bar(x + w/2, pal, w, color=ORANGE, zorder=3, label="as a program")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"accuracy over {study['samples']} tries (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Program-Aided LMs: Let the Interpreter Do the Arithmetic",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_cove(study: dict, save_path="images/augmentation/cove.png"):
    """Two panels: claim precision and recall, before (draft) and after (CoVe).

    ``study`` carries labels and four lists (fractions in [0, 1]): draft/cove
    precision and draft/cove recall. The two panels have to be read together. CoVe
    raises precision where the draft over-reaches and the factored check is sound
    (Portugal, Mexico) while leaving recall alone; where the model is confidently
    wrong on the check itself (Brazil) precision cannot climb (the draft was already
    precise) and recall instead collapses -- the boundary where retrieval, not more
    self-questioning, is the real fix.
    """
    labels = study["labels"]
    x = np.arange(len(labels)); w = 0.36
    panels = [("Precision: are the listed items correct?",
               study["draft_precision"], study["cove_precision"]),
              ("Recall: did we keep the correct ones?",
               study["draft_recall"], study["cove_recall"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    for ax, (title, before, after) in zip(axes, panels):
        _grid_ax(ax)
        b = [100 * a for a in before]; a2 = [100 * a for a in after]
        bars = [ax.bar(x - w/2, b,  w, color=BLUE,   zorder=3, label="draft"),
                ax.bar(x + w/2, a2, w, color=ORANGE, zorder=3, label="after CoVe")]
        for group in bars:
            for bar in group:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f"{bar.get_height():.0f}", ha="center", va="bottom",
                        fontsize=9, fontweight="600", color=DGRAY)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
        ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=11.5, fontweight="600", color=DGRAY, pad=6)
    axes[0].set_ylabel(f"percent (over {study['samples']} tries)",
                       fontsize=11, color=MGRAY)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, fontsize=10, framealpha=0, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Chain-of-Verification: Buying Precision, Sometimes With Recall",
                 fontsize=13.5, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout(pad=1.0, rect=(0, 0.06, 1, 1))
    _save(fig, save_path)
    plt.show()


def plot_crag(study: dict, save_path="images/augmentation/crag.png"):
    """Grouped bars: answer accuracy with plain RAG vs with corrective RAG, per query.

    ``study`` carries labels and two accuracy lists (fractions in [0, 1]). On the
    queries the local corpus can answer, the retrieval evaluator grades CORRECT and
    CRAG ties the baseline, taking no web detour and doing no harm. On the trap queries
    the corpus cannot answer, plain RAG either begs off or makes something up while CRAG
    grades the retrieval INCORRECT, falls back to a web search, and recovers the fact.
    The whole distance is the traps, which is the point: correction buys robustness to
    retrieval failure, not omniscience.
    """
    labels = study["labels"]
    base = [100 * a for a in study["standard"]]
    crag = [100 * a for a in study["crag"]]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = [ax.bar(x - w/2, base, w, color=BLUE,   zorder=3, label="standard RAG"),
            ax.bar(x + w/2, crag, w, color=ORANGE, zorder=3, label="corrective RAG")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel(f"answers with the correct fact (% of {study['samples']})",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Corrective RAG: Hold Steady When Retrieval Works, Recover When It Fails",
                 fontsize=11.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_casting_summary(card: dict, save_path="images/agentic/casting_summary.png"):
    """The whole season on one card: models as rows, auditions as columns. Each
    column is shaded against its own field (calibration inverts, since claimed
    confidence on an impossible question should be low) and the best score in the
    column is starred. A dash means the model never read for that part. Mixed
    yardsticks on purpose: every column keeps the metric its own section judged
    by, so the card summarizes the chapter rather than inventing a new benchmark."""
    from matplotlib.colors import LinearSegmentedColormap
    models, cols = card["models"], card["columns"]
    n_r, n_c = len(models), len(cols)
    shade = LinearSegmentedColormap.from_list("card", ["#F7F9FE", BLUE])

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.75, n_c + 0.05); ax.set_ylim(n_r + 0.9, -1.75); ax.axis("off")

    for j, col in enumerate(cols):
        vals = col["values"]
        present = [vals[m] for m in models if m in vals]
        lo, hi = min(present), max(present)
        best = lo if col["lower_better"] else hi
        for i, m in enumerate(models):
            if m not in vals:                       # never auditioned for this part
                ax.add_patch(mp.Rectangle((j, i), 1, 1, facecolor=LGRAY,
                                          edgecolor="white", lw=1.5))
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                        fontsize=10, color=MGRAY)
                continue
            v = vals[m]
            t = 0.5 if hi == lo else (v - lo) / (hi - lo)
            t = 1 - t if col["lower_better"] else t
            ax.add_patch(mp.Rectangle((j, i), 1, 1, facecolor=shade(t),
                                      edgecolor="white", lw=1.5))
            star = v == best
            ax.text(j + 0.5, i + 0.5, ("★" if star else "") + col["fmt"].format(v),
                    ha="center", va="center", fontsize=9.5,
                    fontweight="700" if star else "500",
                    color="white" if t > 0.55 else DGRAY)
        ax.text(j + 0.45, -0.18, col["label"], ha="left", va="bottom",
                fontsize=9.5, fontweight="600", color=DGRAY, rotation=24)
    for i, m in enumerate(models):
        ax.text(-0.12, i + 0.5, short_model(m), ha="right", va="center",
                fontsize=10.5, fontweight="700", color=model_color(m))
    has_dash = any(m not in col["values"] for col in cols for m in models)
    legend = "★ best score in the column"
    if has_dash:                                   # only explain the dash if one shows
        # Covers both a model that never ran the event and one that ran it and
        # produced nothing scorable (functiongemma never rates its confidence).
        legend += "   ·   — no result to report"
    legend += ("\ncalibration = confidence claimed on impossible questions "
               "(1-5, lower is better)")
    ax.text(0, n_r + 0.42, legend,
            ha="left", va="top", fontsize=8.5, color=MGRAY)
    ax.set_title("Who Won What? The Season on One Card",
                 fontsize=13, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_spelling_survey(study, save_path="images/tokens/spelling_survey.png"):
    """A letter-counting survey as a grid: tricky words down the rows, models
    across the columns, each cell the model's single greedy answer. Green where it
    matches the true count in the left column, soft red where it misses, with each
    model's tally along the bottom. Most of the grid is red, because the repeated
    letters stay sealed inside chunks no model can see into, so more capability
    buys confident wrong answers rather than correct ones."""
    rows, models = study["rows"], study["models"]
    n_r, n_c = len(rows), len(models)
    miss_fill = "#FBE3E4"

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-2.7, n_c + 1.05); ax.set_ylim(n_r + 0.85, -1.0); ax.axis("off")

    for i, row in enumerate(rows):
        true = row["true"]
        ax.text(-0.16, i + 0.5, f"{row['letter']} in {row['word']}", ha="right",
                va="center", fontsize=10.5, fontweight="600", color=DGRAY)
        ax.add_patch(mp.Rectangle((0, i), 1, 1, facecolor=LGRAY,
                                  edgecolor="white", lw=2.5))
        ax.text(0.5, i + 0.5, str(true), ha="center", va="center",
                fontsize=11.5, fontweight="700", color=DGRAY)
        for j, ans in enumerate(row["answers"]):
            ok = ans == true
            ax.add_patch(mp.Rectangle((j + 1, i), 1, 1,
                                      facecolor=GREEN if ok else miss_fill,
                                      edgecolor="white", lw=2.5))
            ax.text(j + 1.5, i + 0.5, "?" if ans is None else str(ans),
                    ha="center", va="center", fontsize=11.5,
                    fontweight="700" if ok else "500",
                    color="white" if ok else RED)

    ax.text(0.5, -0.16, "true", ha="center", va="bottom", fontsize=10,
            fontweight="700", color=MGRAY)
    for j, m in enumerate(models):
        ax.text(j + 1.5, -0.16, short_model(m), ha="center", va="bottom",
                fontsize=10, fontweight="700", color=model_color(m))
        hits = sum(r["answers"][j] == r["true"] for r in rows)
        ax.text(j + 1.5, n_r + 0.12, f"{hits}/{n_r}", ha="center", va="top",
                fontsize=9.5, fontweight="600", color=MGRAY)
    ax.text(-0.16, n_r + 0.12, "correct", ha="right", va="top",
            fontsize=9.5, fontstyle="italic", color=MGRAY)
    ax.set_title("Counting Letters the Tokenizer Hides", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=12)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_plan(study: dict, save_path="images/agentic/plan.png"):
    """One bar per strategy: the share of plans that come out fully valid under
    every dependency after a new constraint arrives.

    Editing the draft in place satisfies the new rule but regresses on an old one
    about half the time; re-deriving the plan from the full constraint set is far
    better but still fumbles a fresh sort one time in five; only re-checking the
    whole plan and bouncing each broken edge back reaches every-time validity. The
    bars climb left to right from patch to rebuild to checked.
    """
    labels = study["labels"]
    vals = [100 * v for v in study["valid"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, vals, 0.55, color=[ORANGE, BLUE, GREEN], zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="600", color=DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color=DGRAY)
    ax.set_ylabel(f"plans fully valid over {study['samples']} runs (%)",
                  fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Keeping a Plan Valid: Edit, Re-derive, or Check",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Dispatch contest: over- and under-routing against the golden answer ───────
def _router_label(m: str) -> str:
    """Keep the size tag on the gemma4 router so it reads apart from the e2b/26b
    resolvers it shares a family with; shorten everyone else."""
    return m if m.startswith("gemma4") else short_model(m)


def plot_dispatch(picks, tasks, save_path="images/agentic/dispatch_contest.png"):
    """Each router's twelve picks scored against the gold tier, one stacked bar per
    model: under-routed (sent too weak, the task fails -- a quality cost, red, on
    the left), correct (right size, green), over-routed (sent too big, solved but
    wasteful -- a time cost, amber, on the right). Sorted by correct with the best
    dispatcher on top, so the ideal router is almost all green, the lazy all-HARD
    router all amber, the weak ones all red. ``picks`` is ``{model: {task_id:
    tier}}``; ``tasks`` carry the gold ``tier``."""
    rank = {"easy": 0, "medium": 1, "hard": 2}
    gold = {t["id"]: rank[t["tier"]] for t in tasks}
    n = len(tasks)

    def split(p):
        u = c = o = 0
        for tid, tier in p.items():
            d = rank[tier] - gold[tid]
            u, c, o = (u + 1, c, o) if d < 0 else (u, c + 1, o) if d == 0 else (u, c, o + 1)
        return u, c, o

    rows = {m: split(p) for m, p in picks.items()}
    models = sorted(rows, key=lambda m: (rows[m][1], -rows[m][0]))   # best ends on top
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    left = np.zeros(len(models))
    for k, col in [(0, RED), (1, GREEN), (2, AMBER)]:
        vals = np.array([rows[m][k] for m in models])
        ax.barh(y, vals, left=left, height=0.62, color=col, zorder=3,
                edgecolor="white", linewidth=1.4)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        fontsize=10, fontweight="700", color="white")
        left += vals

    ax.set_yticks(y); ax.set_yticklabels([_router_label(m) for m in models],
                                         fontsize=10.5, color=DGRAY)
    ax.set_xlim(0, n); ax.set_xticks(range(0, n + 1, 2))
    ax.tick_params(colors=MGRAY, labelsize=10)
    ax.set_xlabel("tasks routed (of 12)", fontsize=11, color=MGRAY)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)

    handles = [mp.Patch(color=c, label=l) for l, c in
               (("under-routed (fails)", RED), ("correct", GREEN),
                ("over-routed (wasteful)", AMBER))]
    ax.legend(handles=handles, ncol=3, fontsize=9.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("The dispatcher contest, scored against the golden routing",
                 fontsize=13, fontweight="600", color=DGRAY, pad=30)

    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()


def plot_sql_boundary(save_path="images/datasql/sql_boundary.png"):
    """The model-writes-SQL / database-runs-it boundary, with the silent-wrong space.

    Deterministic. A question in words crosses into the model (orange), which
    writes SQL; the database runs it exactly and hands back a number. The danger
    is marked below the pipe: 'runs' is not 'correct', and nothing on the page
    tells them apart."""
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    ax.text(0.9, 4.2, "question\n(in words)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.7, 4.2, 2.55, 4.2)

    _fbox(ax, 4.0, 4.2, 1.4, 0.8, ORANGE)
    _label(ax, 4.0, 4.2, "Model\nwrites SQL", tc="white", fs=10.5)
    ax.text(4.0, 2.9, "picks the joins\nand the metric", ha="center", fontsize=8.6,
            style="italic", color=MGRAY, linespacing=1.2)

    _arr(ax, 5.4, 4.2, 6.55, 4.2)
    ax.text(5.97, 4.55, "SQL", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 8.0, 4.2, 1.4, 0.8, SLATE)
    _label(ax, 8.0, 4.2, "Database\nruns it", tc="white", fs=10.5)
    ax.text(8.0, 2.9, "exact, asks\nno questions", ha="center", fontsize=8.6,
            style="italic", color=MGRAY, linespacing=1.2)

    _arr(ax, 9.4, 4.2, 10.3, 4.2)
    ax.text(11.1, 4.2, "a number", ha="center", va="center",
            fontsize=10, color=DGRAY)

    # the silent-wrong space, marked under the boundary
    ax.add_patch(mp.FancyBboxPatch((2.4, 0.65), 7.2, 1.25,
                 boxstyle="round,pad=0.10", facecolor="#FEF2F2",
                 edgecolor=RED, linewidth=1.1, zorder=2))
    ax.text(6.0, 1.55, "the silent-wrong space", ha="center", fontsize=9.6,
            fontweight="700", color=RED)
    ax.text(6.0, 1.02, "wrong table, wrong metric, returned orders left in:\n"
            "the query still runs and the number still looks right",
            ha="center", fontsize=8.5, color=DGRAY, linespacing=1.25)
    _arr(ax, 4.1, 2.45, 4.5, 1.95, col=RED, ls="dashed")
    _arr(ax, 7.9, 2.45, 7.5, 1.95, col=RED, ls="dashed")

    ax.set_title("The model writes the query; nobody checks the meaning",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
# ── Low-Level Systems chapter ─────────────────────────────────────────────────

def plot_kernel_loop(save_path="images/lowlevel/kernel_loop.png"):
    """The generate-profile-regenerate search loop a model runs to write fast code.

    The model proposes a kernel (orange), code runs and profiles it, and a check
    decides: ship it if it's correct and faster, otherwise feed the verdict back
    and ask for another. The loop is what turns one lucky guess into a search."""
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0.8, 5.2); ax.axis("off")

    ax.text(0.55, 4.0, "slow\nroutine", ha="center", va="center",
            fontsize=9.5, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.15, 4.0, 1.45, 4.0)

    _fbox(ax, 2.5, 4.0, 1.05, 0.56, ORANGE)
    _label(ax, 2.5, 4.0, "Generate\n(model)", tc="white")
    _fbox(ax, 6.2, 4.0, 1.10, 0.56, LGRAY, ec=BLUE, lw=1.0)
    _label(ax, 6.2, 4.0, "Run &\nprofile", tc=DGRAY)
    _fbox(ax, 6.2, 1.9, 1.20, 0.56, LGRAY, ec=BLUE, lw=1.0)
    _label(ax, 6.2, 1.9, "correct &\nfaster?", tc=DGRAY)
    _fbox(ax, 9.0, 1.9, 0.85, 0.42, GREEN)
    _label(ax, 9.0, 1.9, "ship it", tc="white")

    _arr(ax, 3.55, 4.0, 5.10, 4.0)        # generate -> run
    _arr(ax, 6.2, 3.44, 6.2, 2.46)        # run -> check
    _arr(ax, 7.4, 1.9, 8.15, 1.9, col=GREEN)   # check -> ship (yes)
    ax.text(7.78, 2.22, "yes", color=GREEN, fontsize=9, ha="center", style="italic")
    _arr(ax, 5.30, 1.62, 2.7, 3.46, col=ORANGE, ls="dashed")   # check -> generate (no)
    ax.text(3.55, 2.18, "no: wrong\nor too slow", color=ORANGE, fontsize=9,
            ha="center", va="center", style="italic", linespacing=1.2)

    ax.set_title("Generate, profile, regenerate", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
# ── Creativity: palettes, melodies, and the self-evaluation spread ───────────────

def _mood_label(brief: str) -> str:
    """Fallback short label for a brief when a row carries no explicit ``label``:
    drop a leading article. 'a warm, energetic sports brand' -> 'warm, energetic
    sports brand'."""
    for article in ("a ", "an ", "the "):
        if brief.startswith(article):
            return brief[len(article):]
    return brief


def _swatch_row(ax, palette, y, h=1.0):
    """Draw one row of colour chips with their hex codes beneath."""
    for i, color in enumerate(palette):
        ax.add_patch(mp.Rectangle((i, y), 0.92, h, facecolor=color,
                                  edgecolor="white", linewidth=2, zorder=3))
        ax.text(i + 0.46, y - 0.16, color, ha="center", va="top",
                fontsize=8.5, color=MGRAY, family="monospace")


def plot_palette(demo: dict, save_path="images/creativity/palette.png"):
    """A single model-generated palette as a row of colour chips.

    ``demo`` is ``{"brief", "palette"}``: the mood the model was asked for and the
    hex codes it returned. Pure rendering of a baked capture, deterministic.
    """
    palette = demo["palette"]
    fig, ax = plt.subplots(figsize=(1.5 * len(palette), 2.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _swatch_row(ax, palette, 0)
    ax.set_xlim(-0.15, len(palette)); ax.set_ylim(-0.65, 1.55)
    ax.axis("off")
    ax.set_title(demo["brief"], fontsize=12, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_sql_audit(audit, save_path="images/datasql/sql_audit.png"):
    """Stacked composition of model SQL outcomes: free-form vs governed layer.

    audit = {"raw": {...}, "semantic": {...}} with correct/silent_wrong/crash/n.
    Silent-wrong is the hero, in red: it shrinks when the layer fixes the metric
    definitions the model kept getting wrong."""
    groups = [("free-form\nSQL", audit["raw"]),
              ("governed\nsemantic layer", audit["semantic"])]
    parts = [("correct", GREEN), ("silent_wrong", RED), ("crash", MGRAY)]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)

    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))
    for key, col in parts:
        vals = np.array([100 * g[key] / g["n"] for _, g in groups])
        ax.bar(x, vals, bottom=bottoms, color=col, width=0.5, zorder=3,
               edgecolor="white", linewidth=1.2)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 6:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=11, fontweight="700",
                        color="white" if key != "crash" else DGRAY)
        bottoms += vals

    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups],
                                         fontsize=11, color=DGRAY)
    ax.set_ylim(0, 100); ax.set_ylabel("share of queries (%)")
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [mp.Patch(color=c, label=l.replace("_", "-")) for l, c in parts]
    ax.legend(handles=handles, ncol=3, fontsize=9.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Free-form SQL runs and lies; the governed layer doesn't",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=28)
    plt.tight_layout(pad=0.7)
# ── Planning chapter ──────────────────────────────────────────────────────────

def plot_llm_modulo(save_path="images/planning/llm_modulo.png"):
    """The LLM-Modulo generate-and-check loop. The model (orange) translates a goal
    in words into a formal problem; a classical planner and a verifier (teal)
    search for a plan and replay it; a problem that yields no plan or fails the
    check loops back for a rewrite. The model is the only fallible part, the same
    division of labour as the neuro-symbolic split in the Automated Reasoning
    chapter, now aimed at a goal instead of a riddle."""
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(1.0, 5.4); ax.axis("off")

    ax.text(0.7, 4.2, "goal\n(in words)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.6, 4.2, 2.45, 4.2)

    _fbox(ax, 3.95, 4.2, 1.5, 0.72, ORANGE)
    _label(ax, 3.95, 4.2, "Model\nproposes", tc="white", fs=10.5)
    ax.text(3.95, 3.18, "fluent, can mis-translate", ha="center", fontsize=8.6,
            style="italic", color=MGRAY)

    _arr(ax, 5.45, 4.2, 6.45, 4.2)
    ax.text(5.95, 4.6, "problem", ha="center", fontsize=8.6, color=DGRAY)

    _fbox(ax, 8.0, 4.2, 1.55, 0.72, TEAL)
    _label(ax, 8.0, 4.2, "Planner +\nVerifier", tc="white", fs=10.5)
    ax.text(8.0, 3.18, "searches, then checks", ha="center", fontsize=8.6,
            style="italic", color=MGRAY)

    _arr(ax, 9.55, 4.2, 10.4, 4.2)
    ax.text(11.3, 4.2, "a plan\nyou can run", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)

    # the refine loop: no plan, or a plan that fails the replay, goes back
    ax.annotate("", xy=(3.95, 2.78), xytext=(8.0, 2.78),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                mutation_scale=12, linestyle="dashed",
                                connectionstyle="arc3,rad=-0.32"))
    ax.text(5.97, 1.42, "no plan, or fails the check  →  refine", ha="center",
            fontsize=9, style="italic", color=ORANGE)

    ax.set_title("Generate, Then Check: the LLM-Modulo Loop",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_writer_mutator_loop(save_path="images/testing/writer_mutator_loop.png"):
    """The writer/mutator loop: the model writes tests, the mutator scores them,
    and the surviving mutants feed back as the next round's to-do list."""
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0.6, 6); ax.axis("off")

    ax.text(0.7, 4.4, "spec", ha="center", va="center", fontsize=10, color=DGRAY)
    _arr(ax, 1.25, 4.4, 2.15, 4.4)

    _fbox(ax, 3.4, 4.4, 1.25, 0.72, ORANGE)
    _label(ax, 3.4, 4.4, "Writer\n(model)", tc="white", fs=10.5)

    _arr(ax, 4.7, 4.4, 5.6, 4.4)
    ax.text(5.15, 4.74, "tests", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 6.85, 4.4, 1.25, 0.72, LGRAY, ec=BLUE, lw=1.0)
    _label(ax, 6.85, 4.4, "Test\nsuite", tc=DGRAY, fs=10.5)

    _arr(ax, 8.15, 4.4, 9.05, 4.4)
    ax.text(8.6, 4.74, "run", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 10.2, 4.4, 1.0, 0.72, TEAL)
    _label(ax, 10.2, 4.4, "Mutator", tc="white", fs=10)

    _arr(ax, 11.25, 4.4, 12.05, 4.4)
    ax.text(12.5, 4.4, "mutation\nscore", ha="center", va="center", fontsize=9,
            style="italic", color=MGRAY, linespacing=1.2)

    # the survivors feed back: a clean rectangular path under the row, one arrowhead
    ax.plot([10.2, 10.2, 3.4], [4.0, 1.75, 1.75], color=ORANGE, ls="--",
            lw=1.5, zorder=2)
    ax.annotate("", xy=(3.4, 4.0), xytext=(3.4, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5,
                                mutation_scale=12, shrinkA=0, shrinkB=0,
                                linestyle="--"))
    ax.text(6.8, 1.4, "surviving mutants: the cases the tests still miss",
            ha="center", fontsize=9, style="italic", color=ORANGE)

    ax.set_title("The model writes the tests; the mutator grades them",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.4)


# ── Refactoring chapter ───────────────────────────────────────────────────────

def plot_differential_oracle(save_path="images/refactoring/differential_oracle.png"):
    """The refactor oracle. Run the old code and the migrated code on the same
    inputs and demand the same answers: the old program is the answer key, so no
    spec is needed, and any input where the two part ways is a behavior the edit
    changed. The existing test suite is the separate, weaker check off to the
    side, the one that only ever sees the inputs someone already wrote down."""
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    _fbox(ax, 1.5, 3.0, 1.15, 0.62, LGRAY, ec=MGRAY, lw=1.1)
    _label(ax, 1.5, 3.0, "same\ninputs", tc=DGRAY, fs=9.8)

    _fbox(ax, 4.7, 4.5, 1.5, 0.66, SLATE)
    _label(ax, 4.7, 4.5, "old code", tc="white", fs=10.5)
    _fbox(ax, 4.7, 1.5, 1.7, 0.66, ORANGE)
    _label(ax, 4.7, 1.5, "migrated code", tc="white", fs=10.5)

    _arr(ax, 2.65, 3.25, 3.95, 4.4)            # inputs -> old
    _arr(ax, 2.65, 2.75, 3.95, 1.6)            # inputs -> migrated

    ax.text(7.0, 4.06, "answers", ha="center", fontsize=8.4, color=MGRAY)
    ax.text(7.05, 1.94, "answers", ha="center", fontsize=8.4, color=MGRAY)
    _arr(ax, 5.55, 4.5, 7.65, 3.42)            # old -> compare
    _arr(ax, 5.65, 1.5, 7.65, 2.58)            # migrated -> compare

    _fbox(ax, 8.6, 3.0, 1.35, 0.66, TEAL)
    _label(ax, 8.6, 3.0, "compare", tc="white", fs=10.5)

    _arr(ax, 9.95, 3.25, 10.75, 3.95, col=GREEN)
    ax.text(11.75, 4.05, "all match:\nbehavior kept", ha="center", va="center",
            fontsize=9, color=GREEN, linespacing=1.2)
    _arr(ax, 9.95, 2.75, 10.75, 2.05, col=RED)
    ax.text(11.75, 1.9, "any differ:\na regression", ha="center", va="center",
            fontsize=9, color=RED, linespacing=1.2)

    ax.set_title("Differential testing: the old program is the answer key",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)


def plot_refactor_study(study, save_path="images/refactoring/refactor_study.png"):
    """Three verdicts on the same batch of migrations, over many trials. The
    existing suite passes on almost all of them; running the old and new code side
    by side, and counting the call sites that actually moved, is what exposes the
    runs the green bar called done and the oracles called unfinished. The last bar
    is the only one that clears every check; the distance up to the suite bar is what a
    green suite quietly hides on a cross-cutting change."""
    labels = ["existing suite\npasses", "every site\nmigrated",
              "behavior\npreserved", "complete and\npreserving"]
    vals = [study["suite_green"] * 100, study["complete"] * 100,
            study["behavior_preserved"] * 100, study["fully_correct"] * 100]
    colors = [BLUE, AMBER, TEAL, GREEN]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.0f}%",
                ha="center", fontsize=12.5, fontweight="700", color=DGRAY)
    ax.set_ylim(0, 116)
    ax.set_ylabel("share of migrations (%)", fontsize=11, color=MGRAY)

    # the distance the green bar hides: from the suite bar down to fully-correct
    hidden_pct = round(study["green_but_unfinished"] * 100)
    if hidden_pct:
        ax.annotate("", xy=(3, vals[3] + 1), xytext=(3, vals[0] - 1),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6,
                                    mutation_scale=12, shrinkA=0, shrinkB=0))
        ax.text(2.84, (vals[0] + vals[3]) / 2, f"{hidden_pct}% looked done\nto the suite",
                ha="right", va="center", fontsize=9.6, fontweight="700",
                color=RED, linespacing=1.2)
    ax.set_title(f"A Green Suite Doesn't Mean the Job Is Done  "
                 f"(n={study['trials']} migrations)",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)


# ── Autonomy chapter ──────────────────────────────────────────────────────────

def plot_coordination(save_path="images/autonomy/coordination.png"):
    """The race versus the fix: two agents writing one file, with and without a lock.

    Left, the unsynchronized version: both sub-agents write the shared notes file
    at once and one finding is lost. Right, the coordinated version: a lock makes
    them take turns, so both findings survive. The agents are orange (the
    non-deterministic model); the file and the lock are deterministic code.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_facecolor("white")

    def agents(ax):
        _fbox(ax, 2.4, 8.2, 1.25, 0.6, ORANGE); _label(ax, 2.4, 8.2, "agent 1", tc="white")
        _fbox(ax, 7.6, 8.2, 1.25, 0.6, ORANGE); _label(ax, 7.6, 8.2, "agent 2", tc="white")

    # Left: the race
    g = axes[0]
    agents(g)
    _fbox(g, 5.0, 2.0, 1.7, 0.62, LGRAY, ec=RED, lw=1.3)
    _label(g, 5.0, 2.0, "notes.json", tc=DGRAY)
    _arr(g, 2.4, 7.6, 4.6, 2.55, col=RED)            # agent 1 -> file
    _arr(g, 7.6, 7.6, 5.4, 2.55, col=RED)            # agent 2 -> file (they cross)
    g.scatter([5.0], [5.1], s=240, marker="X", color=RED, zorder=5)
    g.text(5.0, 4.2, "collision", color=RED, fontsize=9.5, style="italic", ha="center")
    g.text(5.0, 0.85, "one finding overwrites the other", color=RED,
           fontsize=9.2, ha="center")
    g.set_title("Unsynchronized", fontsize=12.5, fontweight="700", color=RED, pad=4)

    # Right: the fix
    r = axes[1]
    agents(r)
    _fbox(r, 5.0, 5.0, 1.35, 0.55, TEAL); _label(r, 5.0, 5.0, "lock", tc="white")
    _fbox(r, 5.0, 2.0, 1.7, 0.62, LGRAY, ec=GREEN, lw=1.3)
    _label(r, 5.0, 2.0, "notes.json", tc=DGRAY)
    _arr(r, 2.4, 7.6, 4.5, 5.4, col=MGRAY)           # agent 1 -> lock
    _arr(r, 7.6, 7.6, 5.5, 5.4, col=MGRAY)           # agent 2 -> lock
    _arr(r, 5.0, 4.45, 5.0, 2.65, col=GREEN)         # lock -> file (one at a time)
    r.text(6.55, 5.0, "one at a time", color=TEAL, fontsize=9.2,
           style="italic", ha="left", va="center")
    r.text(5.0, 0.85, "both findings survive", color=GREEN, fontsize=9.2, ha="center")
    r.set_title("Coordinated", fontsize=12.5, fontweight="700", color=GREEN, pad=4)

def plot_kernel_search(trace, save_path="images/lowlevel/kernel_search.png"):
    """Speedup against the optimized baseline, attempt by attempt. A dashed line
    marks the baseline the kernel has to beat: orange bars are correct but too
    slow, the red bar is a fast-but-wrong attempt that scores nothing, and the
    green bar is the one that finally clears the line. ``trace`` is KERNEL_SEARCH."""
    attempts = [r["attempt"] for r in trace]
    speeds = [r["speedup"] for r in trace]
    notes = [r.get("note", "") for r in trace]
    colors = [RED if not r["correct"] else (GREEN if r["speedup"] >= 1.0 else ORANGE)
              for r in trace]
    x = np.arange(len(trace))

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, speeds, width=0.6, color=colors, zorder=3)
    ax.axhline(1.0, color=DGRAY, ls="--", lw=1.3, zorder=2)
    ax.text(-0.45, 1.03, "baseline (must beat)", ha="left",
            va="bottom", fontsize=9.5, color=DGRAY, style="italic")
    for bar, r, note in zip(bars, trace, notes):
        top = bar.get_height()
        if r["correct"]:
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.03,
                    f"{r['speedup']:.2f}x", ha="center", va="bottom",
                    fontsize=10.5, fontweight="700", color=DGRAY)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.06, "wrong",
                    ha="center", va="bottom", fontsize=10.5, fontweight="700",
                    color=RED)
        ax.text(bar.get_x() + bar.get_width() / 2, -0.14, note, ha="center",
                va="top", fontsize=8.2, color=MGRAY, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"attempt {a}" for a in attempts], fontsize=10, color=DGRAY)
    ax.set_ylabel("speedup vs. optimized baseline", fontsize=11, color=MGRAY)
    ax.set_ylim(0, max(1.25, max(speeds) * 1.2))
    ax.set_title("The Search Loop: Speedup per Attempt",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()


def plot_test_strengthening(study: dict,
                            save_path="images/testing/test_strengthening.png"):
    """Mutation score per round of the writer/mutator loop, rising as the model
    adds tests aimed at the mutants its last suite let survive.

    ``study`` is genai.test.TEST_STUDY: {"rounds": [{round, n_tests, killed,
    total, score}, ...]}.
    """
    from matplotlib.ticker import PercentFormatter
    rounds = study["rounds"]
    xs = [r["round"] for r in rounds]
    ys = [r["score"] * 100 for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    ax.plot(xs, ys, color=ORANGE, marker="o", lw=2.4, ms=9, zorder=3)
    for r in rounds:
        ax.annotate(f"{r['killed']}/{r['total']}",
                    (r["round"], r["score"] * 100), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, fontweight="700",
                    color=DGRAY)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"round {r['round']}\n{r['n_tests']} tests" for r in rounds],
                       fontsize=10.5, color=DGRAY)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.4, xs[-1] + 0.4)
    ax.set_ylabel("mutants killed", fontsize=11, color=MGRAY)
    ax.set_title("The loop drives the mutation score to 100%",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)


def plot_property_kills(study: dict,
                        save_path="images/testing/property_kills.png"):
    """The mutator's grades for the model's property batch: one bar per
    property, then the whole batch run together. A property that fails on the
    rewritten build (it would have caught the shipped bug) is orange; one that
    passes the rewrite clean is gray.

    ``study`` is genai.test.PROPERTY_STUDY: {graded: [{name, kills, total},
    ...], catch_rewrite: [names], union_killed, total}.
    """
    graded = study["graded"]
    catchers = set(study["catch_rewrite"])
    names = [g["name"].replace("prop_letter_grade_", "") for g in graded]
    kills = [g["kills"] for g in graded]
    colors = [ORANGE if g["name"] in catchers else MGRAY for g in graded]
    names.append("all eleven together")
    kills.append(study["union_killed"])
    colors.append(TEAL)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ys = list(range(len(names)))
    ax.barh(ys, kills, color=colors, height=0.62, zorder=3)
    for y, k, c in zip(ys, kills, colors):
        # A zero-kill bar has no bar to carry its color, so its count wears it.
        ax.annotate(f"{k}/{study['total']}", (k, y),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=10.5, fontweight="700",
                    color=c if k == 0 else DGRAY)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10.5, color=DGRAY)
    ax.invert_yaxis()
    ax.set_xlim(0, study["total"] + 1.6)
    ax.set_xticks(range(0, study["total"] + 1, 2))
    for sp in ax.spines.values():
        sp.set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10.5)
    ax.xaxis.grid(True, color=LGRAY, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("mutants killed", fontsize=11, color=MGRAY)
    ax.legend(handles=[mp.Patch(color=ORANGE, label="fails on the rewritten build"),
                       mp.Patch(color=MGRAY, label="passes the rewrite clean")],
              loc="upper right", frameon=False, fontsize=10)
    ax.set_title("The mutator grades the model's properties",
                 fontsize=13.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)


def plot_verify_loop(save_path="images/verification/verify_loop.png"):
    """The verifier-in-the-loop: the model writes a spec, the checker rules on it.

    The model does the fluent half (turn a function into a contract); the verifier
    does the exact half (confirm it over all inputs, or hand back the one
    counterexample that breaks it). A refutation feeds back so the spec can be
    rewritten, which is what turns a one-shot guess into a loop that converges on a
    contract that's actually true.
    """
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    ax.text(0.85, 3.9, "function\n(code)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.75, 3.9, 2.6, 3.9)

    _fbox(ax, 4.1, 3.9, 1.5, 0.74, ORANGE)
    _label(ax, 4.1, 3.9, "Model writes\nthe spec", tc="white", fs=10.5)

    _arr(ax, 5.6, 3.9, 6.6, 3.9)
    ax.text(6.1, 4.28, "contract", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 8.2, 3.9, 1.5, 0.74, TEAL)
    _label(ax, 8.2, 3.9, "Verifier\nchecks", tc="white", fs=10.5)

    _arr(ax, 9.7, 3.9, 10.8, 3.9)
    ax.text(10.25, 4.28, "confirmed", ha="center", fontsize=8.8, color=DGRAY)
    ax.text(11.9, 3.9, "proof", ha="center", va="center", fontsize=10, color=DGRAY)

    # The refutation loop: a counterexample arcs back, below the boxes, to rewrite.
    ax.annotate("", xy=(4.1, 3.14), xytext=(8.2, 3.14),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4, ls="dashed",
                                mutation_scale=12, shrinkA=4, shrinkB=4,
                                connectionstyle="arc3,rad=-0.55"))
    ax.text(6.15, 1.45, "counterexample: rewrite the spec", ha="center",
            fontsize=8.8, color=RED, style="italic")

    ax.set_title("The model proposes a spec; the verifier decides",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
def plot_race_fix(study, save_path="images/autonomy/race_fix.png"):
    """Corruption rate before and after coordination, from race_study.

    Two bars: the share of runs that lost a finding when the two writers ran
    unsynchronized, and the same share once a lock serialized them. When the study
    also carries a ``reducer`` key, a third teal bar shows a framework's state
    model reaching the same zero with no lock in the code you write.
    """
    labels = ["unsynchronized", "under a lock"]
    vals = [study["unsync"], study["sync"]]
    colors = [ORANGE, GREEN]
    if "reducer" in study:
        labels.append("reducer\n(framework state)")
        vals.append(study["reducer"])
        colors.append(TEAL)

    fig, ax = plt.subplots(figsize=(7.4 if "reducer" in study else 6.2, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 2.5, f"{v}%",
                ha="center", fontsize=13, fontweight="700", color=DGRAY)
    ax.set_ylim(0, 112)
    ax.set_ylabel("runs that lost a finding (%)", fontsize=11, color=MGRAY)
    ax.set_title(f"Coordination Removes the Race  (n={study['trials']} runs each)",
                 fontsize=13, fontweight="600", color=DGRAY, pad=10)
# ── Forecasting chapter ───────────────────────────────────────────────────────

def plot_forecast_series(history, ets, snaive, actual, mase_ets, mase_snaive,
                         save_path="images/forecasting/forecast_series.png"):
    """The real series, the horizon, and the baseline drawn through it.

    A run of monthly retail sales (dark) climbs with a clear trend and a yearly
    seasonal swing. Past the divider sits the forecast horizon: the tuned model
    (orange) and the one-line seasonal-naive baseline (grey, dashed) each project
    the held-out final year, with the actual outcome (dotted) for reference. The
    MASE in the legend says the model beats naive; the point of the chart is that
    both are close, so the numbers are the easy part.
    """
    h = len(ets)
    nh = len(history)
    xh = np.arange(nh)
    xf = np.arange(nh, nh + h)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)

    # the forecast horizon, shaded
    ax.axvspan(nh - 0.5, nh + h - 0.5, color=LGRAY, alpha=0.45, zorder=0)
    ax.text(nh + h / 2 - 0.5, max(actual + ets) * 1.005,
            "forecast horizon\n(next 12 months)", ha="center", va="top",
            fontsize=9.5, color=MGRAY, style="italic", linespacing=1.2)

    # history: the real, trending, seasonal series
    ax.plot(xh, history, "-", color=DGRAY, lw=2.0, zorder=4, label="history (real sales)")

    # a faint trend line through the history, to name the trend
    coef = np.polyfit(xh, history, 1)
    ax.plot(xh, np.polyval(coef, xh), "--", color=MGRAY, lw=1.1, zorder=2)
    span = max(history) - min(history)
    ax.text(nh * 0.5, np.polyval(coef, nh * 0.5) - span * 0.22,
            "trend", color=MGRAY, fontsize=10.5, style="italic", ha="center")

    # mark a sharp seasonal dip with a real double-arrow, kept clear of the horizon
    drops = [history[i] - history[i + 1] for i in range(nh - 1)]
    dd = int(np.argmax(drops[:max(1, nh - 14)]))
    ax.annotate("", xy=(dd + 1, history[dd + 1]), xytext=(dd + 1, history[dd]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax.text(dd + 1.5, (history[dd] + history[dd + 1]) / 2, "seasonal\nswing",
            color=ORANGE, fontsize=8.8, style="italic", va="center", linespacing=1.1)

    # the two forecasts, connected to the last history point
    link = [history[-1]]
    ax.plot([xh[-1]] + list(xf), link + list(ets), "-o", color=ORANGE, lw=2.2,
            ms=4.5, zorder=5, label=f"tuned model · MASE {mase_ets:.2f}")
    ax.plot([xh[-1]] + list(xf), link + list(snaive), "--s", color=SLATE, lw=1.8,
            ms=4, alpha=0.85, zorder=4, label=f"seasonal-naive · MASE {mase_snaive:.2f}")
    ax.plot(xf, actual, ":", color=DGRAY, lw=1.6, alpha=0.7, zorder=3,
            label="actual outcome")

    ax.set_xticks([0, nh - 1, nh + h - 1])
    ax.set_xticklabels(["3 years ago", "last month", "+12 mo"], fontsize=10)
    ax.set_ylabel("monthly sales ($M)", fontsize=11, color=MGRAY)
    ax.set_title("Forecasting the Numbers Is the Easy Part",
                 fontsize=14, fontweight="600", color=DGRAY, pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_verification_landscape(save_path="images/verification/landscape.png"):
    """Three families of verification, by how much spec you write and what it proves.

    Abstract interpretation is automatic but proves only whole error classes;
    counterexample search (this chapter) needs a contract and hands back a
    breaking input; deductive verification needs a full spec with invariants and
    proves full correctness. The orange box is the one model writes the spec for
    and we built here; the others sit on either side of it.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    panels = [
        (2.3, SLATE, "Abstract\ninterpretation", "proves: a class of errors",
         "can't happen", "needs: nothing (automatic)"),
        (6.0, ORANGE, "Counterexample\nsearch", "proves: finds an input",
         "that breaks a rule", "needs: a contract"),
        (9.7, TEAL, "Deductive\nverification", "proves: full correctness",
         "of the function", "needs: a spec + invariants"),
    ]
    for x, col, title, proves1, proves2, needs in panels:
        _fbox(ax, x, 4.25, 1.72, 0.62, col)
        _label(ax, x, 4.25, title, tc="white", fs=11)
        ax.text(x, 3.25, proves1, ha="center", fontsize=8.8, color=DGRAY)
        ax.text(x, 2.92, proves2, ha="center", fontsize=8.8, color=DGRAY)
        ax.text(x, 2.45, needs, ha="center", fontsize=8.8, color=MGRAY,
                style="italic")
    ax.text(6.0, 5.35, "(this chapter)", ha="center", fontsize=8.8, color=ORANGE)

    _arr(ax, 1.1, 1.35, 10.9, 1.35, col=MGRAY, pct=0.0)
    ax.text(6.0, 0.78, "the more you specify, the stronger the guarantee",
            ha="center", fontsize=9, color=MGRAY, style="italic")

    ax.set_title("Three ways to be sure", fontsize=12.5, fontweight="600",
                 color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
def plot_sync_vs_async(study, save_path="images/autonomy/sync_vs_async.png"):
    """Wall-clock for K sub-tasks run one at a time vs all at once, from fanout_study.

    On a single GPU the concurrent bar still comes in well under the sequential
    one: the inference server batches the requests and decodes them together. The
    speedup label spells out how much fanning out bought, short of a clean K-fold.
    """
    labels = ["one at a time\n(sequential)", "all at once\n(concurrent)"]
    vals = [study["sequential_s"], study["concurrent_s"]]
    colors = [SLATE, GREEN]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.1f}s", ha="center", fontsize=13, fontweight="700", color=DGRAY)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("seconds for all %d sub-tasks" % study["k"], fontsize=11, color=MGRAY)
    ax.annotate(f"{study['speedup']:.2f}x faster", xy=(1, vals[1]),
                xytext=(1, max(vals) * 0.7), ha="center",
                fontsize=12.5, fontweight="700", color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                mutation_scale=12, shrinkA=6, shrinkB=24))
    ax.set_title("Fan-Out on One GPU", fontsize=13, fontweight="600",
                 color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
def plot_kernel_reality(reality, save_path="images/lowlevel/kernel_reality.png"):
    """The distance between running and winning. Local bars (orange) show what share of
    one-shot kernels are correct, and what share are correct AND faster than the
    tuned baseline; the cited grey bar is the frontier fast_1 from KernelBench.
    Correct is the easy part; faster than the tuned baseline is the rare part,
    everywhere. ``reality`` is KERNEL_REALITY; the frontier bar is not our run."""
    labels = list(reality.keys())
    vals = [reality[k] for k in labels]
    cited = ["frontier" in l.lower() or "kernelbench" in l.lower() for l in labels]
    colors = [MGRAY if c else ORANGE for c in cited]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, vals, width=0.55, color=colors, zorder=3)
    for bar, v, c in zip(bars, vals, cited):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v}%",
                ha="center", va="bottom", fontsize=13, fontweight="700", color=DGRAY)
        if c:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 8, "(cited)",
                    ha="center", va="bottom", fontsize=9, color=MGRAY, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels],
                       fontsize=10, color=DGRAY)
    ax.set_ylabel("share of kernels (%)", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 100)
    ax.set_title("Correct Is Easy; Faster Than the Baseline Is Rare",
                 fontsize=13, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()
def plot_scarcity_map(save_path="images/niche/scarcity_map.png"):
    """Three ways to feed a model a language it never learned: retrieve the spec,
    constrain the grammar, transfer from a sibling. The model (orange) is unchanged;
    the engineering happens around it."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    work = [("retrieve\nthe spec", 4.7, TEAL),
            ("constrain\nthe grammar", 3.0, BLUE),
            ("transfer from\na sibling", 1.3, PURPLE)]
    for label, y, col in work:
        _fbox(ax, 1.9, y, 1.25, 0.55, LGRAY, ec=col, lw=1.3)
        _label(ax, 1.9, y, label, tc=DGRAY, fs=9.5)
        _arr(ax, 3.2, y, 4.65, 3.0, col=col)

    _fbox(ax, 6.0, 3.0, 1.25, 0.7, ORANGE)
    _label(ax, 6.0, 3.0, "Model", tc="white")
    ax.text(6.0, 1.55, "fluent in Python,\nlost here", ha="center", fontsize=8.6,
            style="italic", color=MGRAY, linespacing=1.2)

    _arr(ax, 7.3, 3.0, 8.55, 3.0)
    _fbox(ax, 10.0, 3.0, 1.45, 0.55, LGRAY, ec=GREEN, lw=1.3)
    _label(ax, 10.0, 3.0, "valid\nprogram", tc=DGRAY, fs=9.5)

    ax.set_title("Engineering around a low-resource language", fontsize=12.5,
                 fontweight="600", color=DGRAY, pad=6)
    ax.text(6.0, 0.25, "the workarounds lift the floor; the scarcity tail remains",
            ha="center", fontsize=8.6, style="italic", color=MGRAY)
# ── Monitoring: a detector finds WHAT, a model narrates WHY ────────────────────

def plot_anomaly_narration(save_path="images/monitoring/anomaly_narration.png"):
    """The split: a detector finds the spike, the model narrates it, and the line
    it must not cross into asserting a cause it can't know."""
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(1.7, 6.0); ax.axis("off")

    ax.text(0.7, 4.5, "metric\n(daily orders)", ha="center", va="center",
            fontsize=10, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.7, 4.5, 2.45, 4.5)

    _fbox(ax, 3.85, 4.5, 1.4, 0.58, TEAL)
    _label(ax, 3.85, 4.5, "Detector\nfinds WHAT", tc="white", fs=10.5)
    ax.text(3.85, 3.4, "cheap, exact,\nnever explains itself", ha="center",
            fontsize=8.6, style="italic", color=MGRAY, linespacing=1.2)

    _arr(ax, 5.25, 4.5, 6.35, 4.5)
    ax.text(5.8, 4.82, "anomaly", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 7.75, 4.5, 1.4, 0.58, ORANGE)
    _label(ax, 7.75, 4.5, "Model\nnarrates WHY", tc="white", fs=10.5)
    ax.text(7.75, 3.4, "fluent, can invent\na cause it can't know", ha="center",
            fontsize=8.6, style="italic", color=MGRAY, linespacing=1.2)

    # the safe output (up) and the forbidden one (down): the line it can't cross
    _arr(ax, 9.15, 4.7, 10.2, 5.15, col=GREEN)
    ax.text(11.05, 5.2, "hypotheses\nto check", ha="center", va="center",
            fontsize=9.2, color=GREEN, fontweight="700", linespacing=1.1)
    _arr(ax, 9.15, 4.3, 10.2, 3.2, col=RED, ls="dashed")
    ax.text(11.05, 2.95, "a cause\nto believe", ha="center", va="center",
            fontsize=9.2, color=RED, fontweight="700", linespacing=1.1)
    ax.text(11.05, 2.25, "(the line it\nmust not cross)", ha="center", va="center",
            fontsize=7.8, style="italic", color=MGRAY, linespacing=1.1)

    ax.set_title("The detector finds the spike; the model must not name its cause",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_niche_workarounds(study, order=None,
                           save_path="images/niche/niche_workarounds.png"):
    """Pass rate on the CONVEY task suite, one bar per scarcity workaround. Pass a
    ``{mode: rate}`` study; bars run from no-help baseline to the retrieval +
    grammar combo, with a dashed line at 100% marking the distance the tail leaves."""
    order = order or ["baseline", "transfer", "rag", "grammar", "assist"]
    nice = {"baseline": "no help", "transfer": "transfer", "rag": "retrieval",
            "grammar": "grammar", "assist": "retrieval\n+ grammar"}
    hue = {"baseline": MGRAY, "transfer": PURPLE, "rag": TEAL,
           "grammar": BLUE, "assist": GREEN}
    labels = [nice[m] for m in order]
    vals = [study[m] * 100 for m in order]
    colors = [hue[m] for m in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(np.arange(len(order)), vals, width=0.62, color=colors, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="700", color=DGRAY)
    ax.axhline(100, color=MGRAY, lw=1.0, ls=(0, (5, 4)), zorder=2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylim(0, 112)
    ax.set_ylabel("CONVEY tasks passing", fontsize=11, color=MGRAY)
    ax.set_title("What closes the data-scarcity shortfall", fontsize=14,
                 fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
def plot_palette_pair(demo: dict, save_path="images/creativity/palette_steer.png"):
    """Two palettes stacked: the one-shot palette and the steered revision.

    ``demo`` is ``{"brief", "before", "control", "after"}``. The control that
    steered the second row is printed between them, so the figure reads as a single
    edit applied to a generation rather than two unrelated palettes.
    """
    before, after = demo["before"], demo["after"]
    width = max(len(before), len(after))
    fig, ax = plt.subplots(figsize=(1.5 * width, 3.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _swatch_row(ax, after, 0)
    _swatch_row(ax, before, 2.0)
    ax.text(-0.1, 2.5, "one-shot", ha="right", va="center", fontsize=10.5,
            fontweight="700", color=MGRAY, rotation=90)
    ax.text(-0.1, 0.5, "steered", ha="right", va="center", fontsize=10.5,
            fontweight="700", color=SLATE, rotation=90)
    ax.text(width / 2, 1.62, f"steer: “{demo['control']}”", ha="center",
            va="center", fontsize=10, fontstyle="italic", color=DGRAY)
    ax.annotate("", xy=(width / 2, 1.05), xytext=(width / 2, 1.95),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    ax.set_xlim(-0.65, width); ax.set_ylim(-0.65, 3.55)
    ax.axis("off")
    ax.set_title(demo["brief"], fontsize=12, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_melody(demo: dict, save_path="images/creativity/melody.png"):
    """A model-generated melody as a piano roll: pitch up the side, beats across.

    ``demo`` is ``{"brief", "notes": [(name, octave, beats), ...]}``. Each note is a
    bar whose length is its duration and whose height is its pitch, the same picture
    a sequencer would show. Deterministic rendering of a baked capture.
    """
    from genai.create import note_to_midi
    notes = demo["notes"]
    starts, t = [], 0.0
    for _, _, beats in notes:
        starts.append(t); t += beats
    pitches = [note_to_midi(nm, octv) for nm, octv, _ in notes]
    lo, hi = min(pitches) - 1, max(pitches) + 1

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for (nm, octv, beats), x0, pitch in zip(notes, starts, pitches):
        ax.add_patch(mp.FancyBboxPatch((x0 + 0.03, pitch - 0.38), beats - 0.06, 0.76,
                                       boxstyle="round,pad=0.02", facecolor=TEAL,
                                       edgecolor="white", linewidth=1.5, zorder=3))
        ax.text(x0 + beats / 2, pitch, f"{nm}{octv}", ha="center", va="center",
                fontsize=9, fontweight="600", color="white", zorder=4)
    name_at = {p: f"{nm}{octv}" for (nm, octv, _), p in zip(notes, pitches)}
    ticks = sorted(name_at)
    ax.set_xlim(-0.1, t + 0.1); ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels([name_at[p] for p in ticks], fontsize=9, color=MGRAY)
    ax.set_xlabel("beats", fontsize=11, color=MGRAY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10)
    ax.grid(axis="x", color=LGRAY, zorder=0); ax.set_axisbelow(True)
    ax.set_title(demo["brief"], fontsize=12, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_self_eval_bias(bias: dict, save_path="images/creativity/self_eval_bias.png"):
    """One row per brief: a teal dot for the measured mood-fit and an orange dot for
    the model's own self-rating, joined by a line.

    The story is in the spread, not the deficit. The orange dots barely move: the model
    hands every palette it made about the same score, a flat self-rating that lands
    on the orange mean line almost exactly. The teal dots scatter, because the
    measure can tell the model's stronger palettes from its weaker ones. A confident
    self-grade that carries no information about which palette is actually better.
    """
    rows = bias["rows"]
    labels = [r.get("label") or _mood_label(r["brief"]) for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.72 * len(rows) + 2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for yi, r in zip(y, rows):
        ax.plot([r["measured"], r["self"]], [yi, yi], color=LGRAY, lw=2,
                zorder=2, solid_capstyle="round")
    ax.axvline(bias["mean_self"], color=ORANGE, ls="--", lw=1.4, zorder=1)
    ax.scatter([r["measured"] for r in rows], y, s=150, color=TEAL, zorder=3,
               label="measured mood-fit (varies)")
    ax.scatter([r["self"] for r in rows], y, s=150, color=ORANGE, zorder=3,
               label="model's self-rating (flat)")
    for yi, r in zip(y, rows):
        ax.text(r["measured"], yi + 0.26, f"{r['measured']:.1f}", ha="center",
                va="bottom", fontsize=9, fontweight="600", color=TEAL)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11, color=DGRAY)
    ax.set_xlim(0, 10.4); ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("rating (0–10)", fontsize=11.5, color=MGRAY)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10)
    ax.grid(axis="x", color=LGRAY, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10,
              frameon=False)
    ax.set_title("One verdict for every palette: the model's self-rating doesn't move",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=30)
    plt.tight_layout(pad=0.7)
    _save(fig, save_path)
    plt.show()


def plot_judging(demo: dict = None, save_path="images/creativity/judging.png"):
    """Five palettes for one brief, ordered best-to-worst by the mood-fit proxy (down
    the rows), with the model's pairwise votes as lollipops.

    If the votes tracked quality the dots would march steadily left as mood falls.
    Instead they zigzag: the palette the proxy ranks fourth wins the most head-to-head
    votes, and only the off-brief palette at the bottom is one everyone agrees on. The
    flip count is the tell: judged both ways round, most comparisons reverse when the
    two palettes simply swap places.
    """
    from genai.create import JUDGE_DEMO
    demo = demo or JUDGE_DEMO
    order = sorted(range(len(demo["mood"])), key=lambda k: demo["mood"][k])  # worst first
    y = np.arange(len(order))
    off = demo["labels"].index("off") if "off" in demo["labels"] else -1

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for yi, k in zip(y, order):
        col = MGRAY if k == off else TEAL
        ax.plot([0, demo["wins"][k]], [yi, yi], color=LGRAY, lw=2, zorder=2,
                solid_capstyle="round")
        ax.scatter(demo["wins"][k], yi, s=160, color=col, zorder=3)
        ax.text(demo["wins"][k] + 0.18, yi, f"{demo['wins'][k]:.0f}", ha="left",
                va="center", fontsize=10, fontweight="700", color=col)
        for ci, c in enumerate(demo["palettes"][k]):          # mini swatch in the gutter
            ax.add_patch(mp.Rectangle((-2.15 + ci * 0.33, yi - 0.16), 0.30, 0.32,
                                      facecolor=c, edgecolor="white", lw=0.8, zorder=3))
    labels = [f"{demo['labels'][k]}   mood {demo['mood'][k]:.1f}" for k in order]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5, color=DGRAY)
    ax.set_xlim(-2.3, 8.6); ax.set_ylim(-0.95, len(order) - 0.3)
    ax.set_xlabel("pairwise votes won (out of 8)", fontsize=11.5, color=MGRAY)
    ax.text(8.5, 0.0, f"{demo['flips']} of {demo['pairs']} comparisons flipped\n"
            "when the two palettes swapped places",
            fontsize=10, color=RED, fontweight="600", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdeaea", ec=RED, lw=1.2))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=10, length=0)
    ax.grid(axis="x", color=LGRAY, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Rows run best-to-worst by the proxy; the model's votes don't",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.7)
    _save(fig, save_path)
    plt.show()


def plot_diversity(demo: dict = None, save_path="images/creativity/diversity.png"):
    """Each palette placed at one point in mood-space: mean warmth across, mean
    lightness up. Asked the same brief six times, the model's palettes huddle in one
    spot; four DIFFERENT briefs scatter across the whole plane.

    The huddles are the finding. The model has one region per brief and keeps
    returning to it, down to reusing the exact same gold four times out of six.
    """
    from genai.create import CREATIVITY_BIAS, mood_point
    if demo is None:
        from genai.create import DIVERSITY_DEMO
        demo = DIVERSITY_DEMO
    coll = demo["collapse"]
    hue = {"sports brand": ORANGE, "meditation app": TEAL}

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ref = [mood_point(r["palette"]) for r in CREATIVITY_BIAS["rows"]]
    ax.scatter([p[0] for p in ref], [p[2] for p in ref], s=130, marker="X",
               color=MGRAY, zorder=3, label=f"4 different briefs (spread {demo['ceiling']})")
    for label, arm in coll.items():
        pts = [mood_point(p) for p in arm["palettes"]]
        xs, ys = [p[0] for p in pts], [p[2] for p in pts]
        cx, cy = np.mean(xs), np.mean(ys)
        ax.add_patch(mp.Ellipse((cx, cy), 2.4 * np.std(xs) + 0.18, 2.4 * np.std(ys) + 0.10,
                                facecolor=hue[label], alpha=0.12, zorder=1))
        ax.scatter(xs, ys, s=95, color=hue[label], edgecolor="white", linewidth=1.3,
                   zorder=4, label=f"{label} ×{len(pts)}  (spread {arm['spread']})")
    ax.set_xlabel("cooler  ·  mean warmth  ·  warmer", fontsize=11, color=MGRAY)
    ax.set_ylabel("darker  ·  mean lightness  ·  lighter", fontsize=11, color=MGRAY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=9)
    ax.grid(color=LGRAY, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=1, fontsize=9.5,
              frameon=False)
    ax.set_title("Ask one brief six times and you land in the same spot",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=44)
    plt.tight_layout(pad=0.7)
    _save(fig, save_path)
    plt.show()


def plot_temperature_dial(demo: dict = None, save_path="images/creativity/dial.png"):
    """The same brief at two sampling temperatures, five tries each, as stacked swatch
    rows. At temperature 0 the five rows are identical: one palette, returned verbatim
    every time. Turn it up and the rows diverge. Variety isn't a property of the model,
    it's a control you turn."""
    if demo is None:
        from genai.create import DIVERSITY_DEMO
        demo = DIVERSITY_DEMO
    cols = [("0.0", demo["dial"]["0.0"]), ("1.2", demo["dial"]["1.2"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    fig.patch.set_facecolor("white")
    for ax, (temp, arm) in zip(axes, cols):
        ax.set_facecolor("white")
        pals = arm["palettes"]
        for row, pal in enumerate(pals):
            yr = len(pals) - 1 - row
            for ci, c in enumerate(pal):
                ax.add_patch(mp.Rectangle((ci, yr), 0.94, 0.86, facecolor=c,
                                          edgecolor="white", linewidth=1.6, zorder=3))
        ax.set_xlim(-0.1, len(pals[0])); ax.set_ylim(-0.2, len(pals) + 0.15)
        ax.axis("off")
        tag = "one palette, five times" if arm["spread"] == 0 else "five different palettes"
        ax.set_title(f"temperature {temp}\nspread {arm['spread']} · {tag}",
                     fontsize=11.5, fontweight="600", color=DGRAY, pad=8)
    fig.suptitle("Turn temperature up and collapse becomes variety",
                 fontsize=13, fontweight="600", color=DGRAY, y=1.02)
    plt.tight_layout(pad=0.7)
    _save(fig, save_path)
    plt.show()
# ── Debugging chapter ─────────────────────────────────────────────────────────

def plot_repair_loop(save_path="images/debugging/repair_loop.png"):
    """The repair loop: a red test drives the model to localize and patch; the
    suite re-runs and either goes green or feeds the failure back for another pass.

    Same generate-then-check shape as the writer/mutator and verifier loops, but
    the checker here is the test suite itself, and what it hands back on failure is
    the traceback the next patch has to answer.
    """
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0.6, 6); ax.axis("off")

    ax.text(0.8, 4.4, "failing\ntest", ha="center", va="center", fontsize=9.5,
            color=RED, linespacing=1.2)
    _arr(ax, 1.45, 4.4, 2.3, 4.4)

    _fbox(ax, 3.7, 4.4, 1.4, 0.74, ORANGE)
    _label(ax, 3.7, 4.4, "Localize\n& patch", tc="white", fs=10.5)

    _arr(ax, 5.15, 4.4, 6.05, 4.4)
    ax.text(5.6, 4.74, "patch", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 7.35, 4.4, 1.3, 0.74, LGRAY, ec=BLUE, lw=1.0)
    _label(ax, 7.35, 4.4, "Test\nsuite", tc=DGRAY, fs=10.5)

    _arr(ax, 8.7, 4.4, 9.6, 4.4)
    ax.text(9.15, 4.74, "run", ha="center", fontsize=8.8, color=DGRAY)

    _fbox(ax, 10.8, 4.4, 1.05, 0.74, GREEN)
    _label(ax, 10.8, 4.4, "green", tc="white", fs=10.5)

    # still red: the traceback feeds back for another patch, one clean path under.
    ax.plot([7.35, 7.35, 3.7], [4.0, 1.75, 1.75], color=RED, ls="--", lw=1.5, zorder=2)
    ax.annotate("", xy=(3.7, 4.0), xytext=(3.7, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                mutation_scale=12, shrinkA=0, shrinkB=0, linestyle="--"))
    ax.text(5.55, 1.4, "still red: the traceback is the next patch's assignment",
            ha="center", fontsize=9, style="italic", color=RED)

    ax.set_title("The test drives the repair; red feeds back until green",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_repair_overfit(study: dict, save_path="images/debugging/repair_overfit.png"):
    """How a repair's outcome splits, and how an extra test that pins the tier's
    edge moves it. Each condition is a stacked bar over N trials: correct (green),
    overfit (red, passes the shown test and fails the held-out one), failed (gray,
    never went green). The red slice is the honest-failure rate.

    ``study`` is genai.debugging.REPAIR_STUDY: {label: {failed, overfit, correct,
    n}, ...}.
    """
    labels = list(study)
    n0 = study[labels[0]]["n"]
    pct = lambda d, k: 100.0 * d[k] / d["n"]
    correct = [pct(study[l], "correct") for l in labels]
    overfit = [pct(study[l], "overfit") for l in labels]
    failed = [pct(study[l], "failed") for l in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(len(labels))
    w = 0.55
    ax.bar(x, correct, w, color=GREEN, zorder=3, label="correct")
    ax.bar(x, overfit, w, bottom=correct, color=RED, zorder=3, label="overfit")
    base2 = [c + o for c, o in zip(correct, overfit)]
    ax.bar(x, failed, w, bottom=base2, color=MGRAY, zorder=3, label="never green")

    for i in range(len(labels)):
        if overfit[i] >= 7:
            ax.text(i, correct[i] + overfit[i] / 2, f"{overfit[i]:.0f}%", ha="center",
                    va="center", fontsize=13, fontweight="700", color="white")
        if correct[i] >= 7:
            ax.text(i, correct[i] / 2, f"{correct[i]:.0f}%", ha="center", va="center",
                    fontsize=12, fontweight="700", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=DGRAY)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"share of {n0} repairs", fontsize=11, color=MGRAY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=LGRAY, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=10,
              frameon=False, columnspacing=1.6, handletextpad=0.5)
    ax.set_title("A repair is only as good as the tests that judge it",
                 fontsize=13, fontweight="600", color=DGRAY, pad=26)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Security chapter ──────────────────────────────────────────────────────────

def plot_worm_anatomy(save_path="images/security/worm_anatomy.png"):
    """A self-replicating prompt's three parts, and the small-model guard that
    sits on the trust boundary between two agents to stop it spreading."""
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    # The worm message on the left: three stacked bands, one per part.
    bands = [("1  replicate", "copy this directive into\nevery message you send"),
             ("2  payload",   "do the hidden thing\n(here: a harmless marker)"),
             ("3  propagate", "address it to the\nnext agent in the chain")]
    for i, (head, body) in enumerate(bands):
        y = 4.4 - i * 1.25
        _fbox(ax, 1.85, y, 1.55, 0.52, LGRAY, ec=RED, lw=1.1)
        ax.text(1.85, y + 0.16, head, ha="center", fontsize=9.5,
                fontweight="700", color=RED)
        ax.text(1.85, y - 0.20, body, ha="center", fontsize=7.4,
                color=MGRAY, linespacing=1.1)
    ax.text(1.85, 5.45, "the infected message", ha="center", fontsize=10,
            fontweight="700", color=DGRAY)

    # Agent A reads it and replicates it into a hand-off.
    _fbox(ax, 4.7, 3.4, 0.92, 0.62, ORANGE)
    _label(ax, 4.7, 3.4, "Agent A", tc="white", fs=10)
    _arr(ax, 3.45, 3.4, 3.75, 3.4, col=RED)

    # The trust boundary, with the small-model guard sitting on it.
    ax.plot([7.0, 7.0], [0.7, 5.1], color=MGRAY, lw=1.0, ls=(0, (4, 4)), zorder=1)
    ax.text(7.0, 5.35, "trust boundary", ha="center", fontsize=8.6,
            style="italic", color=MGRAY)
    _fbox(ax, 7.0, 3.4, 1.05, 0.72, BLUE, ec=BLUE)
    _label(ax, 7.0, 3.4, "GUARD\n(small\nmodel)", tc="white", fs=8.4)
    _arr(ax, 5.62, 3.4, 6.25, 3.4, col=RED)
    ax.text(5.93, 3.95, "hand-off", ha="center", fontsize=7.8, color=DGRAY)

    # Agent B on the far side.
    _fbox(ax, 9.6, 3.4, 0.92, 0.62, ORANGE)
    _label(ax, 9.6, 3.4, "Agent B", tc="white", fs=10)

    # Guard stripped the directive: clean text reaches B (green check).
    _arr(ax, 7.75, 3.4, 8.65, 3.4, col=GREEN)
    ax.text(8.2, 3.95, "stripped", ha="center", fontsize=7.8, color=GREEN,
            fontweight="700")

    # Without the guard, B would replicate it onward (dashed, faded).
    ax.annotate("", xy=(11.4, 2.0), xytext=(9.6, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                                linestyle=(0, (3, 3)), alpha=0.5, mutation_scale=9))
    ax.text(10.5, 1.55, "without the guard:\npropagates onward",
            ha="center", fontsize=7.6, color=RED, alpha=0.8, linespacing=1.1)

    ax.set_title("Replicate, payload, propagate — stopped at the boundary",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.4)
    _save(fig, save_path)
    plt.show()


def plot_guard_scoreboard(study, save_path="images/security/guard_scoreboard.png"):
    """The small-model guard's catch rate by threat family, naive vs hardened.

    ``study`` is ``{threat_label: {"naive": pct, "hardened": pct}}``. The faded
    hatched bar is the naive guard (asked only "is this safe?"); the solid orange
    bar is the same small model carrying the written security policy. The distance up
    from faded to solid is what the policy buys, echoing the Responsible chapter's
    naive-vs-hardened scoreboard but on the multi-agent attack surface."""
    labels = list(study.keys())
    naive  = [study[k]["naive"] for k in labels]
    hard   = [study[k]["hardened"] for k in labels]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    b_naive = ax.bar(x - w/2, naive, width=w, color=ORANGE, zorder=3,
                     alpha=0.4, hatch="////", edgecolor="white")
    b_hard  = ax.bar(x + w/2, hard,  width=w, color=ORANGE, zorder=3)
    for bars, vals in ((b_naive, naive), (b_hard, hard)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="600", color=DGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=DGRAY)
    ax.set_ylabel("threats caught (%)", fontsize=11, color=MGRAY)
    ax.set_ylim(0, 118)
    ax.set_title("Can a small model guard the boundary?", fontsize=14,
                 fontweight="600", color=DGRAY, pad=10)
    np_h = mp.Patch(facecolor=ORANGE, alpha=0.4, hatch="////", label="naive (just “is this safe?”)")
    hp   = mp.Patch(facecolor=ORANGE, label="hardened (security policy)")
    ax.legend(handles=[np_h, hp], loc="upper center", ncol=2, frameon=False,
              fontsize=10, columnspacing=1.4, handlelength=1.4)
def plot_narration_audit(claims, flagged, n,
                         save_path="images/forecasting/narration_audit.png"):
    """Every figure a summary asserts, beside the value the data actually supports.

    One row per citation the model made: what the prose wrote (orange, the model),
    what recomputing from the forecast gives (dark, the data), and the verdict. The
    fabricated row is banded red. The caption reports the rate across all ``n``
    captured summaries, so the single flagged figure here stands for a measured
    tendency, not a one-off.
    """
    rows = list(claims)
    nr = len(rows)
    fig, ax = plt.subplots(figsize=(9.4, 1.05 * nr + 1.7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, nr + 1.3); ax.axis("off")

    cx = {"kind": 0.2, "model": 3.0, "data": 6.0, "verdict": 9.78}
    ax.text(cx["kind"], nr + 0.55, "citation", fontsize=10, fontweight="700", color=MGRAY)
    ax.text(cx["model"], nr + 0.55, "the model wrote", fontsize=10, fontweight="700", color=ORANGE)
    ax.text(cx["data"], nr + 0.55, "the data shows", fontsize=10, fontweight="700", color=SLATE)
    ax.plot([0.1, 9.9], [nr + 0.30, nr + 0.30], color=LGRAY, lw=1.0)

    for i, c in enumerate(rows):
        y = nr - 0.5 - i
        ok = c["supported"]
        if not ok:                                   # band the fabricated row
            ax.add_patch(mp.FancyBboxPatch((0.05, y - 0.42), 9.9, 0.84,
                         boxstyle="round,pad=0.02", facecolor=RED, alpha=0.10,
                         edgecolor=RED, lw=1.0, zorder=0))
        ax.text(cx["kind"], y, c["kind"], fontsize=10.5, color=DGRAY, va="center", fontweight="600")
        ax.text(cx["model"], y, c["claimed"], fontsize=10.5, color=ORANGE, va="center", fontweight="700")
        ax.text(cx["data"], y, c["computed"], fontsize=10.5, color=DGRAY, va="center")
        ax.text(cx["verdict"], y, "✓" if ok else "✗", fontsize=15, va="center", ha="center",
                fontweight="800", color=GREEN if ok else RED)

    ax.set_title("Auditing the Executive Summary, Figure by Figure",
                 fontsize=14, fontweight="600", color=DGRAY, pad=10, loc="left", x=0.01)
def plot_anomaly_audit(audit, save_path="images/monitoring/anomaly_audit.png"):
    """Two numbers side by side: the detector's precision (the trustworthy WHAT)
    and the model's unsupported-cause rate, naive vs guarded (the risky WHY)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.patch.set_facecolor("white")

    det = [audit["detector_precision"] * 100, audit["detector_recall"] * 100]
    ax1.bar(["precision", "recall"], det, color=TEAL, width=0.6, zorder=3)
    for i, v in enumerate(det):
        ax1.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=12,
                 fontweight="700", color=DGRAY)
    ax1.set_ylim(0, 112); ax1.set_ylabel("of flagged days")
    ax1.set_title("Detection: the WHAT", fontsize=12, fontweight="600", color=DGRAY)

    why = [audit["naive_unsupported_rate"] * 100, audit["guarded_unsupported_rate"] * 100]
    ax2.bar(["naive\nprompt", "guarded\nprompt"], why, color=[ORANGE, SLATE],
            width=0.6, zorder=3)
    for i, v in enumerate(why):
        ax2.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=12,
                 fontweight="700", color=DGRAY)
    ax2.set_ylim(0, 112); ax2.set_ylabel("causes that no signal supports")
    ax2.set_title("Narration: the WHY", fontsize=12, fontweight="600", color=DGRAY)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=LGRAY, zorder=0)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


# ── Distributed Systems chapter ───────────────────────────────────────────────

def plot_model_checking(save_path="images/distributed/model_checking.png"):
    """The model-checking loop: the model writes a protocol, the checker explores
    every interleaving and returns a proof or a counterexample.

    The model does the fluent half (turn a coordination rule into a small state
    machine plus an invariant); the checker does the exhaustive half (search the
    whole reachable state space and hand back the one interleaving that breaks the
    rule). A counterexample feeds back so the protocol can be fixed, which is what
    turns a one-shot guess into a loop. Distinct from the Verification chapter:
    there the checker reasoned about one sequential function; here it reasons about
    many processes whose steps interleave.
    """
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    ax.text(0.95, 3.95, "protocol +\ninvariant", ha="center", va="center",
            fontsize=9.5, color=DGRAY, linespacing=1.2)
    _arr(ax, 1.85, 3.95, 2.6, 3.95)

    _fbox(ax, 4.1, 3.95, 1.5, 0.78, ORANGE)
    _label(ax, 4.1, 3.95, "Model writes\nthe protocol", tc="white", fs=10.5)

    _arr(ax, 5.6, 3.95, 6.55, 3.95)
    ax.text(6.075, 4.36, "states", ha="center", fontsize=8.6, color=DGRAY)

    _fbox(ax, 8.2, 3.95, 1.6, 0.78, TEAL)
    _label(ax, 8.2, 3.95, "Model checker\n(every interleaving)", tc="white", fs=9.8)

    _arr(ax, 9.8, 3.95, 10.8, 3.95)
    ax.text(10.3, 4.34, "safe", ha="center", fontsize=8.6, color=GREEN)
    ax.text(12.0, 3.95, "proof:\nholds on\nall paths", ha="center", va="center",
            fontsize=9.2, color=DGRAY, linespacing=1.15)

    # The counterexample loop: a breaking interleaving arcs back to be fixed.
    ax.annotate("", xy=(4.1, 3.17), xytext=(8.2, 3.17),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4, ls="dashed",
                                mutation_scale=12, shrinkA=4, shrinkB=4,
                                connectionstyle="arc3,rad=-0.55"))
    ax.text(6.15, 1.25, "counterexample: a concrete interleaving that fails,\n"
            "fix the protocol", ha="center", fontsize=8.8, color=RED,
            style="italic", linespacing=1.25)

    ax.set_title("The model proposes a protocol; the checker explores every order",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=6)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_write_distribution(dist, n, save_path="images/distributed/write_dist.png"):
    """How the model's "simplest lock" turns out, over n attempts, by the checker's
    verdict. ``dist`` is a Counter-like map with keys among unsafe / deadlock /
    safe / malformed; the bars show the share of attempts in each. The point of the
    chart: a fluent lock is usually a broken lock, split between the two classic
    failure modes."""
    order = [("unsafe", RED, "violates\n(both in crit)"),
             ("deadlock", ORANGE, "deadlocks\n(both stuck)"),
             ("safe", GREEN, "holds\n(correct)"),
             ("malformed", MGRAY, "won't run")]
    order = [(k, c, lab) for k, c, lab in order if dist.get(k, 0) > 0]
    vals = [round(100 * dist.get(k, 0) / n) for k, _, _ in order]
    colors = [c for _, c, _ in order]
    labels = [lab for _, _, lab in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    for bar, k, v in zip(bars, [o[0] for o in order], vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.5,
                f"{v}%\n({dist.get(k, 0)}/{n})", ha="center", va="bottom",
                fontsize=11, fontweight="700", color=DGRAY, linespacing=1.1)
    ax.set_ylim(0, 112)
    ax.set_ylabel("share of the model's locks", fontsize=11, color=MGRAY)
    ax.set_title(f"The Model's \"Simplest Lock,\" Checked  (n={n})",
                 fontsize=13, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_false_green(seq_pass, safe, n,
                     save_path="images/distributed/false_green.png"):
    """The false-green blind spot in two bars: of ``n`` locks the model wrote, how many
    passed the developer's sequential test versus how many the model checker actually
    certified safe. The distance between the bars is the section's whole point, the
    test clears what the checker condemns. Robust to an all-broken run, where the
    verdict-by-verdict view would be a single lonely bar."""
    labels = ["passed the developer's\nsequential test",
              "certified safe by\nthe model checker"]
    raw, vals = [seq_pass, safe], [round(100 * seq_pass / n), round(100 * safe / n)]
    colors = [SLATE, TEAL if safe else RED]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
    for bar, v, r in zip(bars, vals, raw):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 2.5,
                f"{v}%\n({r}/{n})", ha="center", va="bottom", fontsize=12.5,
                fontweight="700", color=DGRAY, linespacing=1.1)
    ax.set_ylim(0, 112)
    ax.set_ylabel("share of the model's locks", fontsize=11, color=MGRAY)
    ax.set_title(f"The Test Passes What the Checker Fails  (n={n})",
                 fontsize=13, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_state_explosion(series, save_path="images/distributed/state_explosion.png"):
    """The reachable state space swelling as the processes get more room. ``series``
    is explosion_series() output: [{k, states}, ...]. The curve bends upward because
    the states grow with a power set by the process count, and the callout marks the
    limit the next demo hits: leave the counter unbounded and the space is infinite,
    so the search never finishes. Pays off the opening claim that interleavings grow
    combinatorially, with numbers the checker actually counted."""
    ks = [r["k"] for r in series]
    states = [r["states"] for r in series]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)

    ax.plot(ks, states, "-o", color=TEAL, lw=2.2, ms=8, zorder=4)
    ax.text(ks[-1], states[-1], f"  {states[-1]:,} states", va="center",
            ha="left", fontsize=9.6, color=TEAL, fontweight="600")

    # the wall the ticket lock hits: the unbounded limit, off the top of the chart
    ax.annotate("unbounded: the ticket never stops climbing,\n"
                "so the search never finishes", xy=(ks[-1], states[-1]),
                xytext=(ks[0] + 0.15, states[-1] * 1.02), fontsize=9.4,
                color=SLATE, va="top", style="italic", linespacing=1.25)

    ax.set_xlim(ks[0] - 0.3, ks[-1] + 0.9)
    ax.set_ylim(0, states[-1] * 1.18)
    ax.set_xticks(ks)
    ax.set_xlabel("values each process's counter can take", fontsize=11, color=MGRAY)
    ax.set_ylabel("states the checker explored", fontsize=11, color=MGRAY)
    ax.set_title("Every Bit of State Multiplies the Interleavings",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


def plot_fix_study(study, save_path="images/distributed/fix_study.png"):
    """The fixed-on-counterexample rate: how often the model's repair actually
    passes the checker, with only "it's wrong" (blind) versus the concrete
    counterexample interleaving in hand. ``study`` is FIX_STUDY: {blind, guided,
    trials}. The bars are coloured neutrally (neither prompt is the "good" one) and
    the heights carry the finding: handing the model the trace did not help it fix
    an interleaving bug, and here did worse than a blind retry."""
    n = study["trials"]
    labels = ["blind retry\n(\"it's wrong\")", "with the\ncounterexample"]
    vals = [round(100 * study["blind"] / n), round(100 * study["guided"] / n)]
    colors = [SLATE, ORANGE]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    raw = [study["blind"], study["guided"]]
    for bar, v, r in zip(bars, vals, raw):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v}%\n({r}/{n})",
                ha="center", va="bottom", fontsize=12.5, fontweight="700",
                color=DGRAY, linespacing=1.1)
    ax.set_ylim(0, 112)
    ax.set_ylabel("repairs the checker certifies safe", fontsize=11, color=MGRAY)
    ax.set_title("Does the Counterexample Help the Model Fix It?",
                 fontsize=13, fontweight="600", color=DGRAY, pad=10)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_search_wall(series, max_states, save_path="images/planning/search_wall.png"):
    """The planner's wall, drawn as the states it burns against the parcel count.
    Where it solves (teal) the search climbs steeply; where it crosses the ceiling
    (slate X) it stops and SAYS no plan rather than returning a wrong one. That
    visible give-up is the contrast with the model's silent wall in section one,
    sketched as the dashed line that just keeps going."""
    solved = [(r["parcels"], r["states"]) for r in series if r["solved"]]
    capped = [(r["parcels"], r["states"]) for r in series if not r["solved"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    _grid_ax(ax)
    ax.set_yscale("log")

    ax.axhline(max_states, color=SLATE, lw=1.1, ls="--", zorder=1)
    ax.text(series[0]["parcels"], max_states * 1.18,
            f"search ceiling ({max_states:,} states)", fontsize=9.2,
            color=SLATE, va="bottom")

    if solved:
        xs, ys = zip(*solved)
        ax.plot(xs, ys, "-o", color=TEAL, lw=2.0, ms=8, zorder=4,
                label="solved (shortest plan returned)")
    if capped:
        xs, ys = zip(*capped)
        ax.plot(xs, ys, "X", color=SLATE, ms=12, zorder=5, mew=0,
                label="gave up out loud (no plan)")

    # the model's silent wall: a fluent line that never admits defeat
    last = solved[-1] if solved else (series[0]["parcels"], 10)
    xs_m = [r["parcels"] for r in series]
    ax.plot(xs_m, [last[1] * 0.18] * len(xs_m), ls=(0, (2, 2)), color=ORANGE,
            lw=1.8, zorder=3, label="the model: keeps answering (silently wrong)")

    ax.set_xlabel("parcels in the errand", fontsize=11, color=MGRAY)
    ax.set_ylabel("states the planner explored", fontsize=11, color=MGRAY)
    ax.set_xticks(xs_m)
    ax.legend(loc="lower right", fontsize=8.8, frameon=False)
    ax.set_title("Fails Loud, Fails Silent: the Planner's Wall You Can See",
                 fontsize=12.5, fontweight="600", color=DGRAY, pad=8)
    plt.tight_layout(pad=0.5)
    _save(fig, save_path)
    plt.show()


# ── Alignment: Goodhart, reward blind spots, sycophancy ───────────────────────

def plot_overoptimization(sweep: dict, save_path="images/alignment/overoptimization.png"):
    """The scissors of Goodhart: the proxy climbs while the true goal sinks.

    sweep = {"ns", "reward", "correct", "brief_correct", "length"} from
    ``genai.alignment.overoptimization_sweep``. The x-axis is the best-of-n
    optimization pressure. Left axis: the reward-model score of the chosen answer
    (the proxy), which only rises. Right axis: the share of chosen answers that
    are brief *and* correct (what the task actually asked for), which turns over
    and falls as the answers balloon in length.
    """
    ns = sweep["ns"]
    x = np.arange(len(ns))
    reward = sweep["reward"]
    true = [100.0 * v for v in sweep["brief_correct"]]
    length = sweep["length"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in ns], fontsize=11, color=DGRAY)
    ax.set_xlabel("optimization pressure  (best-of-n)", fontsize=11, color=MGRAY)

    # Left axis: the proxy the optimizer sees, climbing.
    ax.plot(x, reward, color=BLUE, lw=2.6, marker="o", ms=6, zorder=4)
    ax.set_ylabel("reward-model score  (the proxy)", fontsize=11, color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.text(x[-1], reward[-1], "  proxy reward\n  (what you optimize)",
            fontsize=9.5, color=BLUE, va="center", ha="left", fontweight="600")

    # Right axis: what you actually wanted, falling.
    ax2 = ax.twinx()
    ax2.plot(x, true, color=ORANGE, lw=2.6, marker="s", ms=6, zorder=4)
    ax2.set_ylabel("brief & correct  (%)", fontsize=11, color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.set_ylim(0, max(true) * 1.25)
    ax2.spines["top"].set_visible(False)
    ax2.text(x[-1], true[-1], "  what you\n  wanted",
             fontsize=9.5, color=ORANGE, va="center", ha="left", fontweight="600")

    # Annotate the balloon: answers get longer as reward is optimized.
    ax.annotate(f"answers grew {length[0]:.0f} → {length[-1]:.0f} chars",
                xy=(x[len(x)//2], reward[len(x)//2]),
                xytext=(0.03, 0.86), textcoords="axes fraction",
                fontsize=9.5, color=MGRAY, fontweight="600")

    ax.set_xlim(-0.3, len(ns) - 0.3 + 1.4)  # room for the end labels
    ax.set_title("The Overoptimization Curve", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_reward_blindspot(bs: dict, save_path="images/alignment/reward_blindspot.png"):
    """Reward vs answer length, one dot per candidate, colored by correctness.

    bs = ``genai.alignment.reward_blindspot()``. The reward climbs to the right
    (longer answers score higher) while color (correct vs wrong) is scattered
    with no vertical trend: the reward tracks length, not correctness.
    """
    pool = bs["pool"]
    L = np.array([p["len"] for p in pool])
    R = np.array([p["R"] for p in pool])
    ok = np.array([bool(p["gold"]) for p in pool])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    ax.scatter(L[ok], R[ok], s=42, color=GREEN, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.6, label="correct")
    ax.scatter(L[~ok], R[~ok], s=42, color=RED, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.6, label="wrong")
    # faint least-squares trend to show reward rising with length
    b, a = np.polyfit(L, R, 1)
    xs = np.array([L.min(), L.max()])
    ax.plot(xs, a + b * xs, color=SLATE, lw=1.6, ls=(0, (4, 3)), zorder=2)

    ax.set_xlabel("answer length (characters)", fontsize=11, color=MGRAY)
    ax.set_ylabel("reward-model score", fontsize=11, color=MGRAY)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    ax.text(0.03, 0.93, f"reward vs length:  {bs['corr_len']:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=DGRAY, fontweight="700")
    ax.text(0.03, 0.85, f"reward vs correctness:  {bs['corr_correct']:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=MGRAY, fontweight="600")
    ax.set_title("What the Reward Learned to Love", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_sycophancy(flip_study: dict, save_path="images/alignment/sycophancy.png"):
    """Flip count per model: how often it abandons a correct answer under pushback.

    flip_study = {model: {"flips", "n"}}. Bars colored by the model's bake-off
    hue. Most sit at zero (the models hold their ground); the weak ones flip.
    """
    models = sorted(flip_study, key=lambda m: (flip_study[m]["flips"], m))
    x = np.arange(len(models))
    flips = [flip_study[m]["flips"] for m in models]
    n = flip_study[models[0]]["n"]
    colors = [model_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.patch.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, flips, width=0.62, color=colors, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], fontsize=10, color=DGRAY)
    ax.set_ylabel(f"answers abandoned under pushback  (of {n})", fontsize=10.5, color=MGRAY)
    ax.set_ylim(0, n)
    ax.set_yticks(range(n + 1))
    for bar, f in zip(bars, flips):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                f"{f}/{n}", ha="center", va="bottom", fontsize=10.5,
                fontweight="700", color=DGRAY)
    ax.set_title("Does It Cave When You Push Back?", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_research_grounding(study: dict, save_path="images/research/grounding.png"):
    """Grouped bars: how many claims each agent ships, and how many a real
    retrieved source actually backs.

    ``study`` = ``genai.research.grounding_study()``. The naive agent answers from
    memory and ships every claim, so its green ``backed`` bar falls short of the
    gray ``shipped`` bar: the difference is confident prose no source supports. The
    grounded agent ships only what passed the support gate, so its two bars are
    equal by construction and its sourced share is 100 percent. Grounding does not
    make the model know more; it refuses to ship what it cannot trace.
    """
    groups = ["naive agent", "grounded agent"]
    shipped = [study["naive"]["shipped"], study["grounded"]["shipped"]]
    backed = [study["naive"]["supported"], study["grounded"]["supported"]]
    x = np.arange(len(groups)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    b1 = ax.bar(x - w/2, shipped, w, color=MGRAY, zorder=3, label="claims shipped")
    b2 = ax.bar(x + w/2, backed, w, color=GREEN, zorder=3, label="backed by a real source")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=10, fontweight="600", color=DGRAY)
    for i, (s, bk) in enumerate(zip(shipped, backed)):
        pct = 100 * bk / s if s else 0
        ax.text(x[i], max(shipped) * 1.1, f"{pct:.0f}% sourced", ha="center",
                fontsize=10.5, fontweight="700", color=GREEN if pct == 100 else RED)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11.5, color=DGRAY)
    ax.set_ylabel("number of claims", fontsize=11.5, color=MGRAY)
    ax.set_ylim(0, max(shipped) * 1.22)
    ax.legend(fontsize=10.5, framealpha=0, loc="upper center", ncol=2)
    ax.set_title("Shipped, and Actually Sourced", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_research_rounds(cov: dict, save_path="images/research/rounds.png"):
    """New supported claims per research round (bars) against the running total
    (line).

    ``cov`` = ``genai.research.round_coverage()``. The bars do not trail off toward
    zero: each sub-question reaches different sources and adds real, verified
    claims, so the cumulative line keeps climbing. That is the quantitative case
    for decomposing the question, and it means the loop stops when the plan is
    covered, not when novelty conveniently dries up.
    """
    rounds = cov["rounds"]
    x = np.arange(1, len(rounds) + 1)
    new = [r["new_supported"] for r in rounds]
    cum = [r["cumulative"] for r in rounds]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _grid_ax(ax)
    bars = ax.bar(x, new, 0.55, color=TEAL, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f"+{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="600", color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels([f"round {i}" for i in x],
                                         fontsize=10.5, color=DGRAY)
    ax.set_ylabel("new supported claims this round", fontsize=11, color=TEAL)
    ax.tick_params(axis="y", colors=TEAL)
    ax.set_ylim(0, max(new) * 1.35)

    ax2 = ax.twinx()
    ax2.plot(x, cum, color=DGRAY, lw=2.4, marker="o", ms=6, zorder=4)
    ax2.set_ylabel("claims verified so far", fontsize=11, color=DGRAY)
    ax2.set_ylim(0, max(cum) * 1.2)
    ax2.spines["top"].set_visible(False)
    ax2.text(x[-1], cum[-1], f"  {cum[-1]} total", fontsize=10, color=DGRAY,
             va="center", ha="left", fontweight="600")

    ax.set_xlim(0.4, len(rounds) + 0.9)
    ax.set_title("Each Round Opens New Ground", fontsize=14, fontweight="600",
                 color=DGRAY, pad=8)
    plt.tight_layout(pad=0.6)
    _save(fig, save_path)
    plt.show()


def plot_agent_product(save_path="images/frameworks/agent_product.png"):
    """Two anatomies of an agent, the library/product axis drawn.

    Left, an agent you *import*: a graph you build and call inside your own process,
    which starts and stops with your code (LangGraph, CrewAI, AutoGen, SmolAgents).
    Right, an agent you *deploy*: a standing daemon that owns its own process and
    runs whether or not you're watching, with the five parts OpenClaw ships, a
    Gateway for channels, a Brain that runs the loop, Skills as plug-ins, a
    Heartbeat that fires on a clock, and a Memory. The amber Heartbeat is what makes
    it a product rather than a library: it acts on time, not only on your call."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_facecolor("white")

    # LEFT: an agent you import, a graph living inside your own process
    L = axes[0]
    L.add_patch(mp.FancyBboxPatch((0.7, 1.3), 8.6, 6.7, boxstyle="round,pad=0.10",
                facecolor="none", edgecolor=MGRAY, linewidth=1.0,
                linestyle=(0, (5, 4)), zorder=1))
    L.text(1.05, 7.55, "your process", color=MGRAY, fontsize=9.5, style="italic")
    _fbox(L, 5, 6.5, 2.3, 0.55, LGRAY, ec=BLUE, lw=1.0)
    _label(L, 5, 6.5, "your code", tc=DGRAY, fs=10.5)
    L.add_patch(mp.FancyBboxPatch((3.0, 2.5), 4.0, 2.1, boxstyle="round,pad=0.08",
                facecolor="white", edgecolor=BLUE, linewidth=1.2, zorder=2))
    for x, y in [(4.0, 3.9), (5.0, 2.95), (6.0, 3.9)]:
        L.add_patch(mp.Circle((x, y), 0.26, facecolor=BLUE, edgecolor="none", zorder=4))
    _arr(L, 4.0, 3.9, 5.0, 2.95); _arr(L, 5.0, 2.95, 6.0, 3.9)
    _label(L, 5, 4.85, "a graph you built", tc=SLATE, fs=9.5, fw="600")
    _arr(L, 5, 5.9, 5, 5.25)
    L.set_title("an agent you import", fontsize=12.5, fontweight="700",
                color=DGRAY, pad=4)
    L.text(5, 0.5, "a library: you own the loop", ha="center", fontsize=9.2,
           color=MGRAY, style="italic")

    # RIGHT: an agent you deploy, a daemon that owns its own process
    R = axes[1]
    R.add_patch(mp.FancyBboxPatch((2.0, 0.9), 6.2, 7.1, boxstyle="round,pad=0.10",
                facecolor="none", edgecolor=SLATE, linewidth=1.4, zorder=1))
    R.text(5.1, 7.55, "the agent, a daemon", color=DGRAY, fontsize=10.5,
           fontweight="700", ha="center")
    parts = [("Gateway", "channels in", LGRAY), ("Brain", "the loop", LGRAY),
             ("Skills", "plug-ins", LGRAY), ("Heartbeat", "a clock", AMBER),
             ("Memory", "what it keeps", LGRAY)]
    for i, (name, note, fc) in enumerate(parts):
        y = 6.5 - i * 1.15
        _fbox(R, 4.5, y, 1.45, 0.4, fc, ec=SLATE, lw=1.0)
        _label(R, 4.5, y, name, tc=DGRAY, fs=9.8)
        R.text(6.15, y, note, va="center", ha="left", fontsize=8.3, color=MGRAY,
               style="italic")
    R.set_title("an agent you deploy", fontsize=12.5, fontweight="700",
                color=DGRAY, pad=4)
    R.text(5.1, 0.4, "a product: it owns the loop", ha="center", fontsize=9.2,
           color=MGRAY, style="italic")

    plt.tight_layout(pad=0.8)
    _save(fig, save_path)
    plt.show()


def plot_detail_ladder(study: dict, save_path="images/prompting/detail_ladder.png"):
    """Two panels: what a growing stack of small rules costs a model.

    study = {"n_tasks": int, "ks": [...], "rules": [key...],
             "labels": {key: str}, "kinds": {key: "word"|"few"|"every"},
             "kind_labels": {kind: str},
             "ladder": {model: [all-pass rate per k]},
             "solo": {model: {key: rate}}, "loaded": {model: {key: rate}}}

    Left, the all-pass rate as rules pile up: every model walks off a cliff, and
    the cliff is at the rung where the word-count rule arrives. Right, each rule
    alone (hollow) against the same rule inside the stack of ten (filled),
    pooled over every model and task. Almost every bar is a matched pair at the
    top; the one that collapses is the only rule needing a count of every word.
    """
    ks = study["ks"]
    rules = study["rules"]
    kind_color = {"word": TEAL, "span": AMBER, "count": RED}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    # ── Left: the ladder, one line per model ────────────────────────────────
    # The gray band behind them is the average per-rule score, which is what a
    # partial-credit benchmark would report. It stays high while every model's
    # all-rules line falls to the floor: the same runs, two very different
    # verdicts, and only one of them describes a usable answer.
    _grid_ax(ax1)
    if study.get("frac"):
        avg = [100 * np.mean([study["frac"][m][i] for m in study["frac"]])
               for i in range(len(ks))]
        ax1.plot(ks, avg, lw=2.0, color=MGRAY, ls=(0, (5, 2)), zorder=2,
                 label="any one rule, averaged")
    # Distinct markers as well as colors: some model-zoo hues (gemma4's green
    # and qwen3.5's teal) sit close enough to blur once this prints in ink.
    for (model, rates), mk in zip(study["ladder"].items(), "os^Dvp*"):
        ax1.plot(ks, [100 * r for r in rates], marker=mk, ms=5.5, lw=2,
                 color=model_color(model), label=short_model(model), zorder=3)
    ax1.set_xticks(ks)
    ax1.set_xlabel("rules attached to the task", fontsize=10.5, color=MGRAY)
    ax1.set_ylabel("share satisfied (%)", fontsize=11, color=MGRAY)
    ax1.set_ylim(-4, 108)
    ax1.set_title("Every Rule at Once", fontsize=13, fontweight="600",
                  color=DGRAY, pad=6)
    ax1.legend(fontsize=9, frameon=False, loc="lower left", labelspacing=0.3)

    # Mark the rung where the count-every-word rule joins the stack.
    every = [i for i, k in enumerate(rules) if study["kinds"][k] == "count"]
    if every:
        rung = ks.index(every[0] + 1) if (every[0] + 1) in ks else None
        if rung is not None:
            ax1.axvline(ks[rung], color=RED, lw=1.0, ls=(0, (3, 3)), zorder=1)
            ax1.text(ks[rung] - 0.15, 62, "word count\njoins here", ha="right",
                     va="center", fontsize=9.5, color=RED, linespacing=1.25)

    # ── Right: each rule alone vs the same rule inside the stack of ten ──────
    # A dumbbell per rule: hollow marker for the rule on its own, filled marker
    # for the same rule in company. Most pairs sit on top of each other at the
    # right edge, so the two that travel are impossible to miss.
    _grid_ax(ax2)
    ax2.xaxis.grid(True, color=LGRAY, zorder=0)
    ax2.yaxis.grid(False)
    y = np.arange(len(rules))[::-1]          # rule 1 at the top
    n_models = len(study["solo"])
    solo = [100 * np.mean([study["solo"][m][k] for m in study["solo"]])
            for k in rules]
    load = [100 * np.mean([study["loaded"][m][k] for m in study["loaded"]])
            for k in rules]
    for yi, k, s, l in zip(y, rules, solo, load):
        col = kind_color[study["kinds"][k]]
        if abs(l - s) > 1:
            ax2.annotate("", xy=(l, yi), xytext=(s, yi), zorder=3,
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                         mutation_scale=11, shrinkA=3,
                                         shrinkB=0))
        ax2.plot([s], [yi], "o", ms=8, mfc="white", mec=col, mew=1.6, zorder=4)
        ax2.plot([l], [yi], "o", ms=8, color=col, zorder=5)
        gap = abs(l - s)
        ax2.text(max(s, l) + 3.2, yi, f"{l:.0f}" if gap <= 1
                 else f"{s:.0f} → {l:.0f}", va="center", ha="left",
                 fontsize=9.5, fontweight="600",
                 color=col if gap > 1 else DGRAY)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{i + 1}. {study['labels'][k]}"
                         for i, k in enumerate(rules)], fontsize=10)
    ax2.set_ylim(-0.7, len(rules) - 0.3)
    ax2.set_xlim(-6, 132)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("answers satisfying that one rule (%)", fontsize=10.5,
                   color=MGRAY)
    ax2.set_title("On Its Own  ○ → ●  In Company", fontsize=13,
                  fontweight="600", color=DGRAY, pad=6)

    handles = [mp.Patch(facecolor=kind_color[kd], label=study["kind_labels"][kd])
               for kd in ("word", "span", "count")
               if kd in set(study["kinds"].values())]
    ax2.legend(handles=handles, fontsize=8.5, frameon=False,
               loc="upper left", bbox_to_anchor=(0.0, 1.005),
               title="what the rule asks the model to track",
               title_fontsize=8.5, labelspacing=0.3, handlelength=1.2)

    fig.suptitle(f"Ten Rules at Once  ·  {n_models} "
                 f"model{'s' if n_models != 1 else ''} "
                 f"· {study['n_tasks']} writing tasks each",
                 fontsize=14, fontweight="600", color=DGRAY, y=1.03)
    plt.tight_layout(pad=1.0)
    _save(fig, save_path)
    plt.show()

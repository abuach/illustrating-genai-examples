# Illustrating Generative AI: Examples

Companion code for the book *Illustrating Generative AI*. This repository is the
`genai` Python package that the book's notebooks import. Each chapter's cells stay
short (usually one import and one call) because the real work lives here.

```python
from genai import ask, embed, similarity, DocumentStore, run_agent, think

ask("What is a token?")
similarity(embed("cat"), embed("kitten"))
```

Everything runs locally. Language models are served by [Ollama](https://ollama.com),
and the embedding, speech, and image models run through Hugging Face libraries on your
own machine.

## Setup

### 1. Clone as `genai`

The modules import each other as `genai.*`, so the checkout folder must be named
`genai`, and you work from its parent directory:

```bash
git clone https://github.com/abuach/illustrating-genai-examples.git genai
```

### 2. Install Python packages

You need Python 3.13 or newer. From inside the `genai` folder:

```bash
pip install -r requirements.txt
```

This installs everything every chapter uses, including PyTorch and the agent
frameworks, so it's a large download. Only the core group at the top of
`requirements.txt` is needed for `import genai`. Every other package is imported
only by the chapter that uses it.

### 3. Pull the models

Start Ollama (it's expected at `http://localhost:11434`) and pull the defaults:

```bash
ollama pull gemma4:latest
```

```bash
ollama pull nomic-embed-text
```

```bash
ollama pull qwen2.5-coder:latest
```

```bash
ollama pull gemma3:latest
```

Some chapters compare a larger zoo of models, such as `gpt-oss:20b`, `qwen3`, `llama3.1`,
`mistral`, and `gemma4:e2b`. Pull those when a demo asks for them.

### Optional extras

- **Stockfish:** the chess demo's engine row needs a `stockfish` binary on your `PATH`
  (`brew install stockfish`). Move parsing and rule checks work without it.
- **FLUX.1-schnell:** a gated Hugging Face model, so you have to log in first. SD-Turbo
  and SDXL-Turbo don't need a login.
- **Icarus Verilog:** the hardware half of the low-level code chapter simulates
  with `iverilog` (`brew install icarus-verilog`). Its results are also baked in.

## Conventions

- **Short answers by default.** `ask` and `chat` add a brief system prompt and cap
  output at 200 tokens so notebook output stays readable. Override per call:
  `ask("Write a 500-word essay...", max_tokens=800, system=None)`.
- **Thinking models.** Pass `think=False` to turn reasoning off, or `think=True` with
  a `thinking_budget` to reserve tokens for it.
- **Frozen results.** Many chapters compare models or run studies that are slow or
  nondeterministic. Their results are saved as constants or JSON snapshots, so the
  book's figures reproduce exactly. You can rerun them live.

## What's inside

**Foundations**

| Module | Topic |
| --- | --- |
| `llm` | `ask`, `chat`, and next-token distributions over Ollama |
| `tokens` | Real tokenizers side by side (GPT-2, GPT-4, GPT-4o, StarCoder2), BPE from scratch |
| `embed`, `code` | Text and code embeddings, similarity, semantic search, analogies |
| `vision`, `imagegen`, `audio` | Image understanding, local diffusion, Whisper speech-to-text |
| `thinking` | Reasoning models and thinking budgets |
| `perf` | Latency and tokens-per-second measurement |
| `viz` | Plotting helpers used throughout the book |
| `intro`, `chess`, `snake` | The opening demonstrations |

**Prompting and augmentation**

| Module | Topic |
| --- | --- |
| `prompting` | Step-back prompting |
| `pal` | Program-aided language models: the model writes code, Python computes |
| `rag` | Chunking, an in-memory `DocumentStore`, grounded answers |
| `cove` | Chain-of-Verification |
| `crag` | Corrective RAG |
| `book` | The book itself as a corpus in a Chroma vector database |
| `mcp` | A Model Context Protocol server over the book, and a model that uses it |

**Agents**

| Module | Topic |
| --- | --- |
| `agent` | Sophia, the book's agent: tools, recovery, memory, reflection, routing, safety gates |
| `arch` | Who owns control flow, the model or the code |
| `orchestrate`, `crew`, `groupchat`, `codeact` | Sophia rebuilt in LangGraph, CrewAI, AutoGen, and SmolAgents |
| `coord`, `distributed` | Races between concurrent agents, and model checking every interleaving |

**Correctness**

| Module | Topic |
| --- | --- |
| `test` | Model-written tests, graded by mutation testing |
| `verify` | Contracts checked symbolically with CrossHair |
| `reason` | Puzzles translated to Z3 constraints |
| `plan` | LLM-Modulo: the model states the problem, a classical planner solves it |
| `debugging` | Localize, patch, and re-run until the suite is green |
| `refactoring` | Behavior-preserving changes checked against the old version |
| `lowlevel` | Model-optimized numerical kernels: correct *and* faster? |
| `niche` | Working in a language the model has never seen |

**Applications**

| Module | Topic |
| --- | --- |
| `datasql` | Text-to-SQL and silently wrong queries |
| `forecast` | Forecasting with a model narrating the numbers |
| `monitoring` | A detector finds the anomaly, a model suggests hypotheses |
| `research` | Research reports where every claim cites a retrieved source |
| `create` | Creative work with no ground truth |

**Responsible AI**

| Module | Topic |
| --- | --- |
| `security` | Prompt injection, jailbreaks, leakage, redaction, moderation, watermarking |
| `privacy` | Differential privacy and federated averaging |
| `alignment` | Preference pairs, a small reward model, and Goodhart-style overoptimization |

## The book index

`book_index/` is a saved Chroma database of the book's passages. `BookIndex` in
`book.py` opens it, and it's used for search and question answering over the book:

```python
from genai.book import BookIndex

idx = BookIndex().build()   # embeds only passages not already stored
idx.search("byte pair encoding", k=3)
idx.ask("Why does RAG reduce hallucination?")
```

`build()` reads the passages from `book_corpus.json`.

## License

MIT. See [LICENSE](LICENSE).

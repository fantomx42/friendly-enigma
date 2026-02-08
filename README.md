# Ralph AI

An autonomous agent that pairs any local LLM (via [Ollama](https://ollama.com)) with **Wheeler Memory** — a cellular automata system that stores patterns as stable attractors instead of text.

```
Task → Wheeler Memory (encode → dynamics → attractor → recall)
           ↕ weighted context
        Your LLM (via Ollama)
           ↓
     Result feeds back into Wheeler Memory
```

The LLM is an interface layer. Wheeler Memory is the intelligence.

## Quick Start

### 1. Prerequisites

- **Linux** (tested on Arch/CachyOS, should work on Ubuntu/Fedora)
- **Python 3.10+**
- **[Ollama](https://ollama.com)** running locally
- Any model you want — Ralph works with whatever you pull into Ollama

### 2. Install

```bash
git clone https://github.com/fantomx42/friendly-enigma.git ralph
cd ralph/ai_tech_stack

# Create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip install numpy requests
```

### 3. Pull a model

Use whatever Ollama model fits your hardware:

```bash
# Small (4GB VRAM)
ollama pull qwen2.5:3b

# Medium (8GB VRAM) — default
ollama pull qwen3:8b

# Large (16GB+ VRAM)
ollama pull qwen2.5-coder:14b
```

### 4. Run

```bash
cd ai_tech_stack
./ralph_loop.sh "Your objective here"
```

Override the model or context window:

```bash
RALPH_MODEL=qwen2.5-coder:14b ./ralph_loop.sh "Your objective"
RALPH_NUM_CTX=16384 ./ralph_loop.sh "Your objective"   # smaller context to save VRAM
```

Ralph iterates until the LLM outputs `<promise>COMPLETE</promise>` or hits max iterations.

## Wheeler Memory

This is not RAG. There's no "save text, search later" step.

1. Text encodes to a 128x128 spatial frame (each character stamps an 8x8 pattern)
2. Cellular automata dynamics run until the frame settles into a stable attractor
3. The attractor IS the memory — the dynamics created it
4. Recall compares attractor patterns, not text strings
5. Similar meanings collapse to similar attractors, even with different words

### Stability Scoring

Memories get weighted by how well they survive pressure:

| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency (sigmoid normalized) |
| frame_persistence | 30% | Re-encoding produces the same frame |
| compression_survival | 30% | Pattern survives 10 ticks of dynamics |

Higher stability = more context budget when building prompts.

## Void Editor Extension

The `ralph-vscode/` directory contains a VS Code / [Void Editor](https://voideditor.com) extension with:

- Sidebar: run state, iteration progress, Wheeler Memory panel
- Live Wheeler Memory viewer: heatmap patterns + full scrollable text
- Instant refresh when new memories are stored
- llama-server management (optional)

```bash
cd ralph-vscode
npm install && npm run compile
```

Open Void/VS Code with the extension in dev mode:

```bash
code --extensionDevelopmentPath=./ralph-vscode ~/ralph
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RALPH_MODEL` | Ollama model name | `qwen3:8b` |
| `RALPH_NUM_CTX` | Context window (tokens) | `32768` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

### GPU Setup

Ralph uses Ollama for inference — configure your GPU through Ollama's setup:

- **AMD (ROCm):** Set `HSA_OVERRIDE_GFX_VERSION` for your GPU in your shell profile
- **NVIDIA (CUDA):** Works out of the box with Ollama
- **CPU only:** Works fine, just slower

## Directory Structure

| Directory | What it is |
|-----------|-----------|
| `ai_tech_stack/` | Core loop, Ollama client, security, tools |
| `wheeler_ai_training/` | Wheeler Memory — text-to-grid encoder, dynamics |
| `ralph-vscode/` | Void/VS Code editor extension |
| `conductor/` | TDD project management framework |
| `docs/` | SCM axioms and architecture docs |

## How It Works

1. Task arrives (human input)
2. Wheeler Memory encodes text → runs dynamics → finds attractor
3. Stability scoring weights retrieved patterns
4. Weighted context + task → your LLM via Ollama
5. Model generates response
6. Result feeds back into Wheeler Memory
7. Patterns strengthen or fade based on outcomes
8. Loop repeats until `<promise>COMPLETE</promise>` or max iterations

## Development

```bash
cd ai_tech_stack
source venv/bin/activate

# Run tests
PYTHONPATH=. pytest ralph_core/tests/ -v

# Lint
flake8 ralph_core/ --max-line-length=127
```

## License

MIT

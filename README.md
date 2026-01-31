# Ralph AI

<div align="center">

**Autonomous AI Swarm System with Local LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-green.svg)](https://ollama.ai)

*"Me fail English? That's unpossible!"* — The Ralph Wiggum Method

</div>

---

## What is Ralph AI?

Ralph AI is a **hierarchical multi-agent system** that accomplishes tasks through iterative refinement. It runs entirely on local LLMs via [Ollama](https://ollama.ai), requiring no API keys or cloud services.

The system embraces failure as diagnostic information, iterating until success is achieved through the `<promise>COMPLETE</promise>` signal.

## Architecture

```
Human Input
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  TRANSLATOR (phi3:mini)                             │
│  Natural language → Structured TaskSpec             │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  ORCHESTRATOR (deepseek-r1:14b)                     │
│  Strategic planning & task decomposition            │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  MIDDLE MANAGEMENT                                  │
│                                                     │
│  ┌──────────────┐    ◄────►    ┌──────────────┐    │
│  │   ENGINEER   │              │   DESIGNER   │    │
│  │ qwen2.5-coder│              │ mistral-nemo │    │
│  │   14b        │              │   12b        │    │
│  └──────────────┘              └──────────────┘    │
│       Code                        Review           │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  ASICs (Ultra-Small Specialists)                    │
│  regex │ json │ sql │ test │ fix │ doc             │
│  tinyllama:1.1b, deepseek-coder:1.3b, qwen:1.5b    │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Required Models

```bash
# Core agents
ollama pull phi3:mini
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:14b
ollama pull mistral-nemo:12b

# ASICs
ollama pull tinyllama:1.1b
ollama pull deepseek-coder:1.3b
ollama pull qwen2.5-coder:1.5b

# Embeddings
ollama pull nomic-embed-text
```

### 3. Clone & Run

```bash
git clone https://github.com/fantomx42/friendly-enigma.git ralph-ai
cd ralph-ai
pip install -r requirements.txt

./ralph_loop.sh "Create a Python function that validates email addresses"
```

## Usage

```bash
# Basic execution
./ralph_loop.sh "Your objective"

# V2 message-driven pipeline
./ralph_loop.sh --v2 "Your objective"

# Sandboxed execution (Docker)
./ralph_loop.sh --sandbox "Your objective"

# Background daemon with REM sleep consolidation
python ralph_daemon.py

# Voice interface
python ralph_voice.py

# Web UI
./start_ui.sh
```

## Features

| Feature | Description |
|---------|-------------|
| **Autonomous Iteration** | Loops until `<promise>COMPLETE</promise>` (max 10 iterations) |
| **Message Bus (V2)** | Protocol-based agent communication with circuit breakers |
| **Vector Memory** | Semantic search via ChromaDB + nomic-embed-text |
| **Security Checkpoint** | All outputs validated before execution |
| **REM Sleep** | Background memory consolidation from lessons learned |
| **Docker Sandbox** | Isolated execution for untrusted operations |
| **Voice Input** | Speech-to-text interface |
| **Web Dashboard** | Real-time monitoring UI |

## Project Structure

```
ralph_core/
├── agents/          # Translator, Orchestrator, Engineer, Designer, etc.
├── asic/            # Ultra-small specialist spawning
├── protocols/       # Message bus & schemas
├── security/        # Guards, dogs, checkpoint
├── memory.py        # Vector DB integration
├── runner.py        # Main execution pipeline
└── swarm.py         # Agent exports

ralph_ui/            # FastAPI + vanilla JS dashboard
ralph_daemon.py      # Background service with REM sleep
ralph_loop.sh        # Main execution loop
```

## Requirements

- **RAM**: 32GB minimum (64GB recommended)
- **GPU**: 16GB+ VRAM for full model loading
- **OS**: Linux (tested on CachyOS/Arch)
- **Storage**: ~35GB for all models

## How It Works

1. **Translate** — Human input becomes structured TaskSpec
2. **Plan** — Orchestrator decomposes into subtasks
3. **Execute** — Engineer generates solution
4. **Review** — Designer validates output
5. **Iterate** — Failures feed back as diagnostic info
6. **Complete** — Exit on `<promise>COMPLETE</promise>`

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

Built with [Ollama](https://ollama.ai) • [ChromaDB](https://trychroma.com) • [FastAPI](https://fastapi.tiangolo.com)

</div>

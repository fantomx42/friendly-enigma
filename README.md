# Ralph AI

Autonomous AI agent system with single-model architecture and Wheeler Memory integration.

## Overview

Ralph AI accomplishes tasks through iterative refinement using a single 80B MoE model (`qwen3-coder-next`) with stability-weighted context from Wheeler spatial memory.

```
Human Input → [qwen3-coder-next] → Wheeler Memory ↔ Stability Scoring → Output
               All cognitive roles    (meaning survives pressure)
```

**Key Features:**
- Single-model architecture (no agent switching overhead)
- Wheeler Memory: 2D spatial dynamics for associative recall
- SCM-based stability scoring: patterns weighted by survival under compression
- Iterates until `<promise>COMPLETE</promise>` detected

## Quick Start

```bash
cd ai_tech_stack

# Run Ralph with an objective
./ralph_loop.sh "Your objective here"

# Example
./ralph_loop.sh "Write a Python function to check if a number is prime"
```

## Requirements

- **Ollama** with `qwen3-coder-next` model (51GB)
- **Python 3.10+** with venv
- **RAM:** 44GB+ recommended (or accept slow inference with RAM offload)
- **GPU:** AMD ROCm or NVIDIA CUDA via Ollama

### Install Models

```bash
# Update Ollama to pre-release for qwen3-coder-next support
curl -L https://github.com/ollama/ollama/releases/download/v0.15.5-rc4/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
sudo tar --zstd -C /usr -xf /tmp/ollama.tar.zst
sudo systemctl restart ollama

# Pull the model
ollama pull qwen3-coder-next
ollama pull nomic-embed-text
```

## Architecture

### Symbolic Collapse Model (SCM)

Ralph AI is built on the [SCM Axioms](docs/SCM_AXIOMS.md):

1. Meaning is what survives symbolic pressure
2. Meaning is the remainder after compression
3. If a symbol collapses under pressure, it never had real meaning

### Wheeler Memory

Spatial dynamics system that encodes text into 128x128 2D grids and runs cellular-automata-like evolution. Memories are scored by stability:

| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency |
| frame_persistence | 30% | Re-encode matches stored frame |
| compression_survival | 30% | Pattern survives 10-tick dynamics |

Higher stability = more context budget in prompts.

### Components

| Directory | Purpose |
|-----------|---------|
| `ai_tech_stack/` | Core Ralph system |
| `ai_tech_stack/ralph_simple.py` | Single-model loop |
| `ai_tech_stack/ralph_core/` | Agents, protocols, security |
| `ai_tech_stack/ralph_gui/` | Native desktop GUI (Rust/egui) |
| `ai_tech_stack/ralph_ui/` | Web dashboard (FastAPI) |
| `wheeler_ai_training/` | Wheeler Memory neural network |
| `docs/` | SCM axioms, documentation |

## Configuration

Override the model:
```bash
RALPH_MODEL=qwen2.5-coder:14b ./ralph_loop.sh "objective"
```

## Development

```bash
cd ai_tech_stack
source venv/bin/activate
PYTHONPATH=. pytest ralph_core/tests/ -v
```

## License

MIT

## Hardware Tested

- Intel Core Ultra 7 265K
- AMD Radeon RX 9070 XT (16GB VRAM)
- 32GB DDR5 RAM
- CachyOS Linux

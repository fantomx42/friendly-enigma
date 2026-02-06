# Ralph AI

Ralph is a Wheeler Memory system. The memory is the intelligence — not the LLM.

Something comes in. The system changes because of it. That change IS the memory. There is no separate read/write cycle. Every query creates a new attractor in the dynamics. The system learns by existing.

The LLM (currently `qwen3-coder-next`) is a swappable interface layer — it translates between human language and Wheeler Memory. It is not Ralph's brain.

## How It Works

```
  input text
      │
      ▼
 ┌──────────┐     ┌──────────────────────────────┐
 │   LLM    │◄───►│       WHEELER MEMORY          │
 │ (module)  │     │                              │
 └──────────┘     │  encode → dynamics → attractor │
                  │  recall → blend → respond      │
                  │  consolidate → dream → learn    │
                  └──────────────────────────────┘
```

1. Text encodes into a 128x128 spatial frame
2. Cellular automata dynamics find a stable attractor
3. The attractor is compared against stored patterns (recall)
4. Patterns blend, and the response emerges
5. Background processes consolidate and dream

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown.

## Quick Start

```bash
cd ai_tech_stack
./ralph_loop.sh "Your objective here"
```

Override the model:
```bash
RALPH_MODEL=qwen2.5-coder:14b ./ralph_loop.sh "objective"
```

Run Wheeler Memory standalone:
```bash
cd wheeler_ai_training
python wheeler_ai.py
```

## Requirements

- **Ollama** with `qwen3-coder-next` (51GB) or another model
- **Python 3.10+**
- **RAM:** 44GB+ recommended (or accept slower inference with RAM offload)
- **GPU:** AMD ROCm or NVIDIA CUDA via Ollama

### Install Models

```bash
ollama pull qwen3-coder-next
ollama pull nomic-embed-text
```

## Repository Structure

| Directory | What it is |
|-----------|-----------|
| `wheeler_ai_training/` | Wheeler Memory — the core of Ralph |
| `ai_tech_stack/` | Runtime: loop script, security, tools, GUI |
| `conductor/` | TDD project management framework |
| `docs/` | SCM axioms and documentation |

## Symbolic Collapse Model

Ralph is built on the [SCM Axioms](docs/SCM_AXIOMS.md):

1. Meaning is what survives symbolic pressure
2. Meaning is the remainder after compression
3. If a symbol collapses under pressure, it never had real meaning

## Development

```bash
cd ai_tech_stack
source venv/bin/activate
PYTHONPATH=. pytest ralph_core/tests/ -v
```

## Hardware Tested

- Intel Core Ultra 7 265K
- AMD Radeon RX 9070 XT (16GB VRAM)
- 32GB DDR5 RAM
- CachyOS Linux

## License

MIT

# Wheeler Memory

A standalone, biological-grade associative memory system for AI consciousness. Wheeler Memory decouples "memory" from "model," allowing any intelligence to access a persistent, evolving, stability-weighted history of experiences.

```
Input → Wheeler Memory (encode → dynamics → attractor → recall)
           ↓
    Stability-Weighted Context
           ↓
    Downstream AI / Agent
```

## Quick Start

### 1. Prerequisites

- **Linux** (tested on Arch/CachyOS)
- **Python 3.10+**
- **PyTorch**

### 2. Install

```bash
git clone https://github.com/fantomx42/friendly-enigma.git wheeler
cd wheeler
pip install .
```

### 3. Basic Usage

```python
import asyncio
from wheeler.core.memory import WheelerMemory

async def main():
    wm = WheelerMemory("./memory_store")
    await wm.initialize()
    
    # Store a memory
    await wm.store("The SCM Axioms state that meaning is what survives pressure.")
    
    # Recall based on pattern similarity
    results = await wm.recall("What is meaning?")
    for res in results:
        print(f"Found: {res['key']} (Stability: {res['stability']:.2f})")

asyncio.run(main())
```

## How It Works

This is not RAG. There's no "save text, search later" step.

1. **Spatial Encoding**: Text encodes to a 128x128 spatial frame (each character stamps an 8x8 pattern).
2. **Cellular Automata**: Dynamics run until the frame settles into a stable attractor.
3. **Attractor Representation**: The attractor IS the memory — the dynamics created it.
4. **Physical Recall**: Recall compares attractor patterns, not text strings. Similar meanings collapse to similar attractors.

### Stability Scoring

Memories get weighted by how well they survive pressure:

| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency (sigmoid normalized) |
| frame_persistence | 30% | Re-encoding produces the same frame |
| compression_survival | 30% | Pattern survives 10 ticks of dynamics |

## Directory Structure

| Directory | What it is |
|-----------|-----------|
| `wheeler/` | Core library: codec, dynamics engine, storage |
| `twai/` | Next-gen Rust/Leptos frontend + SurrealDB backend |
| `tests/` | Unit and integration tests |
| `scripts/` | Verification and ingestion scripts |
| `docs/` | Architectural axioms and specifications |
| `conductor/` | Development tracks and planning |
| `legacy_archive/` | Ralph AI integration and legacy scripts |

## Ralph AI Integration

Wheeler Memory was originally developed as the "Hippocampus" for **Ralph AI**. The Ralph integration scripts, VS Code extension, and loop logic are available in `legacy_archive/`.

## TWAI

TWAI is the next-generation interface for Wheeler Memory, built with Rust. It provides a Leptos/WASM frontend and an Actix-web backend backed by SurrealDB. See `twai/` for details.

## License

MIT

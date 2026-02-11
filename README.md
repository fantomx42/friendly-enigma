# Wheeler Memory

A standalone, biological-grade associative memory system for AI consciousness. Wheeler Memory decouples "memory" from "model," allowing any intelligence to access a persistent, evolving, stability-weighted history of experiences.

Built on the **Symbolic Collapse Model (SCM)** — meaning is what survives symbolic pressure.

```
Text → Spatial Encoding (128x128) → Reaction-Diffusion Dynamics → Attractor → Storage
                                                                        ↓
                                                              Stability-Weighted Recall
                                                                        ↓
                                                              Downstream AI / Agent
```

## Quick Start

### Prerequisites

- **Linux** (tested on Arch/CachyOS)
- **Python 3.10+**
- **Poetry**

### Install

```bash
git clone https://github.com/fantomx42/friendly-enigma.git wheeler-memory
cd wheeler-memory
poetry install
```

### Store and Recall

```python
import asyncio
from wheeler.core.memory import WheelerMemory

async def main():
    wm = WheelerMemory(".wheeler")
    await wm.initialize()

    await wm.store("The SCM Axioms state that meaning is what survives pressure.")
    await wm.store("Neural attractors encode stable representations of experience.")

    results = await wm.recall("What is meaning?")
    for r in results:
        print(f"{r['key'][:60]}  (score: {r['score']:.3f}, stability: {r['stability']:.2f})")

asyncio.run(main())
```

### CLI

```bash
# Store a memory
wheeler store "Consciousness is the ability to model yourself modeling the world."

# Recall by similarity
wheeler recall "self-awareness"

# Blend two concepts via reasoning engine
wheeler reason "entropy" "information"

# Run autonomic dreaming (consolidation + decay)
wheeler dream --ticks 10

# Visualize attractor evolution
wheeler viz-run "pattern recognition"

# Launch the web dashboard
wheeler dashboard
```

## How It Works

This is not RAG. There's no "save text, search later" step.

1. **Spatial Encoding** — Text encodes to a 128x128 spatial frame via deterministic bag-of-words character stamping (8x8 patterns).
2. **Reaction-Diffusion Dynamics** — A cellular automata engine (dt=0.1, diffusion=0.2, decay=0.01) evolves the frame until it settles into a stable attractor.
3. **Attractor = Memory** — The attractor IS the memory. The dynamics created it, not a lookup table.
4. **Physical Recall** — Recall compares attractor patterns via cosine similarity, not text strings. Similar meanings collapse to similar attractors.

### Stability Scoring

Memories are weighted by how well they survive pressure:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `hit_count` | 40% | Activation frequency (sigmoid normalized) |
| `frame_persistence` | 30% | Re-encoding produces the same frame |
| `compression_survival` | 30% | Pattern survives 10 ticks of dynamics |

Final recall score: `similarity * (0.5 + 0.5 * stability)`

### Reasoning

The `ReasoningEngine` operates directly on attractor frames:

- **Blend** — Weighted superposition of multiple memory frames
- **Contrast** — Difference between two frames (what makes A not-B)
- **Amplify** — Intensify the dominant features of a frame

### Autonomic System

Background consolidation and dreaming, inspired by biological memory:

- Consolidation strengthens frequently-accessed memories
- Dreaming runs dynamics over stored frames, letting weak memories decay
- Configurable tick rate (default: 30s)

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `wheeler/core/` | Engine: codec, dynamics, memory, storage, reasoning, autonomic, viz |
| `wheeler/cli/` | Click CLI (`store`, `recall`, `reason`, `dream`, `viz`, `dashboard`) |
| `wheeler/web/` | Flask dashboard with dark theme and memory cards |
| `twai/` | Next-gen Rust/Leptos frontend + Actix-web/SurrealDB backend |
| `tests/` | pytest + pytest-asyncio test suite |
| `scripts/` | Verification and HuggingFace ingestion scripts |
| `docs/` | SCM Axioms and architecture images |
| `conductor/` | TDD-driven development tracks |
| `legacy_archive/` | Original Ralph AI chat system and VS Code extension |

## TWAI

TWAI is the next-generation interface for Wheeler Memory, built in Rust. Leptos/WASM frontend with an Actix-web backend backed by SurrealDB. See [`twai/`](twai/) for details.

## SCM Axioms

1. Meaning is what survives symbolic pressure.
2. Meaning is the remainder after compression.
3. If a symbol collapses under pressure, it never had real meaning.
4. Meaning is not assigned. It's demonstrated.

Full axioms: [`docs/SCM_AXIOMS.md`](docs/SCM_AXIOMS.md)

## License

MIT

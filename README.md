# Ralph AI

Ralph is a local autonomous AI that pairs a single LLM with Wheeler Memory — a cellular automata system that stores patterns as stable attractors instead of text.

```
                         ┌─────────────────┐
                         │      Task       │
                         └────────┬────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │                     WHEELER MEMORY                       │
   │                                                          │
   │   text → 128x128 frame → dynamics → attractor → recall   │
   │                                                          │
   │   stability scoring weights what matters                 │
   └────────────────────────┬─────────────────────────────────┘
                            │
                   weighted context
                            │
                            ▼
                  ┌───────────────────┐
                  │  Qwen3 Coder Next │
                  │   (via Ollama)    │
                  └─────────┬─────────┘
                            │
                         response
                            │
                            ▼
                ┌─────────────────────┐
                │   feeds back into   │
                │   Wheeler Memory    │
                └─────────────────────┘
```

The LLM is an interface layer. Wheeler Memory is the intelligence.

## Wheeler Memory

This is not RAG. There's no "save text, search later" step.

**How it works:**

1. Text encodes to a 128x128 spatial frame (each character stamps an 8x8 pattern)
2. Cellular automata dynamics run until the frame settles into a stable attractor
3. The attractor IS the memory — it wasn't stored separately, the dynamics created it
4. Recall compares attractor patterns, not text strings
5. Similar patterns surface and blend

**Why attractors?**

Think of a ball rolling into a valley. The valley is the attractor — many starting positions end up there. Text that means similar things collapses to similar attractors, even if the words differ.

### Components

| Component | Description |
|-----------|-------------|
| Frames | 128x128 2D grids, each frame = one memory state |
| Dynamics | Reaction-diffusion CA with tanh activation, wrapped boundaries |
| Hit counting | Tracks activation frequency — learns what matters |
| Associations | Bidirectional links between co-accessed memories |
| Confidence | `0.4×hits + 0.3×diversity + 0.2×connectivity + 0.1×stability` |
| Tension detection | Flags contradictions with high-confidence beliefs instead of overwriting |
| Consolidation | Background process (60s) strengthens frequent associations |
| Dreaming | Background process (30s) randomly blends memories, finds latent connections |

### Stability Scoring

Memories get weighted by how well they survive pressure (SCM Axiom 2: meaning is the remainder after compression):

| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency (sigmoid normalized) |
| frame_persistence | 30% | Re-encoding produces the same frame |
| compression_survival | 30% | Pattern survives 10 ticks of dynamics |

Higher stability = more context budget when building prompts.

## SCM Bridge

The Symbolic Collapse Model connects Wheeler Memory to the LLM. Core axioms:

1. Meaning is what survives symbolic pressure
2. Meaning is the remainder after compression
3. If a symbol collapses under pressure, it never had real meaning
4. Meaning is not assigned — it's demonstrated
5. SCM doesn't measure meaning directly — it measures conditions where meaning can exist

See [docs/SCM_AXIOMS.md](docs/SCM_AXIOMS.md) for the full list.

The "Wheeler Bridge" retrieves patterns proven stable under dynamics, not just text that matches keywords. This is the fundamental difference from vector similarity search.

## Single Model Architecture

Previously Ralph used a multi-agent hierarchy (Translator → Orchestrator → Engineer → Designer). That's gone.

Now: one model does everything. The coordination that agents used to provide comes from Wheeler Memory instead — patterns that survived pressure guide the response.

**Current model:** `qwen3-coder-next` (80B MoE, 3B active per pass) via Ollama

The model handles parsing intent, generating code, reasoning through tasks. Wheeler Memory handles long-term knowledge, deciding importance, and forming beliefs.

## The Loop

1. Task arrives (human input or self-generated)
2. Wheeler Memory encodes text → runs dynamics → finds attractor
3. Stability scoring weights retrieved patterns
4. Weighted context + task → Qwen3 Coder Next
5. Model generates response
6. Result feeds back into Wheeler Memory as new experience
7. Grid evolves — patterns strengthen or fade based on outcomes

Terminates when the model outputs `<promise>COMPLETE</promise>`.

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

- **Ollama** running locally (https://ollama.com)
- **Python 3.10+** with `requests`
- **Base model:** `qwen2.5:3b` (development) or `qwen3-coder-next` (production)
- **GPU:** AMD ROCm or NVIDIA CUDA (CPU works for small models)

### Install Models

```bash
# Dev model (small, fast)
ollama pull qwen2.5:3b

# Embeddings
ollama pull nomic-embed-text

# Production
ollama pull qwen3-coder-next
```

## Repository Structure

| Directory | What it is |
|-----------|-----------|
| `wheeler_ai_training/` | Wheeler Memory — the core of Ralph |
| `ai_tech_stack/` | Runtime: loop script, security, tools, GUI |
| `docs/` | SCM axioms and documentation |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown.

## Hardware Tested

- Intel Core Ultra 7 265K
- AMD Radeon RX 9070 XT (16GB VRAM)
- 32GB DDR5 RAM
- CachyOS Linux

## License

MIT

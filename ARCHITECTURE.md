# Architecture

## Core Principle

Wheeler Memory is Ralph. The LLM is not.

Something comes in. The system changes because of it. That change IS the memory. There is no separate read step and write step. Every interaction creates a new attractor in the dynamics. The system learns by existing.

## System Diagram

```
                    ┌─────────────────────┐
                    │   Human / External  │
                    └─────────┬───────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │    LLM Interface    │
                   │  (swappable model)  │
                   │  qwen3-coder-next   │
                   └─────────┬───────────┘
                             │
                    text in / text out
                             │
              ┌──────────────▼──────────────┐
              │                             │
              │       WHEELER MEMORY        │
              │                             │
              │  ┌───────────────────────┐  │
              │  │     Text Codec        │  │
              │  │  text ↔ 128x128 frame │  │
              │  └───────────┬───────────┘  │
              │              │              │
              │  ┌───────────▼───────────┐  │
              │  │   Cellular Automata   │  │
              │  │   Reaction-Diffusion  │  │
              │  │   → finds attractor   │  │
              │  └───────────┬───────────┘  │
              │              │              │
              │  ┌───────────▼───────────┐  │
              │  │   Knowledge Store     │  │
              │  │   recall by frame     │  │
              │  │   similarity          │  │
              │  └───────────┬───────────┘  │
              │              │              │
              │  ┌───────────▼───────────┐  │
              │  │  Reasoning Engine     │  │
              │  │  blend / contrast /   │  │
              │  │  amplify patterns     │  │
              │  └───────────┬───────────┘  │
              │              │              │
              │  ┌───────────▼───────────┐  │
              │  │   Autonomic System    │  │
              │  │  consolidation +      │  │
              │  │  dreaming (background)│  │
              │  └───────────────────────┘  │
              │                             │
              └─────────────────────────────┘
```

The LLM sits outside Wheeler Memory. It is an interface — a way to convert between human language and the system. It can be swapped for any model. Wheeler Memory is the part that persists, learns, and holds meaning.

## How It Works

### 1. Input arrives

Text is encoded into a 128x128 2D frame via the TextCodec. Each character gets a deterministic spatial pattern (8x8, SHA256-seeded) stamped into a zone on the grid. Characters overlap and interfere.

### 2. Dynamics run

The frame is fed through cellular automata (reaction-diffusion with tanh activation, wrapped boundaries). After N ticks, the system settles into an attractor — a stable pattern. This attractor IS the memory of that input. It wasn't stored separately; the dynamics created it.

### 3. Recall happens through similarity

The attractor is compared against existing frames in the Knowledge Store using correlation. Similar patterns surface. Hit counts go up. Associations form. There's no lookup table — recall is a physical process of pattern matching.

### 4. Reasoning is pattern manipulation

- **Blend**: Superimpose frames (weighted average) — association
- **Contrast**: Subtract frames — find what's different
- **Amplify**: Non-linear strengthening of dominant patterns

### 5. The system runs in the background

The Autonomic System consolidates frequently co-accessed memories (every 60s) and dreams — randomly blending memories to discover latent connections (every 30s).

### 6. Tension detection

When new input is similar (but not identical) to a high-confidence existing memory, the system flags it as a tension rather than blindly overwriting. This is epistemological independence — the system can hold its ground.

## Stability Scoring

Memories are weighted by how well they survive pressure (SCM Axiom 2: meaning is the remainder after compression):

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| hit_count | 40% | How often this memory gets activated |
| frame_persistence | 30% | Does re-encoding produce the same frame? |
| compression_survival | 30% | Does the pattern survive 10 ticks of dynamics? |

Higher stability = more context budget when the LLM constructs prompts.

## Confidence Scoring

Individual memories track their own confidence:

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| hit_count | 40% | Activation frequency (sigmoid-normalized) |
| reinforcement_diversity | 30% | Reinforced from how many different contexts? |
| connectivity | 20% | How many associations to other memories? |
| stability | 10% | Resistance to drift |

## What the LLM Does (and Doesn't Do)

The LLM (currently `qwen3-coder-next`, 80B MoE) handles:
- Parsing human intent from natural language
- Generating human-readable output
- Tool use and code generation within a task loop

The LLM does NOT:
- Store long-term knowledge (Wheeler Memory does)
- Decide what's important (stability scoring does)
- Form beliefs (confidence + tension detection does)

The model is specified by `RALPH_MODEL` env var and can be changed without affecting memory.

## Theoretical Foundation: Symbolic Collapse Model

See [SCM Axioms](docs/SCM_AXIOMS.md) for the full list. The key ideas:

1. Meaning is what survives symbolic pressure
2. Meaning is the remainder after compression
3. If a symbol collapses under pressure, it never had real meaning
4. Stability demonstrates meaning — stories don't create it

## What Was Removed

The previous architecture used a multi-agent hierarchy:
- Translator (phi3:mini) — human input to TaskSpec
- Orchestrator (deepseek-r1:14b) — strategic planning
- Engineer (qwen2.5-coder:14b) — code generation
- Designer (mistral-nemo:12b) — code review
- ASICs — ultra-small specialist models for micro-tasks

This has been replaced by a single model approach. The agent code still exists in `ai_tech_stack/ralph_core/agents/` but is not used. The message bus, circuit breakers, and multi-agent protocols are legacy.

## Repository Structure

```
.
├── wheeler_ai_training/     # Wheeler Memory implementation
│   ├── wheeler_ai.py        # WheelerAI: codec, reasoning, knowledge, autonomics
│   ├── wheeler_cpu.py        # Cellular automata dynamics engine
│   ├── models.py             # Text-to-grid autoencoder (neural network variant)
│   └── train.py              # Training script for neural network variant
├── ai_tech_stack/            # Ralph runtime (submodule)
│   ├── ralph_simple.py       # Single-model loop with Wheeler integration
│   ├── ralph_loop.sh         # Entry point script
│   ├── ralph_core/           # Core runtime (security, tools, memory bridge)
│   └── ralph_gui/            # Native desktop GUI (Rust/egui)
├── conductor/                # TDD project management framework
├── docs/
│   └── SCM_AXIOMS.md         # Symbolic Collapse Model axioms
└── CLAUDE.md                 # Development reference
```

---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #c084fc;
  }
  h2 {
    color: #67e8f9;
  }
  code {
    background: #1e1b4b;
    padding: 2px 6px;
    border-radius: 3px;
  }
  table {
    font-size: 0.9em;
  }
---

# Wheeler Memory
## Project Darman — v0.3.6

**A reconstructive memory substrate for AI — pure-Python, native, local-first.**

We will not pitch the next AGI. We will pitch a real, tested, honest memory engine.

---

## The Problem

### Why memory matters in LLMs

- **Confabulation**: LLMs hallucinate confidently with no uncertainty signal
- **Statelessness**: no memory across sessions = no real personalization
- **Information loss**: context windows close; conversational history is discarded
- **Vector DBs are filing cabinets**: same query, same row, no reconstruction

**Today's LLMs are amnesiacs who don't admit it.**

---

## Market opportunity

### The LLM market needs a memory layer that is honest about itself

| Market segment | Pain |
|---|---|
| Enterprise copilots | Need reliable recall + auditable confidence |
| AI search / RAG | Static retrieval ≠ memory |
| Agentic systems | Persistent state, forgetting curves |
| Personalized assistants | Associative recall, context-dependent reconstruction |

Every major LLM platform is now exploring "memory." The differentiator we've been building is: **memory that admits what it knows, doesn't know, and can't decide.**

---

## Our solution: Wheeler Memory

### A cellular-automaton memory engine

```
text → encoder → 64×64 frame → CA evolution (5–14 ticks) → attractor
                                                                ↓
query → recognize() → BasinSeed | None
                                                                ↓
       reconstruct_from_seed(seed, query) → Pattern
```

- **3-state CA dynamics** (peak/valley/slope) instead of vectors
- **Reconstructive recall** — every recall is context-dependent
- **Three-grid interference** — corpus × experiential × SCM-trust topology
- **Two-tier API** (v0.3.6) — cheap recognition vs expensive reconstruction
- **Native encoders** — no pretrained models in the core
- **Local-first** — runs entirely on-device

---

## Key differentiator #1: reconstructive recall

### Same memory, different reconstructions

```
blend = (1 − α) × stored + α × query        (α = 0.3 default)
reconstructed = evolve_and_interpret(blend)
```

- Query "machine learning" vs "debugging" → different reconstructions of the *same* stored attractor
- Aligned with cognitive science (Loftus on reconstructive memory)
- **Vector DBs can't do this** by construction — they're row stores

---

## Key differentiator #2: two-tier recall (v0.3.6 headline)

### Identity is cheap. Content is expensive. Separate them.

| Tier | What it does | Cost |
|---|---|---|
| **Recognize** | Single Pearson scan against stored attractors using the raw query frame. Returns a `BasinSeed` (id, similarity, basin stability) or `None`. | No CA loop on the query |
| **Reconstruct** | Warm-start the CA from the stored attractor. Re-evolve. | ~2× fewer ticks than cold start across distance bands |

Plus per-basin **Temporal Stability `T`**: drift `(1 − T) × base_rate` toward observed pattern on each `--learn` recall. Mature basins become rigid; fresh basins absorb new context.

---

## Key differentiator #3: three-grid interference

### Epistemic state emerges from physics, not heuristics

```
Answer = Corpus × Experiential × (1 − |SCM|)
```

- **Corpus** — crystallized knowledge, tight attractors
- **Experiential** — episodic memory, loose attractors, fast decay
- **SCM** — Structural Coherence Map (permission topology, sculpted by self-consistency feedback)

Four interference states fall out: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED. *No state is hand-coded.*

---

## Key differentiator #4: native by default

### No pretrained models in the core

- **Hippocampus** — character n-gram random indexing
- **Context-RI** — distributional semantics, trained on **601M words** (WikiText-103 + OpenWebText). **First native encoder with positive SimLex-999 signal**: ρ = +0.255, vs MiniLM's pretrained ceiling of +0.446 (57% of ceiling, no pretrained models).
- **Cortex L1/L2/L3** — graph topology + settlement CA + numpy SGD classifier (~11K params)

`sentence-transformers` is *optional*, behind `pip install -e ".[embed]"`.

---

## Key differentiator #5: temperature & forgetting

### Confidence is computed, not guessed

```
temp = base_from_hits × decay_from_time     (7-day half-life default)
```

| Tier | Range | Meaning |
|---|---|---|
| **Hot** | ≥ 0.6 | Recent or frequently recalled |
| **Warm** | ≥ 0.3 | Middle ground |
| **Cold** | ≥ 0.05 | Rarely accessed |
| **Fading / Dead** | < 0.05 / < 0.01 | Eviction candidates |

Sleep consolidation prunes redundant keyframes from cold bricks. Capacity ceiling: 10,000 attractors with graceful degradation.

---

## Key differentiator #6: local-first, privacy-preserving

### No cloud. No API keys. No data escape.

- All computation on-device (CPU or local GPU)
- All storage local (`~/.wheeler_memory/`)
- Optional Ollama for LLM rendering — entirely local
- HIPAA / GDPR / sovereignty story is **architectural**, not policy-based

Enterprises demand data sovereignty. We deliver it by construction.

---

## Honest technical numbers

### Things we say with citations, not adjectives

| Metric | Value | Source |
|---|---|---|
| Modules / tests / CLIs | 44 / 775 / 16 | repo |
| CA evolution speed | ~3 ms/tick CPU | `wheeler-bench` |
| GPU speedup (RX 9070 XT, batch=1000) | **71×** | `wheeler-bench-gpu` |
| Two-tier warm-vs-cold ticks | ~2× fewer | `bench_recall_warm_vs_cold.py` |
| Context-RI SimLex-999 ρ | +0.255 | `wheeler-simlex` |
| MiniLM ceiling (external, ref) | +0.446 | `wheeler-simlex` |
| MMLU 57-subject zero-shot cortex | 24.3% | `wheeler-mmlu --all --mode cortex` |
| MMLU + L3 classifier | 25.9% | same |
| Random chance | 25.0% | — |

L3 barely beat chance. We say so. We then say what fixes it.

---

## What we have NOT built

### Said out loud

- No multimodal (no images, no audio)
- No federated memory
- No browser runtime
- No live web dashboard right now (the prior `wheeler-ui` was retired in v0.3.6 because the script had drifted out of date with the core)

The pitch is "this is real, here's what works, here's what doesn't."

---

## Engine philosophy

### "The engine is the mind. The LLM is the voice."

Swap GPT-4 → Mistral → local 1.5B model and Wheeler's *behavior* stays the same. Only *phrasing* changes.

- Behavior is deterministic, auditable, provable
- Not dependent on the LLM's capability
- Cheaper to operate at scale (small models work)
- Personality emerges from memory, not from prompt engineering

`wheeler-primary --interactive --show-state` shows both layers explicitly.

---

## Architecture: 44 modules, deep stack

```
agent.py / decoder.py / language_wheeler.py     ← optional renderers
       ↓
recall_api.py (recognize / reconstruct_from_seed)   ← v0.3.6
       ↓
interference.py (three-grid scoring)            ← v0.3.1+
       ↓
storage.py (chunked Pearson search)             ← sacred
       ↓
cortex.py / cortex_scm.py / cortex_classifier.py    ← native L1/L2/L3
       ↓
dynamics.py + accel/ca.py (CPU + HIP/ROCm)      ← CA substrate
```

Modular. Testable. CC BY-NC 4.0.

---

## Competitive positioning

### Wheeler vs alternatives

| Aspect | Vector DB | RAG | Wheeler Memory |
|---|---|---|---|
| Retrieval | Static | Static | **Context-dependent reconstruction** |
| Confidence signal | None | None | **Temperature + interference state** |
| Identity vs content | Conflated | Conflated | **Two-tier (recognize / reconstruct)** |
| Per-item plasticity | None | None | **Per-basin Temporal Stability** |
| Local-first | Sometimes | Sometimes | **Always** |
| Forgetting | Manual | Manual | **Automatic (7d half-life)** |
| Pretrained dependency | Yes | Yes | **Optional** |

A new category, not an upgrade.

---

## Status: open-source, real, tested

- ✓ Core CA engine (5–14 ticks to converge with tuned dynamics)
- ✓ 775 automated tests, all passing
- ✓ Three-grid interference (default since v0.3.1)
- ✓ SCM telemetry + closed-loop A/B eval (v0.3.4)
- ✓ Two-tier recall + per-basin T (v0.3.6 headline)
- ✓ Native distributional encoder (601M-word Context-RI)
- ✓ HIP/ROCm GPU with CPU fallback
- ✓ MMLU benchmark runner (57 subjects, multiple modes)

No VC funding yet. Built by the founding team. CC BY-NC 4.0 on GitHub.

---

## Named after Wheeler's "It from Bit"

> "Every 'it' — every physical thing — derives its existence as an information system from answered physical questions."
> — John Archibald Wheeler

In our system, meaning emerges from CA dynamics. Memory isn't stored; it's *computed*. Epistemic states emerge from interference; they aren't hand-coded.

**Convergence is ground truth.**

---

## Path to commercialization

### Dual licensing model

- **CC BY-NC 4.0** for open-source / non-commercial users
- **Commercial license** for production deployments
- **Managed service** (Wheeler-as-a-Service) for API consumers
- **Enterprise integration** for sovereignty-sensitive verticals (healthcare, finance, defense)

The substrate is real. Productization is a deliberate next step, not a leap.

---

## What we need

### Seed round

- 12–18 month runway to commercial product
- 3 enterprise pilot customers in privacy-sensitive verticals
- Contributor growth in the open-source codebase
- Compute budget for the next two architectural moves: reconstruction scoring + offline T consolidation

We are honest about what works and what doesn't. We expect the same honesty in return.

---

## Try it now

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e .
wheeler-store "self-attention computes relationships between all positions"
wheeler-recall "how does attention work in transformers"
wheeler-recall "..." --recognize --learn   # two-tier path with T accumulation
```

Static demo: open `docs/demos/demo.html` in a browser.

**Contact**: [your email / website]

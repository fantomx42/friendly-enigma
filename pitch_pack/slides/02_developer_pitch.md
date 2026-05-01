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
    font-size: 0.85em;
  }
---

# Wheeler Memory
## Open-source cellular-automaton memory for LLMs — v0.3.6

**Build agentic systems that remember, forget, reconstruct — without depending on pretrained models in the core.**

---

## Why this matters

### The LLM memory problem

- **Vector DBs are filing cabinets**: same query → same row, no reconstruction
- **Reconstruction is novel**: human memory does *not* return verbatim — context reshapes recall
- **Forgetting is useful**: models that never decay never reach epistemic humility
- **Identity ≠ content**: knowing "which memory is this" is cheap; knowing "what does it say" is expensive — but most APIs conflate them

We treat databases like memory. Wheeler is something different.

---

## What you can build

| Use case | What's possible |
|---|---|
| **Agentic systems** | Persistent state, forgetting curves, contextual decision-making |
| **Chatbots** | Personalized recall that reconstructs based on conversation context |
| **Semantic search** | Pearson + spatial similarity, with associative warmth (2-hop spreading activation) |
| **Memory evaluation** | Convergence-state distribution, oscillation detection, basin-stability metrics |
| **Privacy-first LLMs** | All-local memory, no cloud, no pretrained model in the core |
| **Native semantic** | Context-RI distributional encoder trained on 601M words, 0 pretrained weights |

Pure Python. Run anywhere. CC BY-NC 4.0.

---

## Quick start

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e .                           # core (numpy/scipy/matplotlib/psutil)
# pip install -e ".[embed]"                # optional: sentence-transformers
```

```bash
wheeler-store "self-attention computes weighted relationships between positions"
wheeler-recall "how does attention work in transformers"
wheeler-recall "..." --recognize           # v0.3.6: identity only, no CA on query
wheeler-recall "..." --recognize --learn   # v0.3.6: + per-basin Temporal Stability
```

No GPU required. CPU works fine. Python 3.11+ only.

---

## Encoders (no pretrained models in the core)

| Encoder | Cost | Notes |
|---|---|---|
| `hash` | 0 | SHA-256 deterministic; default for benchmarks |
| `hippocampus` | low | Native character n-gram random indexing |
| `blended` (default) | low | hippocampus(0.7) + language wheeler(0.3) |
| `context` | medium | Distributional RI, 601M words, **SimLex-999 ρ=+0.255** |
| `embedding` | high | Optional MiniLM via `pip install -e ".[embed]"` |
| `word`, `word-blended` | low | Word-level RI variants |

**Context-RI is the headline**: first native encoder with positive SimLex signal — 57% of MiniLM's pretrained ceiling, no pretrained models.

---

## The 3-state CA rule

```python
# Conceptual, real version is in wheeler_memory/dynamics.py
for cell in grid:
    neighbors = von_neumann(cell)             # 4-connected, wrapping
    if cell == max(neighbors):
        cell += MAX_PUSH_STRENGTH * (+1 - cell)   # push toward +1
    elif cell == min(neighbors):
        cell += MAX_PUSH_STRENGTH * (-1 - cell)   # push toward -1
    else:
        cell += SLOPE_FLOW_STRENGTH * (uphill_neighbor - cell)
```

- Continuous values in [−1, +1] (not strictly 3-state)
- Default `MAX_PUSH_STRENGTH = 0.57`, `SLOPE_FLOW_STRENGTH = 0.55`
- Convergence: **5–14 ticks** with current tuning
- ~3 ms/tick on CPU; **71× speedup** on RX 9070 XT at batch=1000

---

## Convergence states

| State | Meaning | What we do |
|---|---|---|
| **CONVERGED** | Stable fixed point reached | Store the attractor |
| **OSCILLATING** | Periodic cycle detected | Surface as epistemic uncertainty |
| **CHAOTIC** | Iteration cap exhausted | Reject the seed, retry with rotation |
| **DEGENERATE** | <5% alive cells | Reject — 0-dominant frame |

`oscillation.py` detects cycles by role-space periodicity. `rotation.py` retries seeds at 90/180/270° to escape bad initial conditions.

---

## Two-tier recall (v0.3.6 headline)

```python
from wheeler_memory import recognize, recognize_top_k, reconstruct_from_seed

seed = recognize("how does attention work")        # BasinSeed | None
if seed is not None:
    pattern = reconstruct_from_seed(seed,
                                    query="how does attention work",
                                    alpha=0.3)
    print(pattern.text, pattern.convergence_ticks)

# Top-k for ranked identity-only use
seeds = recognize_top_k("how does attention work", k=5)
```

- Recognition does a single Pearson scan over stored attractors using the **raw query frame** — no CA loop on the query.
- Reconstruction warm-starts the CA from the stored attractor — ~2× fewer ticks vs cold start across distance bands (`scripts/bench/bench_recall_warm_vs_cold.py`).

---

## Per-basin Temporal Stability (T)

Each stored basin carries a `T ∈ [0, 1]` in its `index.json` metadata.

```
T_new = (1 − T_EMA_RATE) × T_old + T_EMA_RATE × observed_stability
drift_rate = (1 − T) × BASIN_DRIFT_BASE_RATE
new_basin = stored + drift_rate × (observed − stored)
```

- Defaults: `T_INIT_DEFAULT = 0.0`, `T_EMA_RATE = 0.1`, `BASIN_DRIFT_BASE_RATE = 0.02`
- Fresh basins (T=0) absorb new context fast; mature basins (T→1) become rigid
- Updates apply only when `learning_enabled=True` (off by default)
- Persisted under one fcntl lock per chunk, mmap-safe

---

## Three-grid interference (default recall path)

```
Answer(i,j) = Corpus(i,j) × Experiential(i,j) × (1 − |SCM(i,j)|)
```

| Grid | Role | Push | Decay |
|---|---|---|---|
| **Corpus** | Crystallized knowledge | 0.57 (tight) | 7-day half-life |
| **Experiential** | Episodic memory | 0.35 (loose) | 2-day half-life |
| **SCM** | Trust topology (permission, not content) | — | Sculpted by self-consistency feedback |

Four interference states emerge: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED.

`scm_telemetry.jsonl` captures every SCM event. `scripts/scm_ab_eval.py` runs the closed-loop A/B eval.

---

## Reconstructive recall

```python
blend = (1 - α) * stored_attractor + α * query_seed
reconstructed = evolve_and_interpret(blend)   # re-evolve under CA
```

- **Default α = 0.3** — memory-dominant
- α = 0.5 — balanced
- α = 0.7 — query-dominant (stress test, hallucination probing)
- Aligned with Loftus on reconstructive memory

`reconstruction.py` (the live primitive) and the v0.3.6 wrapper `reconstruct_from_seed` both use this formula. Same memory, different reconstructions in different contexts.

---

## Temperature

```
temp = base_from_hits × decay_from_time
base_from_hits  = min(1.0, 0.3 + 0.7 × hit_count / HIT_SATURATION)
decay_from_time = 2 ^ (-days_since_last_access / HALF_LIFE_DAYS)
```

Defaults: `HIT_SATURATION = 10`, `HALF_LIFE_DAYS = 7.0`.

| Tier | Range | Behaviour |
|---|---|---|
| **Hot** | ≥ 0.6 | Recent or frequently recalled |
| **Warm** | ≥ 0.3 | Default for new memories |
| **Cold** | ≥ 0.05 | Recall-eligible but stale |
| **Fading** | ≥ 0.01 | Eviction candidates |
| **Dead** | < 0.01 | Full eviction |

Capacity ceiling: `MAX_ATTRACTORS = 10_000`. `EVICTION_RATIO = 0.10`. `MIN_AGE_DAYS = 1.0`.

---

## Cortex L1/L2/L3 (native semantic scoring)

```
retrieved attractors
        ↓
L1: Pearson adjacency graph + BFS clustering
        ↓
L2: settlement CA — opinion diffusion until convergence
        ↓
L3: native classifier — numpy SGD, ~11K params
```

- L1 in `cortex.py`
- L2 in `cortex_scm.py` (Soft Constraint Satisfaction)
- L3 in `cortex_classifier.py`

`scripts/train_cortex_classifier.py` trains it. We say honestly: L3 loss barely moved from chance — needs more data or richer features. The next move is **reconstruction scoring** (let the CA settle and read the answer off the attractor).

---

## Architecture: 44 modules

```
ENCODING
  hashing.py / hippocampus.py / embedding.py / word_encoder.py / brick.py

CA ENGINE
  dynamics.py / oscillation.py / rotation.py
  accel/ca.py + accel/hip/*  (HIP/ROCm GPU bindings)
  npu/ scaffolding for OpenVINO + Coral (stub today)

STORAGE & RECALL
  storage.py (sacred) / chunking.py (sacred) / cache.py
  reconstruction.py / recall_api.py / t_metadata.py

THREE-GRID INTERFERENCE
  scm_grid.py / experiential.py / interference.py / similarity.py
  trajectory.py / trajectory_cache.py

CORTEX
  cortex.py / cortex_scm.py / cortex_classifier.py

LIFECYCLE
  temperature.py / warming.py / consolidation.py / eviction.py / attention.py

THEORIES (production helpers)
  theories/basin.py / theories/metrics.py / theories/synthesis.py
  (lichtenberg, resonance, structured archived to notes/theories/)

AGENTS & RENDERING
  agent.py / decoder.py / language_wheeler.py / generation.py

CONFIG
  constants.py (only file edited during autoresearch)
```

---

## Validation: 44 modules, 775 tests

| Group | Coverage |
|---|---|
| Dynamics | CA convergence, batch parity (CPU vs GPU), oscillation detection |
| Storage | Pearson search, chunking, fcntl locking, cache invalidation |
| Recall | `recognize` no-CA-loop guarantee, two-tier drift, top-k ordering |
| Interference | Three-grid scoring, four state classification, SCM cold-start |
| Reconstruction | Blending fidelity, α tuning, context dependence |
| Temperature | Decay curves, tier thresholds, eviction sweeps |
| Cortex | L1 graph correctness, L2 settlement, L3 classifier training |
| Theories | basin width, energy, hallucination classification, apple test |

```bash
pytest -m "not slow"   # full suite
```

---

## CLI surface (16 commands)

| Command | Purpose |
|---|---|
| `wheeler-store` | Encode + store a memory |
| `wheeler-recall` | Default recall (three-grid). Flags: `--recognize`, `--learn`, `--no-interference`, `--encoder`, `--embed` |
| `wheeler-temps` | List memories with temperature/freshness |
| `wheeler-forget` | Delete memories by text |
| `wheeler-sleep` | Run sleep consolidation |
| `wheeler-agent` / `wheeler-primary` | LLM wrappers (require Ollama) |
| `wheeler-crystallize` | Pre-train from a JSONL/CSV/TXT/Parquet corpus |
| `wheeler-mmlu` | MMLU benchmark runner — `semantic`, `cortex`, `learn`, `learn-interference`, `recall-text`, `decode`, `ternary*`, `multi-choice`, `reverse-lookup`, `spatial` modes |
| `wheeler-bench`, `wheeler-bench-gpu` | Quality + GPU benchmarks |
| `wheeler-info`, `wheeler-scm`, `wheeler-scrub`, `wheeler-simlex`, `wheeler-generate` | Diagnostics, SCM topology, brick inspector, SimLex eval, generative engine |

There is no `wheeler-ui` web server in v0.3.6 — the prior implementation was retired because it had drifted out of date with the core. Static demos live at `docs/demos/`.

---

## Extensibility

```python
# Custom encoder
from wheeler_memory.storage import store_memory
store_memory("...", encoder="hash")          # any registered encoder
store_memory("...", encoder=my_custom_encoder)

# Custom chunk
from wheeler_memory import chunking
# (chunking.py is sacred — extension via configuration, not code edits)

# Custom CA dynamics
from wheeler_memory.dynamics import apply_ca_dynamics_parameterized
apply_ca_dynamics_parameterized(frame, push=0.45, slope=0.35)

# GPU dispatch is automatic — accel/ca.py auto-falls back to CPU
```

Extension is config-driven where possible. The sacred files (`hashing.py`, `storage.py`, `chunking.py`, `rotation.py`, `bench_quality.py` TEST_INPUTS) are off-limits by design.

---

## GPU acceleration

```bash
# Build
cd wheeler_memory/accel/hip
make                           # default: gfx1201 (RX 9070 XT / RDNA 4)
GPU_ARCH=gfx1100 make          # RDNA 3 (RX 7000 series)
```

- HIP kernels in `wheeler_memory/accel/hip/`
- Python bindings in `wheeler_memory/accel/ca.py`
- Auto-fallback to CPU when no `.so` is built or GPU is unavailable
- v2 kernel supports variable grid sizes (64×64 to 1000×1000)
- 71× speedup at batch=1000 on RX 9070 XT

---

## Honest MMLU numbers

All 57 subjects, 14,042 questions, test split:

| Run | Score | Δ vs chance |
|---|---|---|
| Zero-shot cortex (0 stored memories) | **24.3%** | -0.7% |
| Cortex + 1,812 learned facts | **25.3%** | +0.3% |
| Cortex + L3 classifier | **25.9%** | +0.9% |
| Random chance | 25.0% | — |

The previous external-MiniLM baseline at 27.5% was **removed** because it depended on a pretrained model. We don't fake the win.

Next move: **reconstruction scoring** — evolve the query, settle the CA, read the attractor's answer, compare to choices.

---

## Empirical work the repo carries

- **SimLex-999** with `wheeler-simlex` — Context-RI ρ=+0.255 (best native)
- **Apple Test** (`scripts/bench/apple_test_semantic.py`) — exclude a concept from a domain, crystallize neighbors, query for the held-out concept; ML domain shows weak topology, Physics + Biology silent
- **Diversity / paraphrase reports** in `docs/assets/reports/`, generated by `notes/exploration/test_diversity*.py`, `test_paraphrase*.py`
- **Warm-vs-cold ticks** with `scripts/bench/bench_recall_warm_vs_cold.py` — three input-distance bands, recognition rate reported separately

All reproducible from the repo. Datasets live under `datasets/` (gitignored for the large ones).

---

## Sacred files & autoresearch

`wheeler_memory/CLAUDE.md` documents the off-limits list:

- `hashing.py` — deterministic SHA-256, foundational
- `bench_quality.py` TEST_INPUTS — fixed corpus, comparable across experiments
- `storage.py` — locked storage contract
- `chunking.py` — locked domain routing
- `rotation.py` — locked rotation logic

Autoresearch protocol: edit only `constants.py`, run `wheeler-bench --commit <hash7> --changed "<param>"`, keep if score improved, revert if it dropped >10%. See `program.md`.

---

## What's not built (said out loud)

- No multimodal (no images, no audio, no CLIP, no wav2vec)
- No federated memory or peer-to-peer sync
- No browser / WebAssembly runtime
- No live web dashboard right now
- No reconstruction-scoring MMLU mode yet (next on the roadmap)

This list exists on purpose. The pitch is real, not heroic.

---

## How to contribute

### High-impact areas

- **Reconstruction scoring** — the next MMLU mode where the CA itself answers the question
- **Sleep consolidation of T** — offline pass that consolidates accumulated Temporal Stability into the stored corpus
- **SCM scoring fix** — replace scalar `mean_openness` with spatial product per candidate (known issue)
- **Multimodal encoders** — CLIP for images, wav2vec for audio
- **Native verb semantics** — Context-RI's verb ρ is +0.05; bag-of-words distributional methods just are like that, but a context-aware variant could close the gap

### Documentation

- Tutorials for two-tier recall use cases
- Reconstruction-scoring walkthrough once it lands
- IIT / Wheeler "It from Bit" connection writeups

---

## Philosophy: convergence is ground truth

### Sacred (rare changes, deliberate version bumps)

- The CA rule (3-state local dynamics)
- Pearson correlation as the recall similarity primitive
- The SHA-256 encoding for hash-mode memories
- The chunked storage contract

### Commentary (extend freely)

- Encoders, decoders, agents
- CLI tools
- GPU kernels (CUDA path is open if anyone wants it)
- Cortex variants
- New benchmarks

Upstream changes are rare; downstream extensions are encouraged.

---

## Try it now

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory && pip install -e .
wheeler-store "..." && wheeler-recall "..."
wheeler-recall "..." --recognize --learn   # v0.3.6 two-tier path
wheeler-mmlu --subjects high_school_biology --mode cortex --samples 20
```

Static demo: open `docs/demos/demo.html` in a browser.

GitHub: **`github.com/fantomx42/wheeler-memory`** — CC BY-NC 4.0.

---

# "Convergence is ground truth."

**Star the repo. Read the docs. Build with us.**

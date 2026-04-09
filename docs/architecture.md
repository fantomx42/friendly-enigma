# Architecture

## Overview

```
Input Text
    ↓
Encode  ─── hash         SHA-256 deterministic (exact match)
        ├── hippocampus   character n-gram random indexing (native, no pretrained models)
        ├── embedding     MiniLM sentence-transformer (requires .[embed])
        ├── blended       hippocampus(0.7) + language_wheeler(0.3) ← DEFAULT
        ├── word          word-level random indexing (SVD on PMI matrix)
        ├── context       context-window RI (distributional semantics, trained on WikiText-103)
        └── word-blended  hippocampus + word encoder hybrid
    ↓
64×64 seed frame (float32, values in [-1, +1])
    ↓
3-State CA Evolution ──── rotation retry (0°/90°/180°/270°)
    ├── CONVERGED      → store attractor + brick
    ├── OSCILLATING    → epistemic uncertainty detected
    ├── DEGENERATE     → 0-dominant attractor (<5% alive cells)
    └── CHAOTIC        → input needs rephrasing
    ↓
Attractor: saved as .npy in chunk/attractors/
Brick:     saved as .npz in chunk/bricks/
```

---

## 1. Chunked Storage

Memories are routed to domain-specific sub-stores called **chunks**, inspired
by cortical region specialisation. Each chunk has its own directory tree under
`~/.wheeler_memory/chunks/<name>/`.

### Named chunks

| Chunk | Representative keywords |
|---|---|
| `code` | python, rust, bug, debug, compile, git, docker, sql, javascript, … |
| `hardware` | printer, 3d print, solder, gpio, pcb, arduino, bambu, filament, … |
| `daily_tasks` | grocery, dentist, schedule, meeting, errand, laundry, workout, … |
| `science` | physics, equation, quantum, genome, calculus, theorem, molecule, … |
| `meta` | wheeler, attractor, brick, cellular automata, rotation, chunk, … |
| `general` | (fallback — anything that doesn't match another chunk) |

### Routing

**Store** — `select_chunk(text)` counts keyword hits for every named chunk and
picks the winner. Ties go to `general`.

**Recall** — `select_recall_chunks(query)` selects up to 3 chunks by hit score,
then appends `general` and any on-disk chunks not yet selected. Recall therefore
always includes `general` plus whichever domain(s) the query resembles most.

### Directory layout

```
~/.wheeler_memory/chunks/<name>/
├── attractors/          # one .npy per memory (64×64 float32)
├── bricks/              # one .npz per memory (full evolution history)
├── index.json           # { hex_key: { text, state, timestamp, metadata … } }
└── metadata.json        # last_accessed, store_count for the chunk itself
```

---

## 2. Memory Bricks

A **MemoryBrick** is the complete temporal record of how a memory formed — every
CA frame from the initial seed to the final attractor.

```python
@dataclass
class MemoryBrick:
    evolution_history: list[np.ndarray]   # one 64×64 frame per tick
    final_attractor:   np.ndarray          # last stable state
    convergence_ticks: int                 # ticks taken to converge
    state:             str                 # CONVERGED | OSCILLATING | DEGENERATE | CHAOTIC
    metadata:          dict                # rotation_used, wall_time_seconds, …
```

Bricks are saved as compressed `.npz` files via `MemoryBrick.save()` and loaded
with `MemoryBrick.load()`. The stacked history array and metadata JSON live
together in a single file.

### Visualising a brick

```bash
wheeler-scrub --text "fix the python debug error"
# Opens an interactive matplotlib viewer with a tick slider
```

### Debugging oscillating bricks

`find_divergence_point()` scans the evolution history backwards using
`get_cell_roles()` to locate the tick where periodicity began. This is useful
for identifying inputs that reliably produce oscillating attractors — a signal
that the input may benefit from rephrasing or a different rotation angle.

```python
from wheeler_memory.brick import MemoryBrick

brick = MemoryBrick.load("~/.wheeler_memory/chunks/code/bricks/<hex>.npz")
if brick.state == "OSCILLATING":
    t = brick.find_divergence_point()
    print(f"Oscillation started at tick {t}")
    print(f"Period: {brick.metadata.get('cycle_period')} ticks")
```

---

## 3. Temperature Dynamics

Every memory has a **temperature** in `[0, 1]` that reflects how recently and
frequently it has been recalled.

### Formula

```
temp = base_from_hits × decay_from_time

base_from_hits  = min(1.0,  0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (−days_since_last_access / 7)
```

Constants: **half-life = 7 days**, **hit saturation = 10 hits**.

A brand-new memory (0 hits, recalled this instant) starts at `0.3 × 1.0 = 0.3`.
After 10+ recalls it can reach `1.0`. After 7 days without recall the
temperature halves.

### Tiers

| Tier | Threshold | Meaning |
|---|---|---|
| `hot` | ≥ 0.6 | Frequently accessed and recent |
| `warm` | ≥ 0.3 | Default for new or moderately accessed memories |
| `cold` | < 0.3 | Stale — candidate for archival |

### Access tracking

`bump_access(entry)` increments `hit_count` and updates `last_accessed` to
`utcnow()` every time a memory appears in a recall result. This happens
automatically inside `recall_memory()`.

Temperature is factored into ranking when `temperature_boost > 0.0`:

```
effective_similarity = pearson_correlation + temperature_boost × temperature
```

List temperatures with:

```bash
wheeler-temps
```

---

## 4. Rotation Retry

CA dynamics are sensitive to initial conditions. Some seed frames fall into
oscillating or chaotic trajectories that never converge. **Rotation retry**
escapes these basins by physically rotating the seed frame before re-evolving:

```
Attempt 1:   0°  → evolve → CONVERGED? → store & return
Attempt 2:  90°  → evolve → CONVERGED? → store & return
Attempt 3: 180°  → evolve → CONVERGED? → store & return
Attempt 4: 270°  → evolve → still fails → return FAILED_ALL_ROTATIONS
```

Rotation changes the neighbour topology of every cell, placing the dynamics on
a different trajectory through state space. In practice, 0° covers the vast
majority of inputs; 90°/180°/270° act as safety nets for edge cases.

Per-angle success counts are persisted in `~/.wheeler_memory/rotation_stats.json`:

```json
{ "0": 142, "90": 3, "180": 1, "270": 0 }
```

This lets you audit how often the system needs to retry and at which angles
convergence tends to succeed.

---

## 5. Open WebUI Integration

Wheeler Memory ships a pipeline for [Open WebUI](https://openwebui.com) that
injects relevant memories as a system-prompt prefix before every LLM response.

### How it works

1. The pipeline's `pipe()` method receives the user message.
2. `recall_memory()` is called with `use_embedding=True` and `reconstruct=True`
   so results are semantically matched and context-biased via the Darman
   reconstruction architecture.
3. Results above `min_similarity=0.1` are formatted:
   ```
   [Wheeler Memory - Episodic Context]
   [HOT 0.87] "fix the python debug error" (sim=0.34)
   [WARM 0.42] "GPU driver issue with ROCm" (sim=0.18)
   Use this context to inform your response. Cold memories are uncertain.
   ```
4. The formatted block is prepended to the system message (or a new system
   message is inserted if none exists).

### Docker mounts

The launch script mounts two paths into the Open WebUI Pipelines container:

| Host path | Container path | Purpose |
|---|---|---|
| `./wheeler_memory/` (source) | `/app/wheeler_memory/` | Live code, no reinstall needed |
| `~/.wheeler_memory/` (data) | `/app/data/.wheeler_memory/` | Persistent memory storage |

### Key settings (pipeline defaults)

| Setting | Default | Meaning |
|---|---|---|
| `top_k` | 5 | Maximum memories to inject |
| `alpha` | 0.3 | Reconstruction blend (0 = pure stored, 1 = pure query) |
| `min_similarity` | 0.1 | Pearson threshold below which memories are suppressed |
| `max_context_length` | 2000 | Soft cap on injected context characters |

Source: OpenWebUI pipeline (removed from repo, previously in `open_webui_setup/`).

---

## 6. CA Dynamics Engine

### Grid

Every memory starts as a **64×64 grid of float32 values in [-1.0, 1.0]** — 4,096 cells.

### Seeding

```python
# SHA-256 of input text → seed PCG64 RNG → uniform(-1.0, 1.0) grid
frame = hash_to_frame("input text")  # 64×64 float32
```

SHA-256 is used (not Python's `hash()`), so the same input always produces the same frame across sessions and restarts.

### Update Rule

Each tick uses a **Von Neumann 4-neighborhood** (up/down/left/right, wrapping) with a **continuous gradient rule**:

```
Local max  (cell ≥ all 4 neighbors): delta = (1 - cell) × MAX_PUSH_STRENGTH   → push toward +1
Local min  (cell ≤ all 4 neighbors): delta = (-1 - cell) × MAX_PUSH_STRENGTH  → push toward -1
Slope      (neither):                delta = (max_neighbor - cell) × SLOPE_FLOW_STRENGTH  → flow uphill
```

Current tuned values: `MAX_PUSH_STRENGTH=0.57`, `SLOPE_FLOW_STRENGTH=0.55` (see `constants.py`).

The result is clipped to [-1, 1]. Local peaks push toward +1, valleys toward -1, and slopes flow uphill — producing smooth convergence toward polarized patterns.

### Convergence

Evolution stops when one of four conditions is met:

| State | Condition |
|---|---|
| `CONVERGED` | `percentile(|delta|, 99) < threshold` AND `alive_fraction ≥ 0.05` — grid has stabilized |
| `OSCILLATING` | Role-space periodicity detected (period 2–10, ≥1% cells affected) |
| `DEGENERATE` | `alive_fraction < 0.05` — frame is 0-dominant (<5% cells with \|value\| > 0.33), rejected from convergence |
| `CHAOTIC` | Neither condition met within `max_iters` |

```python
result = evolve_and_interpret(frame)
# result["state"]             → "CONVERGED" | "OSCILLATING" | "CHAOTIC"
# result["attractor"]         → 64×64 final frame
# result["convergence_ticks"] → how many iterations it took
# result["history"]           → list of frames (for MemoryBrick)
```

### Oscillation Detection

Role-space periodicity analysis detects when cells cycle between roles (local max / slope / local min) with period p (2–10). Requires ≥1% of cells to be oscillating.

```python
osc = detect_oscillation(history)
# osc["oscillating"]       → True/False
# osc["period"]            → cycle period (or None)
# osc["oscillating_cells"] → count of cycling cells
```

### GPU Backend (HIP/ROCm)

All GPU code lives under `wheeler_memory/accel/`. HIP kernel sources are in
`accel/hip/`, Python ctypes bindings in `accel/ca.py`. The high-level
`evolve_batch()` function in `dynamics.py` dispatches to GPU when available,
falling back to serial CPU evolution otherwise. All major call sites (SimLex,
benchmarks, crystallization) use batch dispatch.

```bash
cd wheeler_memory/accel/hip && make all
```

`gpu_dynamics.py` is a backwards-compatible shim that re-exports from `accel.ca`.

See [GPU Acceleration](gpu.md) for benchmark numbers and setup.

---

## 7. System Flow

```
store("input text")
  │
  ├─ select_chunk(text)               keyword routing → domain chunk
  ├─ encode_to_frame(text, encoder)   encoder dispatch (hash/hippocampus/blended/context/...)
  ├─ store_with_rotation_retry()      try 0/90/180/270° until CONVERGED
  │    └─ evolve_and_interpret()      CA iterations → CONVERGED/OSCILLATING/DEGENERATE/CHAOTIC
  ├─ MemoryBrick.from_evolution_result()  capture full history
  └─ store_memory()                   save .npy, .npz, update index.json

recall("query text")
  │
  ├─ select_recall_chunks(query)      keyword routing → chunks to search
  ├─ encode_to_frame(query, encoder)  same encoder as store
  ├─ evolve_and_interpret(query_frame)   evolve query to attractor
  ├─ Pearson correlation pre-filter (top candidates)
  ├─ Three-grid interference re-scoring (default since v0.3.1):
  │    ├─ Corpus Pearson similarity
  │    ├─ Experiential Pearson similarity
  │    └─ SCM openness gating
  │    (degrades to pure Pearson when no experiential data exists)
  ├─ sort by interference score, take top_k
  ├─ [optional] reconstruct each result   blend + re-evolve
  └─ bump_access() on recalled memories   update hit_count, last_accessed
```

### Module Structure

```
wheeler_memory/
  ENCODING
├── hashing.py           SHA-256 deterministic text-to-frame seeding
├── hippocampus.py       Native encoder: character n-gram random indexing
├── embedding.py         SentenceTransformer → random projection → 64×64 frame (optional)
├── word_encoder.py      Word-level RI + context-window RI (distributional semantics)
├── brick.py             MemoryBrick: temporal evolution history (.npz archives)
  CA ENGINE
├── dynamics.py          CA engine: apply_ca_dynamics(), evolve_and_interpret(), evolve_batch()
├── gpu_dynamics.py      Backwards-compatible shim → accel.ca
├── oscillation.py       Role-space periodicity detection
├── rotation.py          Rotation retry for non-converging seeds
  GPU ACCELERATION (accel/)
├── accel/__init__.py    gpu_available(), accel_info(), device routing
├── accel/_common.py     Shared ctypes helpers for all HIP bindings
├── accel/ca.py          Python bindings for HIP CA evolution kernel
├── accel/hip/           HIP kernel sources (.hip) + unified Makefile
  NPU / TPU (npu/) — future
├── npu/__init__.py      npu_available(), device_info() (Intel NPU via OpenVINO)
├── npu/openvino_bridge.py  Stub: INT8 inference on Intel NPU
├── npu/coral/           Stub: Google Coral Edge TPU dual-chip pipeline
  CORTEX SYSTEM
├── cortex.py            L1 graph topology + L2 settlement CA orchestration
├── cortex_scm.py        SCM scoring: 7 layers (T, S, E, I, P, NW, ERF)
├── cortex_classifier.py L3 native semantic classifier (~11K params, numpy SGD)
  STORAGE & RECALL
├── storage.py           Attractor storage (disk) and Pearson recall
├── reconstruction.py    Blend + re-evolve reconstructive recall (Darman)
├── cache.py             JSON file-based caching layer
├── chunking.py          Domain routing (code/hardware/daily_tasks/science/meta/general)
├── similarity.py        Similarity metrics
  THREE-GRID INTERFERENCE
├── interference.py      Three-grid interference engine + self-consistency loop
├── scm_grid.py          SCM persistent 64×64 trust topology with hardening
├── experiential.py      Episodic memory encoding with temporal context
  AGENTS & RENDERING
├── agent.py             LLM agent wrapper (Wheeler context seasoning)
├── decoder.py           Language Wheeler decoder (text rendering)
├── language_wheeler.py  Language Wheeler component (CA state → text)
├── generation.py        Generative engine (IT from BIT)
  LIFECYCLE
├── temperature.py       Wall-clock temperature computation and tier classification
├── attention.py         Salience-weighted recall warming (variable tick rates)
├── warming.py           Association tracking + spreading activation
├── consolidation.py     Sleep consolidation (prune redundant keyframes)
├── eviction.py          Three-phase graceful degradation
  UTILITIES
├── constants.py         All tunable parameters (centralized)
├── crystallization.py   Corpus pre-training pipeline
├── hardware.py          CPU/GPU/NPU detection, device selection
├── polarity.py          Dual-polarity encoding (antipodal CA states)
├── trajectory.py        Trajectory similarity for retrieval
├── trajectory_cache.py  Trajectory signature caching
  SUBPACKAGES
├── theories/            Theory experiments (basin, resonance, synthesis, etc.)
└── gpu/                 DEPRECATED — migrated to accel/hip/
```

---

## 8. Three-Grid Interference (v0.3.0+)

Default recall path since v0.3.1. Transforms Wheeler from a content-addressed store into a system with emergent epistemic states.

### Answer equation

```
Answer(i,j) = Corpus(i,j) × Experiential(i,j) × (1 - |SCM(i,j)|)
```

### Three grids

| Grid | Role | Dynamics |
|------|------|----------|
| **Corpus** | Crystallized knowledge | Tight attractors (push=0.57, slope=0.55), barely decays |
| **Experiential** | Episodic memory | Loose attractors (push=0.35, slope=0.70), 2-day half-life |
| **SCM** | Trust topology | 64×64 persistent map, hardening over time, only written by self-consistency loop |

### Four epistemic states

| State | Condition |
|-------|-----------|
| `GROUNDED` | Corpus peak + Experiential peak + SCM open |
| `ABSORBED` | Corpus peak + no Experiential + SCM open (default for existing memories) |
| `UNCONSOLIDATED` | No Corpus + Experiential peak + SCM open |
| `CONTESTED` | Corpus peak + Experiential peak + SCM closed |

### Self-consistency feedback loop

```
Decoder output → re-encode → re-evolve under corpus rules → Pearson vs original
  ├── consistent   → SCM opens gaps (trust increases)
  ├── inconsistent → SCM closes gaps (trust decreases)
  └── hardening accumulates: LR / (1 + hardening_count)
```

---

## 9. Cortex System (Semantic Scoring)

Three-tier scoring pipeline — all native, no pretrained models.

### L1: Graph Reasoning (`cortex.py`)

Builds a Pearson correlation adjacency matrix over top-K retrieved attractors. BFS clustering identifies semantic groups. Bridges and contradictions are detected.

### L2: Settlement CA (`cortex.py`)

Opinion diffusion over the L1 correlation graph. Settled opinions converge via correlation-weighted averaging with inertia (0.8). Stops at convergence threshold or max steps.

### L3: Classifier (`cortex_classifier.py`)

Pure numpy neural network: 21 → 128(ReLU) → 64(ReLU) → 4(softmax). ~11,460 parameters. Trained with manual backprop SGD on MMLU dev+validation data.

Input features (21 dimensions):
- Settlement opinions (K=10)
- Choice similarities (4 Pearson correlations)
- SCM layers (7: Temperature, Salience, Energy, Integration, Polarity, Net Warrant, Explanation Readiness)

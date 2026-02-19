# Architecture

## Overview

```
Input Text
    ↓
Encode  ─── SHA-256 hash (default, exact match)
        └── Sentence embedding (--embed, semantic match)
    ↓
64×64 seed frame (float32, values in [-1, +1])
    ↓
3-State CA Evolution ──── rotation retry (0°/90°/180°/270°)
    ├── CONVERGED      → store attractor + brick
    ├── OSCILLATING    → epistemic uncertainty detected
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
    state:             str                 # CONVERGED | OSCILLATING | CHAOTIC
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

Source: `open_webui_setup/pipelines/wheeler_memory_pipeline.py`.

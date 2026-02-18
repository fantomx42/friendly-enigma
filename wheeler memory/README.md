# Wheeler Memory

A cellular automata-based associative memory system where meaning is what survives symbolic pressure.

Instead of "save text, search later" (traditional RAG), Wheeler Memory evolves input through 3-state cellular automata dynamics until it converges to a stable attractor — a physical pattern that *is* the memory. Recall works by Pearson correlation between query attractors and stored attractors.

## How It Works

```
Input Text
    ↓
SHA-256 → 64×64 seed frame (values in [-1, +1])
    ↓
3-State CA Evolution (iterate until convergence)
    ├→ CONVERGED   → Store attractor + brick
    ├→ OSCILLATING → Epistemic uncertainty detected
    └→ CHAOTIC     → Input needs rephrasing
```

**The three cell roles** (von Neumann neighborhood):

| Role | Condition | Update | Meaning |
|------|-----------|--------|---------|
| Local Maximum | `cell >= all 4 neighbors` | Push toward +1 (0.35) | Attractor basin center |
| Slope | Neither max nor min | Flow toward max neighbor (0.20) | Transitional |
| Local Minimum | `cell <= all 4 neighbors` | Push toward -1 (0.35) | Repellor / valley |

Convergence typically happens in 39-49 ticks (~3ms on CPU). The result is a unique QR-code-like binary pattern per input.

## Chunked Memory Architecture

Memories are routed to domain-specific chunks via keyword matching — inspired by how different cortical regions handle different domains:

```
~/.wheeler_memory/
  chunks/
    code/           attractors/ bricks/ index.json metadata.json
    hardware/       ...
    daily_tasks/    ...
    science/        ...
    meta/           ...
    general/        ...  (default catch-all)
  rotation_stats.json
```

Routing is automatic: `"fix the python debug error"` → **code**, `"buy groceries and schedule dentist"` → **daily_tasks**, `"something random"` → **general**.

On recall, the system searches across all matching chunks plus general, merging results by similarity.

## The Brick

Every memory stores its full temporal evolution as a "brick" — a 3D structure (width × height × ticks) that you can scrub through like a video timeline:

```
Tick 0:  Noisy seed     → ████░░██░░██
Tick 20: Forming        → █░░█░░░█░░░█
Tick 43: Converged      → █░░█░░░█░░░█  (frozen attractor)
```

This gives you complete debuggability — you can see exactly how and when each memory formed.

## Rotation Retry

When the initial CA evolution lands in a bad attractor basin, the system rotates the seed frame (90°/180°/270°) and retries. Rotation changes the neighbor topology, creating a different dynamical trajectory from the same information. Per-angle success statistics are tracked for future optimization.

## Install

```bash
git clone https://github.com/fantomx42/friendly-enigma.git
cd "wheeler memory"
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.11+ with numpy, scipy, and matplotlib.

## CLI Tools

### Store a memory

```bash
wheeler-store "fix the python debug error"
# Chunk:    code (auto)
# State:    CONVERGED
# Ticks:    43
# Rotation: 0° (attempt 1)
# Time:     0.003s
# Memory stored successfully.

wheeler-store --chunk hardware "solder the GPIO header"   # explicit chunk
echo "piped input" | wheeler-store -                       # stdin
```

### Recall memories

```bash
wheeler-recall "python bug"
# Rank  Similarity  Chunk        State        Ticks  Text
# ----------------------------------------------------------------------------------
# 1        0.0145  code         CONVERGED       43  fix the python debug error
# ...

wheeler-recall --chunk code "debug error"   # search specific chunk
wheeler-recall --top-k 10 "something"       # more results
```

### Scrub a brick timeline

```bash
wheeler-scrub --text "fix the python debug error"           # find by text
wheeler-scrub --text "solder header" --chunk hardware       # in specific chunk
wheeler-scrub path/to/brick.npz                              # direct path
```

Opens an interactive matplotlib viewer with a tick slider.

### Diversity report

```bash
wheeler-diversity
# Evolves 20 diverse test inputs, computes pairwise correlations.
# PASS when avg correlation < 0.5 and max < 0.85.
```

## Temperature Dynamics

Every memory has a **temperature** reflecting how recently and frequently it's been accessed. Temperature is computed lazily on recall and listing — no background processes.

```
temp = base_from_hits × decay_from_time

base_from_hits = min(1.0, 0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (-days_since_last_access / 7.0)
```

**Tiers:**

| Tier | Range | Meaning |
|------|-------|---------|
| hot  | >= 0.6 | Frequently accessed, recent |
| warm | >= 0.3 | Default for new or moderately accessed |
| cold | < 0.3  | Stale, candidate for archival |

New memories start at temperature 0.3 (warm). Each recall bumps hit_count and resets last_accessed, keeping active memories hot. Unused memories decay with a 7-day half-life.

### Inspect temperatures

```bash
wheeler-temps                     # all memories
wheeler-temps --chunk code        # specific chunk
wheeler-temps --tier hot          # filter by tier
wheeler-temps --sort hits         # sort by hit count
```

### Temperature-boosted recall

```bash
wheeler-recall --temperature-boost 0.1 "python bug"
```

When `--temperature-boost` is nonzero, ranking uses `similarity + boost × temperature` — hotter memories get a slight ranking bonus. Default boost is 0.0 (pure similarity ranking, identical to previous behavior).

## Python API

```python
from wheeler_memory import (
    store_with_rotation_retry,
    recall_memory,
    list_memories,
    select_chunk,
    select_recall_chunks,
)

# Store
result = store_with_rotation_retry("fix the auth bug", chunk="code")
# result["state"] == "CONVERGED"

# Recall
matches = recall_memory("authentication error", top_k=5)
for m in matches:
    print(m["text"], m["similarity"], m["chunk"])

# Chunk routing
select_chunk("debug the python script")          # → "code"
select_chunk("buy groceries")                     # → "daily_tasks"
select_recall_chunks("quantum physics equation")  # → ["science", "general"]

# List all memories
for mem in list_memories():
    print(mem["chunk"], mem["text"])
```

## Theoretical Foundation

Wheeler Memory implements the **Symbolic Collapse Model (SCM)**:

1. **Meaning is what survives symbolic pressure** — stable attractors represent survived concepts
2. **Memory and learning are the same process** — each interaction reshapes the landscape
3. **Uncertainty is observable in dynamics** — convergence = clarity, oscillation = ambiguity, chaos = contradiction
4. **Time is intrinsic to memory** — convergence speed reflects concept complexity; full history is preserved

Named after John Archibald Wheeler's "It from Bit" — information emerges from physical-like dynamics.

## Project Structure

```
wheeler_memory/
    __init__.py          # Public API exports
    brick.py             # MemoryBrick temporal structure
    chunking.py          # Chunk routing & management
    dynamics.py          # 3-state CA engine
    hashing.py           # SHA-256 text→frame
    oscillation.py       # Epistemic uncertainty detection
    rotation.py          # Rotation retry with learning
    storage.py           # Store/recall/list with chunked storage
    temperature.py       # Temperature dynamics (hot/warm/cold)
scripts/
    wheeler_store.py     # CLI: store
    wheeler_recall.py    # CLI: recall
    wheeler_temps.py     # CLI: temperature inspector
    scrub_brick.py       # CLI: brick timeline viewer
    test_diversity.py    # CLI: attractor diversity report
```

## Future Work

- **GPU acceleration** — HIP kernels for parallel chunk evolution on AMD GPUs (ROCm)
- **Embedding-based routing** — semantic similarity for chunk selection instead of keywords
- **Reconstructive recall** — memories influenced by current context (Darman architecture)

## License

MIT

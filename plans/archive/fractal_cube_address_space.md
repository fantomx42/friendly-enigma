# Fractal Cube Address Space

> Discussed 2026-03-17. Theory + implementation plan. Not yet built.

---

## The Concept

Each Wheeler Memory cell is conceptually a cube. Each face of that cube is itself a cube —
recursively, all the way down. Navigation is address traversal: pick a face, go deeper.

**Address length = depth of understanding.**
- Short address: surface familiarity
- Long address: deep comprehension
- The coordinate chain literally encodes how well the system knows something

**The ternary → 6-face mapping is canonical.**
The CA state space (-1, 0, +1) maps onto cube geometry: 3 axes × 2 directions = 6 faces.
This is not metaphor — the cell value space IS the face space.

| Face | Axis | Direction | CA State |
|------|------|-----------|----------|
| +X | X | positive | +1 (peak) |
| -X | X | negative | -1 (valley) |
| +Y | Y | positive | +1 (peak) |
| -Y | Y | negative | -1 (valley) |
| +Z | Z | positive | 0 (slope toward +) |
| -Z | Z | negative | 0 (slope toward -) |

(Exact face→state mapping TBD — the point is ternary maps cleanly to 6 faces.)

---

## The Portal Mechanic

Normally a recursive address space bottoms out — you hit the floor and stop. The insight:
**the floor is a door.**

When you reach a terminal attractor:
1. Compute its SHA hash (the attractor's binary representation → SHA-256)
2. That hash simultaneously serves as:
   - **Terminal address coordinate** — where you are in the address chain
   - **Reconstruction seed** — how to rebuild this attractor (`hash_to_frame` already uses SHA)
   - **New `cube³:0` origin** — the floor becomes a new top-level cube at a finer scale

The address chain is theoretically infinite. Meaning is bottomless.

**Why this is nearly free**: `text_to_hex` and `hash_to_frame` in `hashing.py` already implement
this implicitly. The chain is just recursive composition of existing functions:

```
text
  → text_to_hex(text)              # SHA-256 hex key (already exists)
  → hash_to_frame(text)            # seeded RNG frame (already exists)
  → evolve_and_interpret(frame)    # CA evolution (already exists)
  → sha256(attractor.tobytes())    # portal hash (one new line)
  → hash_to_frame(portal_hash)     # next level frame (already exists)
  → evolve_and_interpret(frame)    # deeper attractor (already exists)
  → ...
```

The portal is just `sha256(attractor.tobytes())` used as the next seed. Everything else exists.

---

## What Exists in the Codebase

### Good news (reuse directly)
- `hashing.py`: `text_to_hex()`, `hash_to_frame()` — the whole chain is composable now
- `dynamics.py`: `evolve_and_interpret()` — CA evolution at each depth level
- `storage.py`: `recall_memory()` — for wiring face-navigation to semantic recall
- `wheeler_3d_viewer/`: Three.js + FastAPI + WebSocket — existing rendering infrastructure
  - Already renders 64×64 cubes with height mapping
  - Already streams CA evolution frames
  - WebSocket protocol already established

### Gap
- No hierarchical address concept in storage or index
- No face-navigation concept
- No recursive depth traversal endpoint
- No fractal cube frontend (existing viewer is flat grid, not recursive)

---

## Parallelisation via Cube Faces

**The key insight**: the 6 faces are independent by construction. All 6 children of any node share
no dependencies — they can be computed simultaneously.

At each node, instead of picking one face and going linear, expand all 6 at once:

```
attractor
  → portal_hash = sha256(attractor.tobytes())
  → child[0] = evolve(hash_to_frame(portal_hash + b'\x00'))   ─┐
  → child[1] = evolve(hash_to_frame(portal_hash + b'\x01'))    │
  → child[2] = evolve(hash_to_frame(portal_hash + b'\x02'))    │  all parallel
  → child[3] = evolve(hash_to_frame(portal_hash + b'\x03'))    │
  → child[4] = evolve(hash_to_frame(portal_hash + b'\x04'))    │
  → child[5] = evolve(hash_to_frame(portal_hash + b'\x05'))   ─┘
```

At depth N: **6^N independent CA evolutions**, all computable in parallel at that level.
The GPU batch evolution already exists — just feed it 6 frames instead of 1.

**Storage consequence**: instead of storing one attractor per memory, store the **full cube** —
all 6 face-attractors. Recall matches against 6 signals per memory instead of 1. Richer, better
discrimination, and the 6 evolutions cost little extra on GPU.

**The branching factor of 6 is not a design choice — it falls out of the ternary geometry.**
3 axes × 2 directions = 6 faces. You don't pick 6, the math does.

---

## Implementation Plan

### Difficulty: Medium
Backend is low complexity (composing existing functions). Frontend is the hard part.

---

### Core math: `attractor_cube(seed) -> dict[int, np.ndarray]`

New function in `wheeler_memory/hashing.py`:

```python
def attractor_portal_hash(attractor: np.ndarray) -> str:
    """SHA-256 of attractor bytes — the portal to the next depth level."""
    return hashlib.sha256(attractor.tobytes()).hexdigest()

def expand_cube(portal_hash: str) -> dict[int, str]:
    """
    Deterministic per-face child seeds from a portal hash.
    6 independent seeds, one per face — ready for parallel evolution.
    """
    return {
        face: hashlib.sha256(portal_hash.encode() + bytes([face])).hexdigest()
        for face in range(6)
    }
```

Then batch-evolve all 6 children simultaneously:

```python
def evolve_cube(portal_hash: str) -> dict[int, dict]:
    """Evolve all 6 face-children in parallel. Returns face_index → result."""
    face_seeds = expand_cube(portal_hash)
    frames = [hash_to_frame(seed) for seed in face_seeds.values()]
    # Batch GPU evolution (6 frames at once)
    results = [evolve_and_interpret(f) for f in frames]  # parallelise here
    return {face: result for face, result in zip(range(6), results)}
```

---

### Backend additions

**New endpoint**: `POST /api/traverse` body: `{seed: TEXT, address: [0,3,1,5,2]}`

```python
def traverse_address(seed_text: str, face_sequence: list[int]) -> dict:
    current_seed = seed_text
    chain = []

    for face_index in face_sequence:
        frame = hash_to_frame(current_seed)
        result = evolve_and_interpret(frame)
        attractor = result["attractor"]
        portal_hash = attractor_portal_hash(attractor)
        face_seeds = expand_cube(portal_hash)

        chain.append({
            "seed": current_seed,
            "face_taken": face_index,
            "portal_hash": portal_hash,
            "state": result["state"],
            "ticks": result["convergence_ticks"],
        })

        current_seed = face_seeds[face_index]

    return {"chain": chain, "depth": len(face_sequence), "terminal_seed": current_seed}
```

**Also**: `/api/expand` — returns all 6 children of a node (for pre-fetching next level in parallel).

**Semantic wiring**: at each depth, `recall_memory(current_seed, top_k=3)` surfaces which stored
memories resonate with this address — traversal becomes semantically meaningful, not just geometric.

---

### Frontend

The existing Three.js viewer renders a flat 64×64 grid. The fractal cube explorer needs:

1. **Central cube** rendered as a single large cube with 6 clickable faces
2. **Click a face** → animate into that face → render the next-level attractor at that depth
3. **Address bar** showing current address chain (e.g., `cube³:0 → face:3 → face:1`)
4. **Recall panel** showing semantically resonant memories at current depth
5. **Back navigation** — address is just the chain, pop last face to go up

Two options:
- **Extend existing Three.js viewer** (lower effort, consistent tech)
- **React + Three.js (react-three-fiber)** (better component model for recursive UI)

The React prototype mentioned in discussion doesn't exist in the repo — needs to be built.
Recommended: extend the existing Three.js viewer as `wheeler_3d_viewer/fractal.html` + new backend
routes in `app.py` before committing to a full React rewrite.

---

### Files to create/modify

| File | Change |
|------|--------|
| `wheeler_memory/hashing.py` | Add `attractor_portal_hash()`, `expand_cube()`, `evolve_cube()` |
| `wheeler_3d_viewer/app.py` | Add `/api/traverse`, `/api/expand` endpoints + WebSocket stream |
| `wheeler_3d_viewer/fractal.html` | New fractal cube frontend |
| `wheeler_3d_viewer/fractal.js` | Face navigation, parallel pre-fetch of all 6 children |

Core `wheeler_memory/` changes are minimal — just 3 functions in `hashing.py`.

---

## The Time Axis — Don't Throw It Away

The CA evolution is inherently 3D: `(Time, X, Y)`. Every tick stacked like a deck of cards —
noise at the bottom, stable attractor at the top. The system currently stores only that top slice
and discards the rest.

**The discarded dimension is information-rich:**

| Signal | Source | What it encodes |
|--------|--------|-----------------|
| Convergence speed | `convergence_ticks` (already in index) | Familiarity — fast = strong existing basin, slow = novel or sitting near a boundary |
| Trajectory shape | brick history frames (already saved) | Smooth spiral = clean basin membership. Oscillating near-miss = input between two competing attractors |
| Entropy curve | per-tick entropy across history frames | Rate of internal structure formation — encodes semantic complexity of input |

**Current state:**
- Full trajectory: saved in bricks ✓ — never used ✗
- Convergence ticks: saved in index ✓ — never used as familiarity signal ✗
- Entropy curve: not computed, not stored ✗

**The familiarity signal is free.** `convergence_ticks` already exists. Fast convergence = input
landed cleanly in a known basin. Slow convergence / late oscillation = basin boundary = ambiguous.
Richer confidence signal than Pearson similarity alone, costs nothing extra.

**Connection to fractal cube:** trajectory shape tells you which face you're near. Smooth trajectory
= deep in one basin's gravity well. Oscillating trajectory = near a face boundary between two basins.
The trajectory literally locates you in the address space.

**Connection to meta-rules / salience:** skipping ticks is already compressing the time axis in
real time — "I know what the next N layers look like, jump to N+1." The time dimension was being
exploited implicitly. Making it explicit just surfaces what's already happening.

### What to build

1. **Familiarity signal** (nearly free): use `convergence_ticks` from index as a confidence
   modifier in recall — fast convergence boosts score, slow convergence flags ambiguity.

2. **Trajectory entropy curve**: on store, compute per-tick entropy across brick history frames,
   store as a compact summary (e.g., 10-point downsampled curve) in the index metadata.

3. **Shape classifier**: detect trajectory type from history —
   - `SMOOTH` — monotone entropy decrease, clean basin
   - `OSCILLATING` — late-stage oscillation, basin boundary
   - `CHAOTIC` — already detected, maps to 0-dominant / uncertainty state

4. **Secondary recall signal**: weight recall results by trajectory similarity — queries that
   converge smoothly should prefer memories that also converged smoothly, and vice versa.

---

## Open Questions

1. **Face → portal mapping**: Does each face lead to a different sub-cube, or does each level
   have one portal and the face just determines which "region" of it you enter? Simplest version:
   `portal_hash + str(face_index)` → deterministic per-face child address.

2. **0-face**: With the ternary fix, 0 now has semantic meaning (uncertainty). Does a 0-dominant
   attractor produce a "foggy" portal — one that's harder to navigate, requires more depth?

3. **Address storage**: Should traversed addresses be stored in the index so the system can
   remember "I've been here before"? Adds `visited_addresses` to index metadata.

4. **Semantic wiring precision**: At each depth, recall against the portal hash (deterministic) or
   against the text of the attractor's top recalled memory (drifts with the landscape)?

---

## Notes
- The portal mechanic validates "It from Bit" architecturally: the hash IS the address IS the seed.
  There's no separation between location and content. The coordinate is the thing.
- Address length as comprehension depth is testable: MMLU questions the system answers correctly
  should require fewer address steps than ones it fails. This is a novel evaluation metric.
- Long-term: daydreaming through the fractal space (random face walks) is a natural extension
  of the daydream mechanism — the system explores its own address space while idle.
- The full-cube storage model (6 attractors per memory) is a direct upgrade to the current
  single-attractor model. Recall gets 6x the signal, parallelisation is free on GPU, and the
  address space structure emerges from the storage format itself.

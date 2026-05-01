# Ternary Dynamics Fix + Daydreaming

> Discussed 2026-03-17. Not yet implemented.

---

## 1. Dynamics Bug — 0 is not a stable fixed point

**File**: `wheeler_memory/dynamics.py`, `apply_ca_dynamics()`

When a cell and all 4 neighbours are equal (flat plateau), `is_max` and `is_min` are both True.
The `np.where` chain applies `is_min` last and overwrites `is_max`, pushing the cell toward -1.
A uniform 0 region should be stable but isn't.

**Fix** — introduce `is_flat`:

```python
is_flat = is_max & is_min  # all neighbours equal — no movement

delta = np.zeros_like(frame)
delta = np.where(is_max & ~is_flat, (1 - frame) * MAX_PUSH_STRENGTH, delta)
delta = np.where(is_min & ~is_flat, (-1 - frame) * MAX_PUSH_STRENGTH, delta)
delta = np.where(~is_max & ~is_min, (max_neighbor - frame) * SLOPE_FLOW_STRENGTH, delta)
# is_flat cells: delta stays 0
```

Three stable fixed points with semantic meaning:

| State | Meaning |
|-------|---------|
| +1 dominant | Strong positive knowledge — confident recall |
| -1 dominant | Strong negative knowledge — confident rejection / avoidance |
| 0 dominant | Epistemic uncertainty — the system doesn't know; needs more information or re-evaluation |

0-dominance is the CA's native confidence signal. A query that evolves toward a 0-dominant attractor
means the system has no stable opinion — no separate heuristic needed, the grid geometry IS the signal.

With daydreaming, 0-dominant attractors are the most susceptible to drift: cold, unstable, easy to
perturb into a ±1 basin when new relevant information arrives. Not-knowing is an unstable state that
resolves naturally when the system learns something.

This eventually replaces / augments the Pearson similarity floor in `decoder.py` (currently hardcoded
at 0.18) — check 0-dominance of the attractor directly instead.

**Tests to add** in `tests/test_dynamics.py`:
- `test_flat_zero_plateau_is_stable` — `np.zeros((64,64))` → after one tick, still all zeros
- `test_flat_positive_plateau_is_stable` — `np.ones((64,64))` → unchanged
- `test_flat_negative_plateau_is_stable` — `-np.ones((64,64))` → unchanged

**Visual intent**: three-color lava lamp behaviour — +1/0/-1 blobs competing at their boundaries,
flowing and settling into a stable three-color mosaic. Currently the gif shows blue (0) as pure
transit; after the fix, blue becomes a genuine third phase that holds ground.

**Note**: this will slightly change convergence behaviour for any attractor that previously collapsed
a 0-region to -1. Run MMLU benchmark after to check impact.

---

## 2. Daydreaming — spontaneous replay with natural erosion

**The gap**: stored attractors are immutable after being written. Nothing causes cold/old memories to
drift, crowded basins to erode, or the landscape to self-organise over time.

**Mechanism**: spontaneous replay weighted toward cold memories.
- Pick a stored attractor (cold memories sampled more often)
- Add small Gaussian perturbation
- Re-evolve through CA
- Hot memories re-converge to the same attractor (stable, reinforced)
- Cold memories wander to a nearby basin (drift = natural wear)
- If the drifted attractor lands close to another existing one → evict the colder duplicate

This is two ideas combined: spontaneous replay (2) is the prerequisite for idle CA wandering (1).

### Files to create
- `wheeler_memory/daydream.py` — core logic
- `scripts/wheeler_daydream.py` — CLI

### Files to modify
- `wheeler_memory/constants.py` — add daydream constants
- `pyproject.toml` — register `wheeler-daydream` entry point

### constants.py additions

```python
# Daydream
DAYDREAM_PERTURBATION_SCALE: float = 0.05   # std of Gaussian noise added before re-evolution
DAYDREAM_DRIFT_THRESHOLD: float = 0.02       # min mean absolute drift to update stored attractor
DAYDREAM_EROSION_THRESHOLD: float = 0.90     # Pearson similarity above which two attractors merge
DAYDREAM_COLD_BIAS: float = 2.0              # exponent for inverse-temperature sampling weight
```

### daydream.py — algorithm

`run_daydream(chunk_dir, n_steps=50, rng_seed=None)`

Per step:
1. Load index → all hex_keys + temperatures
2. Weights: `w = (1 - temperature) ** DAYDREAM_COLD_BIAS`
3. Sample one attractor key by weight
4. Load `.npy`
5. Perturb: `np.clip(attractor + rng.normal(0, PERTURBATION_SCALE, shape), -1, 1)`
6. `result = evolve_and_interpret(perturbed)`
7. `drift = mean(abs(result['attractor'] - original))`
8. If `drift > DAYDREAM_DRIFT_THRESHOLD`:
   - Overwrite stored `.npy`
   - Increment `daydream_count`, set `last_daydream` in index
9. Pearson scan vs all other attractors
10. If any correlation > `DAYDREAM_EROSION_THRESHOLD`: evict lower-temperature duplicate

Reuse:
- `evolve_and_interpret()` — `dynamics.py`
- Index I/O — `storage.py`
- Eviction — `eviction.py`
- Temperature values — already in index metadata

### CLI

```
wheeler-daydream [--chunk CHUNK] [--steps N] [--seed SEED] [--verbose]
```

pyproject.toml registration — match existing pattern (see other `scripts/wheeler_*.py` entries).

### Future
Integrate into `wheeler-sleep` so sleep = consolidate + daydream in one pass.

---

## Verification

```bash
pytest tests/test_dynamics.py -v        # new plateau tests pass
pytest tests/ -v                        # existing suite unbroken

wheeler-store --embed "test concept alpha"
wheeler-store --embed "test concept beta"
wheeler-daydream --steps 10 --verbose   # shows drift per step, any merges
```

Erosion test: store two near-identical texts, run daydream, verify the colder one is evicted.

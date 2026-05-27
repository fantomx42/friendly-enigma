# Per-Cell Crystallization Probe — 2026-05-22

**Status:** Negative result. Reverted to baseline same day.
**Branch:** `main` (probe applied, swept, reverted)
**Companion artifacts:** [assets/per-cell-crystallization-probe-2026-05-22/](assets/per-cell-crystallization-probe-2026-05-22/) — probe script + visual outputs.

## Why this experiment

Working assumption being tested: that adding per-cell hardness to the CA loop — cells harden when locally settled, erode with T, soft-freeze further updates — would produce **web-like clumps of locally-coherent state on a partially-evolving substrate**, instead of a single global attractor.

This is closer to **excitable-media** dynamics than the current **Lyapunov relaxation** model. The current code (and [CANON.md](../../CANON.md) §1.1, §3.6.3; [architecture.md](../architecture.md) §6) all describe convergence as a single global p99(|Δ|) + alive-fraction stop on the whole 64×64 grid. Every cell updates every tick until the entire frame halts. The only per-cell hit/erode primitive in the repo (`SCMGrid.hardening`, `wheeler_memory/scm_grid.py:41-147`) lives on a separate non-CA trust grid that is fed by the self-consistency loop, not the CA tick loop.

So the question was: *can per-cell hardening, added inside the existing CA loop, produce spatially heterogeneous clumps the way excitable media do — or does the CA's underlying dynamics make that impossible?*

## What was changed

Three minimal-surface-area changes, all gated behind a `CA_CELL_HARDENING_ENABLED` flag defaulted to `False`. The off-by-default path is byte-identical to baseline (verified: full pytest suite, 746 passed / 1 skipped / 28 deselected, unchanged).

### `wheeler_memory/constants.py` — appended 6 probe constants

```python
CA_CELL_HARDENING_ENABLED = False  # master flag; False = byte-identical to baseline
CA_HARDEN_RATE = 0.05              # hardness gain per tick on cells with |delta| < SETTLE_EPSILON
CA_EROSION_RATE = 0.02             # multiplicative decay per tick on non-settled cells
CA_SETTLE_EPSILON = 1e-3           # |delta_ij| threshold for "settled this tick"
CA_FROZEN_THRESHOLD = 0.9          # hardness threshold for the frozen mask (measurement only)
CA_FREEZE_FRACTION_STOP = 0.80     # early-exit when this fraction of cells exceeds CA_FROZEN_THRESHOLD
```

### `wheeler_memory/dynamics.py` — hardness plumbed through `evolve_and_interpret`

- Import `constants` as `_constants` for runtime-mutable probe params (the `from .constants import X` pattern caches the value at import time, blocking probe-time mutation).
- Allocate `hardness = np.zeros_like(frame)` per call (transient, no persistence).
- Per-tick: `frame = frame_old + (frame_new - frame_old) * (1 - hardness)` — soft freeze. Then bump hardness on cells with `|raw_delta| < SETTLE_EPSILON`; erode multiplicatively otherwise; clip `[0, 1]`.
- Added a freeze-fraction early-exit alongside the existing p99/alive rule.
- All four return paths (CONVERGED via p99, CONVERGED via freeze, OSCILLATING, CHAOTIC/DEGENERATE) wrapped with `_attach_probe_state` which attaches `_probe_hardness`, `frozen_fraction`, `mean_hardness` to results when hardening is on.
- GPU dispatch bypassed when flag on (HIP kernel is unaware of hardness).

Full patch is preserved in [assets/per-cell-crystallization-probe-2026-05-22/](assets/per-cell-crystallization-probe-2026-05-22/) — re-apply with `git apply` to reproduce.

### `scripts/probe_crystallization.py` (archived)

Single-comparison and `--sweep` modes. Computes connected-component count, Moran's I, largest-component fraction, |Δframe| against a baseline run. Forces CPU dispatch via `WHEELER_DISABLE_GPU=1`.

## Sweep results

| scenario | ticks | frozen_frac | mean_hardness | n_components | largest_pct | Moran's I | exit reason |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 21 | 0 | 0 | 0 | — | -0.310 | p99_alive |
| probe-def | 21 | 0.023 | 0.62 | **93** | 0.02 | -0.311 | p99_alive |
| probe-fast | 12 | 0.961 | 0.97 | **1** | 1.00 | -0.310 | freeze_fraction |
| probe-hippo | 24 | 0 | 0.68 | 0 | — | -0.304 | p99_alive |
| probe-tight | 31 | 0.969 | 0.99 | **1** | 1.00 | -0.318 | freeze_fraction |
| probe-max | 12 | 0.961 | 0.97 | **1** | 1.00 | -0.310 | freeze_fraction |

Configurations:
- **probe-def**: `HARDEN_RATE=0.05, SETTLE_EPSILON=1e-3`, hash seed.
- **probe-fast**: `HARDEN_RATE=0.30, SETTLE_EPSILON=0.01`, hash seed.
- **probe-hippo**: default params, hippocampus (n-gram) seed.
- **probe-tight**: default params, `stability_threshold=1e-6, max_iters=200`.
- **probe-max**: all probe-fast + probe-tight overrides combined.

|Δframe| mean (probe vs baseline) was ≤ 0.011 across every hash-seeded scenario — i.e., the probe-on frames are essentially identical to the baseline attractor; hardness only slowed the trajectory, it did not change the destination.

## Finding

**The substrate produces exactly two regimes, neither of which is "clumps":**

1. **Low-rate regime** (HARDEN_RATE ≤ 0.05): the global p99 convergence rule fires before hardness can cross the freeze threshold on most cells. A small fraction of stragglers crosses, but they appear as ~90 disconnected **single-cell components** — salt-and-pepper noise, not clumps. The largest connected component is ~1 cell. Mean hardness across the grid is 0.6 (almost-frozen everywhere), but the 0.9 cutoff is too high for the substrate's natural settling timeline.
2. **High-rate regime** (HARDEN_RATE ≥ 0.30, or longer convergence horizon): essentially every cell freezes nearly simultaneously into **one giant 4096-cell component**. The freeze_fraction termination fires, but the geometry is "the whole grid is one blob", not multiple disjoint patches.

**Neither regime produces the user's "web-like clumps" geometry.** And there's no parameter setting in between that does — the transition is sharp because cells settle nearly synchronously.

## Why (the architectural finding)

The 3-state CA update rule (`apply_ca_dynamics`, `dynamics.py:54-71`) is **Lyapunov-monotonic and spatially synchronous**:
- Every cell evaluates its neighborhood every tick.
- Updates are deterministic, sign-preserving (local max pushes to +1, local min to -1, slope toward `max_neighbor`).
- There is no asynchrony, no refractory state, no traveling-front behavior.

Excitable-media clumps (Belousov–Zhabotinsky, neural waves, forest-fire CAs) arise from **spatial asymmetry in settling time** — some patches stabilize while neighbors keep moving because the update rule has propagation delays, async update schedules, or non-monotonic local dynamics. The current CA has none of those. Cells converge in lockstep.

Adding per-cell hardness on top of a synchronously-converging substrate cannot manufacture spatial heterogeneity that the substrate itself does not produce. The data is unambiguous: either nobody freezes in time (all settle ~together, hardness misses the window) or everybody freezes (all settle ~together, hardness catches them all). The bimodality is itself the evidence.

## What would actually produce clumps

Not the goal of this experiment, but for the record: clump geometries on this substrate would require modifying `apply_ca_dynamics` itself. Minimal candidates:

- **Asynchronous update**: each tick, update a random subset of cells (e.g. 50%). Introduces spatial heterogeneity in settling time.
- **Refractory state**: after a cell flips role, it cannot update again for K ticks. Creates propagation delays.
- **Non-monotonic dynamics**: e.g., a cell's role can invert under specific neighbor configurations. Breaks Lyapunov monotonicity, enabling waves and persistent patterns.

Each of these would diverge from canon's "global attractor per memory" framing in CANON.md §1.1 and architecture.md §6. They are not small experiments; they are alternative substrates.

## Decision

Reverted to baseline. The probe code is preserved alongside this report for reference. The flag-off path was byte-identical to baseline, so no live functionality was affected by the experiment, but keeping the dead code in the build adds maintenance surface for no benefit — clean removal is correct.

Filed under: substrate-level negative results. If the user wants clump geometry in the future, the right starting point is to design a different update rule, not to revive this probe.

## Reproduction

To re-run this experiment:

1. Apply the diffs (see [assets folder](assets/per-cell-crystallization-probe-2026-05-22/) for full patch content) to `wheeler_memory/constants.py` and `wheeler_memory/dynamics.py`.
2. Copy `assets/per-cell-crystallization-probe-2026-05-22/probe_crystallization.py` back to `scripts/`.
3. Run `python scripts/probe_crystallization.py --sweep` from the project root.
4. Outputs to `probe_out/` (PNGs) and stdout (metric tables).

`pytest -m "not slow and not embed"` should pass byte-identically with the diffs applied (flag defaults to False), confirming the refactor preserves the GPU/CPU/OSCILLATING/DEGENERATE code paths.

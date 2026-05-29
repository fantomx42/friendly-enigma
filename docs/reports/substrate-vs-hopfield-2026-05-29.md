# Substrate vs Hopfield + SCM Ablation — 2026-05-29

**Status:** Two pre-registered head-to-heads. **Both FAIL** on the registered criteria. Findings stand; no code reverted.
**Branch:** `claude/refine-local-plan-NSoSl`
**Companion artifacts:** `substrate_comparison.tsv`, `scm_ablation.tsv`, scripts under `scripts/bench/`.

## Why this experiment

After the per-cell crystallization probe (2026-05-22) characterized the substrate as *synchronous, Lyapunov-monotonic descent to a single global attractor* — essentially the recall mechanism of a classical Hopfield network — a precise empirical question came into focus:

> *Is Wheeler measurably more than a Hopfield net + an embedding?*

This report records two pre-registered tests of that question, designed and committed before any code ran (see `plans/im-not-sure-what-steady-church.md`). Each test isolates one architectural variable; together they target the substrate and the SCM-as-waveguide claims, the two load-bearing pieces of the project.

## Pre-registered criteria

The criteria were written before running anything:

- **Experiment A — substrate.** Pass iff Wheeler CA's capture radius (largest ε with mean fidelity ≥ 0.9 in `bench_reconstruction.py`'s perturbation regime) is **strictly greater** than a Hebbian Hopfield network trained on the same TEST_INPUTS attractors.
- **Experiment B — SCM-as-waveguide.** Pass iff physics-question recall accuracy under mixed-domain storage is **strictly greater** with the full SCM than with the SCM forced to all-zeros, with the gap excluding zero under bootstrapped 95% CI.

No moving goalposts. Each experiment reports independently.

## Experiment A — substrate head-to-head

`scripts/bench/bench_substrate_vs_hopfield.py`. Same TEST_INPUTS (20 patterns,
α = 0.0049, well below Hopfield's α=0.138 limit). Same Gaussian perturbation
ε ∈ {0.1, 0.25, 0.5, 0.75, 1.0}. Same Pearson fidelity ≥ 0.9 recovery
criterion. Wheeler scored against its continuous attractor; Hopfield (centered
Hebbian — patterns are biased ~77% +1, standard correction applies) scored
against the sign-snapped attractor.

| ε | Wheeler fidelity | recovery | Hopfield fidelity | recovery |
|---:|---:|---:|---:|---:|
| 0.10 | 0.911 | 100% | 1.000 | 100% |
| 0.25 | 0.911 | 100% | 1.000 | 100% |
| 0.50 | 0.905 | 75% | 1.000 | 100% |
| 0.75 | 0.823 | 0% | 1.000 | 100% |
| 1.00 | 0.700 | 0% | 1.000 | 100% |

| metric | Wheeler | Hopfield |
|---|---:|---:|
| capture radius | ε ≤ 0.50 | **ε ≤ 1.00** |
| mean fidelity | 0.850 | **1.000** |
| recovery rate | 55% | **100%** |
| spurious rate | 60% | 0% |
| mean ticks | 16.0 | 1.6 |

**Verdict: FAIL.** Centered Hopfield captures further than Wheeler at every
tested noise level.

### Honest caveat (committed in advance)

Wheeler's CA is spatially local (von Neumann neighborhood, `dynamics.py:54-71`); classical Hopfield is all-to-all. Locality typically *lowers* capacity, so:

- A Wheeler **win** would have been an unambiguous architectural victory (overcoming the locality handicap).
- A Wheeler **loss** is *ambiguous*: it could mean the substrate is genuinely worse, or it could be the locality tax. This experiment cannot distinguish them.

What it does pin down is the bound: at α=0.0049, Wheeler's substrate **does not measurably exceed** a textbook Hopfield baseline.

α=0.0049 is also deep in Hopfield's trivial regime — a regime where any healthy associative memory should saturate. Wheeler's 55% recovery at this load is itself the more striking number: it sits below what the literature considers easy. The post-2026-05-22 picture (Lyapunov-monotonic synchronous descent) is consistent with this: the basin geometry is well-behaved but the basins are narrow relative to the noise budget used by `bench_reconstruction.py`.

## Experiment B — SCM-as-waveguide ablation

`scripts/bench/bench_scm_ablation.py`. 10 physics-flavoured facts + 10
history-flavoured facts, hippocampus encoder (intra-class correlation ~0.43,
cross-class ~0.07 — a real crosstalk regime). Three conditions:

1. Physics-only storage, physics queries.
2. Mixed storage (physics + history), physics queries, full Wheeler SCM.
3. Mixed storage, physics queries, SCM forced to all-zeros + apply_learning=False.

| corruption | warmup epochs | physics-only | mixed + full SCM | mixed + frozen SCM | SCM gap | CI 95% | verdict |
|---:|---:|---:|---:|---:|---:|---|---|
| 0.30 (pre-reg) | 1 | 100% | 100% | 100% | +0.0% | [+0.0%, +0.0%] | FAIL_INERT_SCM |
| 0.70 (stress) | 3 | 100% | 100% | 100% | +0.0% | [+0.0%, +0.0%] | FAIL_INERT_SCM |

**Verdict: FAIL_INERT_SCM.** The two conditions were degenerately identical because the SCM never accumulated any state during warmup — `scm_nonzero_cells = 0/4096` in both runs, even after 3 epochs of 70% corrupted cues.

### Why the SCM didn't activate

Reading the code explains the result. The SCM has two update channels:

1. **`update_from_recall`** (`scm_grid.py:151-247`) — gated by a cold-start condition: `not np.any(m_sign != 0) and advantage < 0`. The first call to a fresh SCM has `kappa_base = 0`. If the first recall returns `kappa > 0` (any successful retrieval), `advantage = kappa - 0 > 0`, cold-start does not seed, the mask `(m_sign != 0) & (credit > 0)` is empty, and the function returns 0 updates. From that point on, `kappa_base` ratchets upward via EMA over kappa, and cold-start can never trigger again because it requires `advantage < 0` AND `m_sign` still all-zero — a configuration the system only ever passes through on the very first call.

2. **`update(mask, direction)`** via `self_consistency_check` (`interference.py:378-440`) — only ever called by `scripts/wheeler_mmlu.py`. Not invoked by the standard recall path.

So on any benchmark whose first recall succeeds — i.e. any benchmark Wheeler currently passes — **the SCM does not learn during recall**. It is reachable only via the MMLU decoder loop's self-consistency check.

This is consistent with the May audit (`audit-2026-05-04.md`) finding that the §3.3.5 "sleeping giant" framing was outdated — but the closure of that loop turns out to be narrow enough that the typical recall path never wakes it.

## Combined reading

Both experiments returned the same direction. Mapped back to the outcome matrix committed in `plans/im-not-sure-what-steady-church.md`:

> Wheeler ties / loses + SCM gap null → Position 2 confirmed. Right-size: this is a well-engineered ternary associative memory. Retire the intelligence claim. That is a real, honest, shippable thing — not a failure.

The system that exists:

- A spatially-local, Lyapunov-monotonic ternary CA that produces clean global-attractor settling, demonstrably stable (quality score 0.0107, p99 convergence, 0.9997 alive fraction).
- 85% mean reconstruction fidelity against the sacred corpus, 55% recovery rate at the configured noise budget.
- A hippocampus encoder climbing on SimLex (0.10 → 0.26) — real progress on native semantics, still behind the MiniLM bar (0.43).
- An SCM grid whose recall-driven update channel is reachable but, by the current cold-start gate, does not actually fire on normal recall paths.

What is *not* supported by current evidence:

- The substrate beating a baseline Hopfield network at low load (it doesn't, at this α).
- The SCM contributing measurably to recall accuracy on standard paths (it didn't, in either configuration tested).
- Native CA reasoning at MMLU-style tasks (best config 25.9%, statistically at chance per `results/BASELINES.md`).

## What this rules in and out

**Rules out** (under the current code and pre-registered tests):

- "The CA substrate provides more recall capacity than Hopfield." Not at α=0.0049 on TEST_INPUTS.
- "The SCM gating measurably improves recall under cross-domain crosstalk." Not while its update path remains dormant on successful recalls.

**Does not rule out**:

- That the SCM contributes via the MMLU decoder path (`self_consistency_check` does fire there). Not tested in this report; would require a different benchmark.
- That Wheeler exceeds Hopfield at higher α (closer to capacity), where Hopfield's catastrophic cliff might show and locality might earn its keep via graceful degradation. Not tested — sacred TEST_INPUTS pins N=20.
- That a different substrate (asynchronous, refractory, non-monotonic — the directions the 2026-05-22 probe report names) would behave differently. Out of scope.

## What to do with this

Three options, in increasing scope:

1. **Right-size and ship what works.** The reconstruction-memory thing exists and is well-engineered. Document it as such, retire reasoning claims, prune scaffolding around the unvalidated parts.
2. **Open one more door before deciding.** Extend the comparison to higher α (synthetic patterns matched to TEST_INPUTS statistics, swept N upward) to see if Hopfield's α=0.138 cliff lets Wheeler overtake at scale. This would be informative but does not retroactively change today's verdict.
3. **Acknowledge the SCM cold-start gate as a bug, not a feature, and revisit.** The current trigger condition makes the SCM unreachable on successful recalls; if the intent was "learn from every recall," the gate is wrong and the SCM has never been measured at its design-intent behavior. This is a separate piece of work; this report does not prejudge it.

## Decision

No code reverted. The two new benchmarks ship as the recorded baseline. The TSVs (`substrate_comparison.tsv`, `scm_ablation.tsv`) are versioned alongside `reconstruction.tsv` so future runs are comparable. Both experiments are re-runnable on any host:

```bash
python scripts/bench/bench_substrate_vs_hopfield.py --cpu-only
python scripts/bench/bench_scm_ablation.py --cpu-only
```

Filed under: pre-registered architectural negative results. Follow-up direction is the user's call.

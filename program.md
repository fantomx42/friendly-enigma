# Wheeler Memory — Autoresearch Program

Autonomous parameter optimization for the CA-based associative memory system.
Modelled on Karpathy's autoresearch: modify one file, run a fixed benchmark, keep or revert.

---

## Objective

Minimize the composite quality score output by `wheeler-bench`.

```
score = 0.6 * avg_abs_correlation
      + 0.2 * (1 - convergence_ratio)
      + 0.1 * (median_ticks / 1000)
      + 0.1 * (1 - avg_alive_fraction)
```

Lower is better.  Current baseline is ~0.30–0.35.  Target: score < 0.25.

---

## Rules

### Modify only this file
```
wheeler_memory/constants.py
```
All other files are off-limits.

### Never touch
- `wheeler_memory/hashing.py` — deterministic encoding is sacred
- `scripts/bench_quality.py` — especially the TEST_INPUTS list
- `wheeler_memory/storage.py`, `chunking.py`, `rotation.py`

### Experiment loop
1. Edit `constants.py` (one or a few parameters at a time)
2. Commit: `git add wheeler_memory/constants.py && git commit -m "Experiment N: <description>"`
3. Run: `wheeler-bench --commit <hash7> --changed "<param>" --notes "<why>"`
4. If score improved: keep, start next iteration
5. If score worsened by > 10 %: `git revert HEAD --no-edit`, note it in results.tsv

---

## Parameter space

### Tier 1 — CA Dynamics (highest impact, explore first)

| Constant | Current | Safe range | Effect |
|----------|---------|------------|--------|
| `MAX_PUSH_STRENGTH` | 0.35 | [0.20, 0.50] | Higher → sharper attractors, faster convergence, possibly lower diversity |
| `SLOPE_FLOW_STRENGTH` | 0.20 | [0.10, 0.30] | Higher → faster information mixing; too high collapses diversity |

Tip: try increasing `MAX_PUSH_STRENGTH` to 0.40–0.45 first.  Often improves avg correlation.

### Tier 2 — Salience thresholds (moderate impact)

| Constant | Current | Safe range | Effect |
|----------|---------|------------|--------|
| `SALIENCE_THRESHOLD_MED` | 1e-4 | [5e-5, 5e-4] | Default convergence precision; lower → tighter basins |
| `SALIENCE_MAX_ITERS_MED` | 1000 | [500, 2000] | Default iteration cap; raise if median_ticks is near the cap |

### Tier 3 — Temperature / warming (lower benchmark impact, but affects real usage)

| Constant | Current | Safe range |
|----------|---------|------------|
| `HALF_LIFE_DAYS` | 7.0 | [3.0, 14.0] |
| `WARMTH_HOP1` | 0.05 | [0.02, 0.10] |
| `ASSOCIATION_THRESHOLD` | 0.5 | [0.40, 0.70] |

---

## Suggested exploration order

1. **Iterations 1–8**: vary `MAX_PUSH_STRENGTH` in steps of 0.05 (try 0.30, 0.40, 0.45, 0.25 …)
2. **Iterations 9–15**: fix best push strength; vary `SLOPE_FLOW_STRENGTH` in steps of 0.05
3. **Iterations 16–22**: fine-tune `SALIENCE_THRESHOLD_MED` around best configuration
4. **Iterations 23+**: micro-adjustments; try combinations

---

## Success criteria

| Metric | Target |
|--------|--------|
| `score` | < 0.25 |
| `avg_correlation` | < 0.40 |
| `convergence_ratio` | ≥ 0.90 |
| `max_correlation` | < 0.80 |
| `median_ticks` | < 400 |

---

## Setup

```bash
cd "/home/tristan/LocalLLM/wheeler memory"
source .venv/bin/activate
pip install -e .

# Get baseline
wheeler-bench

# After editing constants.py:
git add wheeler_memory/constants.py
git commit -m "Experiment 1: MAX_PUSH_STRENGTH 0.35 → 0.40"
wheeler-bench --commit $(git rev-parse --short HEAD) --changed "MAX_PUSH_STRENGTH" --notes "testing sharper attractors"
```

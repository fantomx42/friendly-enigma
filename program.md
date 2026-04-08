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

Lower is better.  Current best: **0.009** (target achieved).  Original baseline was ~0.30–0.35.

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

### Tier 1 — CA Dynamics (highest impact — tuned, see results.tsv)

| Constant | Tuned value | Original | Safe range | Effect |
|----------|-------------|----------|------------|--------|
| `MAX_PUSH_STRENGTH` | **0.57** | 0.35 | [0.20, 0.60] | Higher → sharper attractors, faster convergence |
| `SLOPE_FLOW_STRENGTH` | **0.55** | 0.20 | [0.10, 0.60] | Higher → faster information mixing |

Note: tuned values exceed the original safe-range ceilings. The expanded ranges reflect empirical results from 50 autoresearch iterations. **Caveat**: these aggressive dynamics erode distributional semantic signal (SimLex-999 rho drops from +0.046 raw to +0.034 evolved). Future tuning should balance convergence quality against signal preservation.

### Tier 2 — Salience thresholds (tuned)

| Constant | Tuned value | Original | Safe range | Effect |
|----------|-------------|----------|------------|--------|
| `SALIENCE_THRESHOLD_MED` | **0.1** | 1e-4 | [1e-4, 0.1] | Coarser convergence → fewer ticks |
| `SALIENCE_MAX_ITERS_MED` | 1000 | 1000 | [500, 2000] | Unchanged — median ticks dropped to ~5 via threshold tuning |

### Tier 3 — Temperature / warming (unexplored, lower benchmark impact)

| Constant | Current | Safe range |
|----------|---------|------------|
| `HALF_LIFE_DAYS` | 7.0 | [3.0, 14.0] |
| `WARMTH_HOP1` | 0.05 | [0.02, 0.10] |
| `ASSOCIATION_THRESHOLD` | 0.5 | [0.40, 0.70] |

### Tier 4 — Context-RI / distributional semantics (new)

| Constant | Current | Safe range | Effect |
|----------|---------|------------|--------|
| `CONTEXT_RI_BLEND` | **0.9** | [0.0, 1.0] | Blend weight: context-RI vs hippocampus n-grams. Higher = more distributional signal |
| `CONTEXT_RI_WINDOW` | 5 | [2, 10] | Context window radius for co-occurrence accumulation |

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
cd "/home/tristan/projects/wheeler-memory"
source .venv/bin/activate
pip install -e .

# Get baseline
wheeler-bench

# After editing constants.py:
git add wheeler_memory/constants.py
git commit -m "Experiment 1: MAX_PUSH_STRENGTH 0.35 → 0.40"
wheeler-bench --commit $(git rev-parse --short HEAD) --changed "MAX_PUSH_STRENGTH" --notes "testing sharper attractors"
```

---

## Overnight Run Log — 2026-04-07 (completed)

### Outcome
The overnight run completed 12 iterations but did not improve beyond 24.0% MMLU accuracy
baseline. Parameters tested: `RECALL_MIN_SIM` (various values), `RECALL_ENCODER` (blended, hash).
Several iterations hit errors (failed commits, failed param edits) and large regressions.

**Key result:** CONTEXT_RI_BLEND tuning (manual, pre-overnight) was the real win:
- CONTEXT_RI_BLEND 0.5 → 0.9 boosted SimLex-999 rho from +0.034 to **+0.101**

### Current Best State (as of 2026-04-08)
- wheeler-bench score: **0.009** (CA dynamics solved)
- SimLex-999 context-RI (evolved): rho = **+0.101** (CONTEXT_RI_BLEND=0.9)
- SimLex-999 context-RI (raw frames): rho = +0.046
- SimLex-999 hippocampus: rho = -0.032 (no semantic signal, expected)
- SimLex-999 MiniLM (external ceiling): rho = +0.446

### Open Problem
CA evolution still erodes distributional signal (raw +0.046 → evolved +0.034 before blend tuning).
The aggressive dynamics (MAX_PUSH 0.57, SLOPE_FLOW 0.55) were optimized for convergence quality,
not signal preservation. Next research direction: find CA dynamics that amplify semantic structure.

### Next Sweep Candidates
- Softer CA dynamics specifically for context-RI frames (lower MAX_PUSH/SLOPE_FLOW)
- Per-encoder CA parameters (corpus-tight vs context-loose, like experiential)
- Larger context windows (`CONTEXT_RI_WINDOW` > 5)
- Richer training corpora beyond WikiText-103

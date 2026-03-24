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

## Overnight Agenda — 2026-03-24

### Current State
- Best wheeler-bench score: 0.009 (CA dynamics solved)
- MMLU semantic (hippocampus): 28.7% baseline
- MMLU semantic (hippo-word, learned vectors from 17k-word corpus): 27.7%
- Multi-choice mode added but encoder-limited (~27% regardless of params)
- Word co-occurrence vector training activated in learn pass (SVD on PMI)
- Previous overnight (2026-03-23): swept RECALL_K, RECALL_MIN_SIM, RECALL_ENCODER — all reverted

### Tonight's Priority: Semantic mode with hippo-word blend tuning

Word vectors are trained from the full main data dir (17,451 words). The hippo-word
encoder blends hippocampus n-grams with learned word vectors weighted by `WORD_HIPPO_BLEND`.
The optimal blend may not be 0.3 (default). Tonight sweeps the blend ratio and related params.

**Pre-step (run ONCE before loop):**
```bash
cd /home/tristan/projects/wheeler-memory && source .venv/bin/activate
python -c "
from wheeler_memory.word_encoder import train_word_vectors, save_word_vectors
vectors, vocab = train_word_vectors()
save_word_vectors(vectors, vocab)
print(f'Trained {len(vocab)} word vectors')
"
```

**Benchmark command for each iteration:**
```bash
wheeler-mmlu --subjects high_school_physics conceptual_physics college_physics --mode semantic --encoder hippo-word --split test 2>&1 | tail -5
```

**Parameters to sweep (all in constants.py):**
1. `WORD_HIPPO_BLEND` — try 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 (current: 0.3)
   Rationale: Find optimal hippocampus-to-word ratio; 0.3 may be suboptimal
2. `RECALL_K` — try 3, 5, 10, 15, 20 (current: 10)
   Rationale: Recall count affects semantic scoring through cache search
3. `RECALL_MIN_SIM` — try 0.0, 0.05, 0.10, 0.15 (current: 0.15)
   Rationale: Threshold affects which attractors contribute to scoring

Score to track: MMLU accuracy (%) on physics subjects via semantic mode. Target: > 30%.

### Model
Local: qwen3.5:9b (code generation)
Orchestration: Claude opus (loop management)

### Budget
- Max iterations: 20
- Stop condition: MMLU semantic > 32% OR max iterations reached

### Benchmark Command Override
```bash
wheeler-mmlu --subjects high_school_physics conceptual_physics college_physics --mode semantic --encoder hippo-word --split test
```

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

## Overnight Agenda — 2026-04-07

### Current State
- wheeler-bench score: 0.009 (CA dynamics solved — convergence is great)
- SimLex-999 context-RI (raw frames, no CA): rho = +0.046 (weak positive signal)
- SimLex-999 context-RI (evolved): rho = +0.034 (CA evolution HURTS — erodes signal)
- SimLex-999 hippocampus: rho = -0.032 (noise, no semantic signal)
- SimLex-999 MiniLM (external ceiling): rho = +0.446
- Context-RI vectors trained on WikiText-103 (1.16M lines) + Wheeler memories (3,275 entries)
- Vocab: 500,227 words, 384-dim context vectors

### Key Observation
CA evolution destroys semantic signal. Raw context-RI frames score +0.046 but drop to +0.034
after evolution. The aggressive dynamics (MAX_PUSH 0.57, SLOPE_FLOW 0.55 — both above safe
range ceilings) were tuned for convergence quality, not signal preservation. The goal is to
find dynamics that AMPLIFY semantic structure rather than eroding it.

### Tonight's Priority: CA dynamics that preserve/amplify distributional semantics

Sweep CA dynamics parameters to maximize SimLex-999 Spearman rho for the `context` encoder.
The quality score (`wheeler-bench`) must not regress above 0.05 (currently 0.009).

**Pre-step (run ONCE before loop):**
```bash
cd /home/tristan/projects/wheeler-memory && source .venv/bin/activate
# Context-RI vectors already trained — verify they exist:
ls -la ~/.wheeler_memory/context_ri_vectors.npz
```

**Benchmark command for each iteration:**
```bash
wheeler-simlex --encoder context --mode pearson 2>&1 | grep "Spearman rho"
```

**Guard rail — also check quality hasn't regressed:**
```bash
wheeler-bench 2>&1 | grep "score"
```

**Parameters to sweep (all in constants.py):**

Phase 1 — Soften the CA push (iterations 1-8):
1. `MAX_PUSH_STRENGTH` — try 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 (current: 0.57)
   Rationale: Weaker push preserves more of the input frame's distributional structure.
   Lower values may allow subtle similarity patterns to survive evolution.

Phase 2 — Tune mixing rate (iterations 9-14):
2. `SLOPE_FLOW_STRENGTH` — try 0.10, 0.15, 0.20, 0.25, 0.30, 0.40 (current: 0.55)
   Rationale: Slower mixing preserves local structure. The distributional signal is
   encoded in the spatial pattern — aggressive mixing homogenizes it away.

Phase 3 — Context-RI blend and convergence (iterations 15-22):
3. `CONTEXT_RI_BLEND` — try 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0 (current: 0.5)
   Rationale: Higher blend = more context signal, less hippocampus n-gram noise.
4. `SALIENCE_THRESHOLD_MED` — try 5e-3, 1e-3, 5e-4, 1e-4, 5e-5 (current: 1e-4)
   Rationale: Looser convergence may stop evolution before it erases signal.

Phase 4 — Fine-tune combinations (iterations 23+):
5. Best MAX_PUSH × best SLOPE_FLOW × best CONTEXT_RI_BLEND combinations.

### Score Tracking

| Metric | Baseline | Target | Guard |
|--------|----------|--------|-------|
| SimLex rho (evolved) | +0.034 | > +0.10 | — |
| SimLex rho (raw) | +0.046 | — | reference only |
| wheeler-bench score | 0.009 | — | must stay < 0.05 |

### Model
Local: gemma4:26b or qwen3:14b (code generation)
Orchestration: Claude opus (loop management)

### Budget
- Max iterations: 50
- Stop condition: SimLex rho > +0.15 OR max iterations reached
- Revert threshold: SimLex rho worsens by > 20% from best OR wheeler-bench > 0.05

### Benchmark Command Override
```bash
wheeler-simlex --encoder context --mode pearson
```

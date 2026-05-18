# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Canon

Architectural source-of-truth: [CANON.md](./CANON.md). When canon and this file conflict, canon wins on architecture; this file wins on operational workflow.

## Setup

```bash
source .venv/bin/activate
pip install -e ".[embed]"   # with sentence-transformers
pip install -e .            # minimal (numpy/scipy/matplotlib/psutil)
```

System Python is externally-managed (Arch/CachyOS) — always use the venv. `ruff format` auto-runs on `.py` edits via PostToolUse hook.

## Testing

```bash
pytest                              # all
pytest -m "not slow"                # skip slow
pytest -m "not embed"               # skip sentence-transformers
pytest tests/test_dynamics.py::test_converges
```

Markers: `slow`, `embed`.

## Benchmarks

```bash
wheeler-bench                       # CA quality (lower better, target < 0.25)
wheeler-simlex --sweep              # live encoder signal (SimLex-999)
wheeler-mmlu --subjects high_school_physics --mode cortex
```

Quality score: `0.6·avg_corr + 0.2·(1-conv_ratio) + 0.1·(ticks/1000) + 0.1·(1-alive)`.

## Sacred files — do not modify without reason

| File | Why |
|---|---|
| `wheeler_memory/hashing.py` | Deterministic SHA-256 — foundational to stored data |
| `wheeler_memory/storage.py` | Locked storage contract |
| `wheeler_memory/chunking.py` | Locked domain routing |
| `wheeler_memory/rotation.py` | Locked rotation logic |
| `scripts/bench_quality.py` TEST_INPUTS | Fixed corpus — results must be comparable across runs |
| `scripts/wheeler_simlex.py` ALL_ENCODERS | Encoder registry that drives sweeps |

## Constraints

- **Pure Python core.** No LLM/ML dependencies beyond numpy/scipy/matplotlib/psutil. `sentence-transformers` is optional (`.[embed]`).
- **Recall is CPU-only by design** (canon §1.4). HIP kernels in `accel/` accelerate **batch operations only** — never wire GPU into the recall path.
- **No Rust, no conductor, no `trauma.py`.** Removed Feb 2026; do not reintroduce.
- **`constants.py` is the ONLY file modified during autoresearch tuning.** See `program.md`.
- Don't wrap the CA — the dynamics ARE the product. No LLM-based scoring.

## Architecture

### Three-grid interference (the recall equation)

```
Answer(i, j) = Corpus(i, j) × Experiential(i, j) × (1 − |SCM(i, j)|)
```

Three same-shaped 64×64 grids with different temporal dynamics:

| Grid | Temperature | Role | File |
|---|---|---|---|
| Corpus | Cold, slow | Durable knowledge | `storage.py` |
| Experiential | Hot, fast | Working memory | `experiential.py` |
| SCM (Map) | Glacial | Trust topology — gates where interference is permitted | `scm_grid.py` |

SCM is **feedback-only** — no autonomous CA dynamics. Hardened cells (`|SCM|→1`) become opaque; quiescent cells let interference through. Scoring is cell-wise weighted Pearson with `w = 1 − |SCM|`.

### SCM acronym collision

- `scm_grid.py` — Structural Coherence **Map**. The gate in the answer equation.
- `cortex_scm.py` — Structural Coherence **Measure**. Post-recall classifier (`SYNTHESIS / NOVEL / HALLUCINATION`).

Different objects. Check which file is in scope when reading "SCM".

### CA dynamics

3-state rule on Von Neumann 4-neighborhood with wrapping boundaries. Local maxima push toward +1 (`MAX_PUSH_STRENGTH`), minima toward −1, slopes flow uphill (`SLOPE_FLOW_STRENGTH`). Terminal states: `CONVERGED`, `OSCILLATING`, `DEGENERATE`, `CHAOTIC`.

### Two-tier recall (v0.3.6)

`wheeler_memory/recall_api.py` splits identity from content:

- `recognize(query)` — single-pass Pearson against stored attractors on the **raw** query frame. No CA loop. Returns `BasinSeed` or `None`.
- `reconstruct_from_seed(seed, query)` — warm-starts CA from stored attractor blended with raw query (~2× fewer ticks vs cold path).

Per-basin **Temporal Stability** `T ∈ [0, 1]` lives in `index.json` (`metadata.t_stability`). With `--learn`, mature basins (T → 1) are near-rigid; fresh basins absorb input rapidly.

### Encoders

| Encoder | Use when |
|---|---|
| `hash` | Default for `wheeler-recall` and reproducible benchmarks |
| `hippocampus` | Native n-gram RI — active production target; default for `wheeler-simlex` |
| `embedding` | MiniLM baseline to clear (requires `.[embed]`) |
| `blended` | Default for user-facing surfaces |

SimLex-999 is the **live signal** of encoder progress, not MMLU.

### Cortex (3-tier scoring, separate from the three grids)

L1 correlation graph (`cortex.py`) → L2 settlement CA on graph topology → L3 native classifier (`cortex_classifier.py`, ~11K-param numpy SGD).

### Module map

| Group | Files |
|---|---|
| Encoding | `hashing.py`, `hippocampus.py`, `embedding.py`, `word_encoder.py`, `brick.py` |
| CA engine | `dynamics.py`, `oscillation.py`, `rotation.py`, `diagnostics.py` |
| Storage & recall | `storage.py`, `reconstruction.py`, `recall_api.py`, `t_metadata.py`, `cache.py` |
| Three-grid interference | `scm_grid.py`, `experiential.py`, `interference.py`, `similarity.py`, `trajectory.py` |
| Cortex | `cortex.py`, `cortex_scm.py`, `cortex_classifier.py` |
| Lifecycle | `temperature.py`, `attention.py`, `warming.py`, `consolidation.py`, `eviction.py` |
| Agents | `agent.py`, `decoder.py`, `generation.py`, `language_wheeler.py` |
| Accel | `accel/hip/` (batch ops only — recall stays CPU) |
| Config | `constants.py` (all tunables) |

**Live vs archived `theories/`:** `wheeler_memory/theories/` is production-supporting (imported by agent/decoder/mmlu). `notes/theories/` is archived (`lichtenberg.py`, `resonance.py`, `structured.py` — moved in v0.3.6).

### MMLU framing

Currently ~25% (near chance). **Corpus-limited, not architecture-limited** (canon §8.2). Treat MMLU as a corpus-health metric — do not "fix" it by altering the engine.

## CLI

All commands registered in `pyproject.toml [project.scripts]` — run `pip install -e .` after adding entry points. Common flags: `--data-dir`, `--chunk`, `--encoder`, `--salience`, `--verbose`. Full reference: [docs/cli.md](docs/cli.md).

## Autoresearch

See `program.md`. Loop: edit one param in `constants.py` → commit → `wheeler-bench --commit <hash7> --changed "<param>"` → keep if improved, revert if worse by >10%. Key params: `MAX_PUSH_STRENGTH`, `SLOPE_FLOW_STRENGTH`, `SALIENCE_THRESHOLD_MED`, `SALIENCE_MAX_ITERS_MED`.

## Hooks (auto-configured)

PostToolUse on Write|Edit:

1. `ruff format` on edited `.py` (10s)
2. `pytest` when `wheeler_memory/` or `tests/` change (60s)
3. `wheeler-bench` when `constants.py` changes (120s)

Review/disable via `/hooks`.

## Open work (canon §9)

1. FCAS address resolution `[DESIGNED]` — wire `(hash, depth)` tuples into recall.
2. Wheeler-native eval design `[SPECULATIVE]`.
3. Corpus population strategy `[OPEN]` — gates MMLU.
4. Cross-cube interference semantics `[SPECULATIVE]`.

See `plans/` for active threads.

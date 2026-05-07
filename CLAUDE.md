# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Canon

For architectural source-of-truth (axioms, three-grid model, SCM grid topology, retrieval/reconstruction division, status of every component), read [CANON.md](./CANON.md). When canon and this file conflict, canon wins on architecture and this file wins on operational workflow.

## Setup

```bash
cd "/home/tristan/projects/wheeler-memory"
source .venv/bin/activate
pip install -e ".[embed]"   # with sentence-transformers
pip install -e .             # minimal (numpy/scipy/matplotlib/psutil only)
```

System Python is externally-managed (Arch/CachyOS) — always use the venv. Formatting: `ruff format` (auto-runs via PostToolUse hook on .py edits).

## Testing

```bash
pytest                          # all tests
pytest -m "not slow"            # skip slow tests
pytest -m "not embed"           # skip sentence-transformers tests
pytest tests/test_dynamics.py   # single file
pytest tests/test_dynamics.py::test_converges  # single test
```

Markers: `slow` (long-running), `embed` (requires sentence-transformers).

## Benchmarks

```bash
wheeler-bench                                          # CA quality score (lower is better)
wheeler-mmlu --subjects high_school_physics --mode semantic  # MMLU benchmark
```

Quality score formula: `0.6*avg_corr + 0.2*(1-conv_ratio) + 0.1*(ticks/1000) + 0.1*(1-alive)`. Target: < 0.25.

## Sacred Files — DO NOT MODIFY

- `wheeler_memory/hashing.py` — deterministic SHA-256 encoding, foundational to all stored data
- `scripts/bench_quality.py` TEST_INPUTS list — fixed benchmark corpus, results must be comparable across experiments
- `wheeler_memory/storage.py` — locked storage contract
- `wheeler_memory/chunking.py` — locked domain routing
- `wheeler_memory/rotation.py` — locked rotation logic

## Architecture Constraints

- **Pure Python.** The CA is the reasoning engine, not a wrapper around language models.
- **No external LLM/ML dependencies in core.** `sentence-transformers` is optional (`.[embed]`); core must work without it.
- **No Rust, no conductor, no trauma.py.** These were removed Feb 2026. Do not reintroduce.
- **`constants.py` is the ONLY file modified during autoresearch** parameter tuning. See `program.md` for the full protocol.

## Anti-Patterns

- Don't add Python dependencies beyond numpy/scipy/matplotlib/psutil to core
- Don't create wrapper classes around the CA — the dynamics ARE the product
- Don't add LLM-based scoring or reasoning — native CA intelligence is the point
- Don't refactor modules you weren't asked to touch
- Don't create new files when editing existing ones suffices

## Architecture

### Data Flow

```
Text → Encoder → 64×64 frame ([-1,+1]) → CA evolution (~40-50 ticks) → Attractor
                                                                          ↓
                                              Chunked storage (~/.wheeler_memory/chunks/<domain>/)
                                                                          ↓
Query → Same pipeline → Query attractor → Pearson correlation search → Top-K results
                                                                          ↓
                                              [Optional] Reconstructive recall (blend + re-evolve)
```

Encoders: `hash` (deterministic, default for benchmarks), `hippocampus` (native n-gram), `embedding` (sentence-transformers), `blended` (default for user-facing).

CA dynamics: 3-state rule — local maxima push toward +1, minima toward -1, slopes flow uphill. Von Neumann (4-neighbor) topology with wrapping boundaries.

### Module Map (`wheeler_memory/`)

| Group | Files | Role |
|-------|-------|------|
| Encoding | `hashing.py`, `hippocampus.py`, `embedding.py`, `word_encoder.py` | Text → 64×64 frame |
| CA Engine | `dynamics.py`, `oscillation.py`, `rotation.py` | Frame → attractor evolution |
| Storage | `storage.py`, `chunking.py`, `brick.py`, `cache.py` | Attractor persistence + retrieval |
| Lifecycle | `temperature.py`, `warming.py`, `consolidation.py`, `eviction.py`, `attention.py` | Memory freshness, spreading activation, sleep, capacity |
| Cortex | `cortex.py`, `cortex_scm.py`, `cortex_classifier.py` | L1 graph → L2 settlement CA → L3 classifier |
| Agents | `agent.py`, `decoder.py`, `generation.py` | LLM wrapper, Language Wheeler, IT-from-BIT engine |
| Config | `constants.py` | All tunable parameters (centralized) |

### Cortex System (3-tier semantic scoring)

- **L1**: Pearson correlation adjacency graph over retrieved attractors, BFS clustering
- **L2**: Settlement CA — opinion diffusion on the correlation graph until convergence
- **L3**: Native semantic classifier — scores choices without external models

### Domain Chunks

Memories auto-route by keyword: `code`, `science`, `hardware`, `daily_tasks`, `meta`, `general` (fallback). Each chunk has independent index, attractors, bricks, and associations.

## CLI

All 16 commands registered in `pyproject.toml [project.scripts]`. Run `pip install -e .` after adding new entry points.

Common flags across commands: `--data-dir`, `--chunk`, `--encoder` (hash|hippocampus|embedding|blended|word), `--salience` (low|medium|high), `--verbose`.

## Autoresearch Protocol

See `program.md` for full details. Summary:

1. Edit only `constants.py` (one or a few params)
2. Commit the change
3. Run `wheeler-bench --commit <hash7> --changed "<param>"`
4. Keep if improved, revert if score worsened >10%

Key parameters: `MAX_PUSH_STRENGTH` (attractor sharpness), `SLOPE_FLOW_STRENGTH` (mixing rate), `SALIENCE_THRESHOLD_MED` (convergence precision), `SALIENCE_MAX_ITERS_MED` (iteration cap).

## Hooks (auto-configured)

Three PostToolUse hooks fire on Write|Edit:
1. **ruff format** — auto-formats any edited .py file (10s timeout)
2. **pytest** — runs full test suite when `wheeler_memory/` or `tests/` files change (60s timeout)
3. **wheeler-bench** — runs quality benchmark when `constants.py` changes (120s timeout)

Review/disable via `/hooks` in Claude Code.

## Task Discipline

For any non-trivial request, break work into discrete tasks using TaskCreate before starting. This prevents scope creep and makes progress visible. Complete each task before moving to the next — do not batch unrelated changes into a single edit pass.

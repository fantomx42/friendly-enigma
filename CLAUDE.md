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

> Architecture itself lives in [CANON.md](./CANON.md) §1.4 ("What this is not"). These are the enforcement rules that follow:

- **Pure Python in the engine.** No CUDA / ROCm / Vulkan paths inside the recall path. (The HIP kernel under `wheeler_memory/accel/` is batch-dispatch only — see `docs/architecture.md` §6.)
- **Core works without `sentence-transformers`.** The `.[embed]` extra is optional; core depends only on numpy/scipy/matplotlib/psutil.
- **No Rust, no conductor, no trauma.py.** These were removed Feb 2026. Do not reintroduce.
- **`constants.py` is the ONLY file modified during autoresearch** parameter tuning. See `docs/program.md` for the full protocol.

## Anti-Patterns

- Don't add Python dependencies beyond numpy/scipy/matplotlib/psutil to core
- Don't create wrapper classes around the CA — the dynamics ARE the product
- Don't add LLM-based scoring or reasoning — native CA intelligence is the point
- Don't refactor modules you weren't asked to touch
- Don't create new files when editing existing ones suffices

## Code Layout

> Architecture lives in [CANON.md](./CANON.md): three-grid model (§3), recall formula (§4), encoder layer plurality (§1.2), cortex three-tier scoring (§3.5), substrate-clock semantics for T (§3.6). This section is a "where does what live" pointer only.

### Module Map (`wheeler_memory/`)

| Group | Files | Role |
|-------|-------|------|
| Encoding | `hashing.py`, `hippocampus.py`, `embedding.py`, `word_encoder.py`, `language_wheeler.py` | Text → 64×64 frame (hashing is SHA-256 deterministic; language_wheeler is POS/grammatical) |
| CA Engine | `dynamics.py`, `oscillation.py`, `rotation.py` | Frame → attractor evolution |
| Storage | `storage.py`, `chunking.py`, `brick.py`, `cache.py`, `trajectory.py`, `trajectory_cache.py`, `similarity.py`, `polarity.py` | Attractor persistence + retrieval; trajectory fingerprints; pearson/spatial/hybrid similarity; dual-polarity antipodal pairs |
| Lifecycle | `temperature.py`, `warming.py`, `consolidation.py`, `eviction.py`, `attention.py` | Memory freshness, spreading activation, sleep, capacity, salience budget |
| Three-grid interference | `interference.py`, `scm_grid.py`, `experiential.py`, `reconstruction.py` | `Answer = Corpus·Experiential·(1−\|SCM\|)`; persistent trust topology; episodic grid; context-blended reconstructive recall |
| Recall API | `recall_api.py`, `recall_learning.py`, `t_metadata.py` | Two-tier recognize/reconstruct from `BasinSeed`; substrate side-effects (T-clock EMA, basin drift); per-basin Temporal Stability T |
| FCAS address layer | `fcas.py` | Fractal Cube Address Space (CANON §6) — composes hashing/dynamics/recall_api without touching sacred files |
| Cortex | `cortex.py`, `cortex_scm.py`, `cortex_classifier.py` | L1 graph → L2 settlement CA → L3 classifier |
| Agents / generation | `agent.py`, `decoder.py`, `generation.py`, `crystallization.py` | Ollama tool loop; Wheeler-primary (small model as codec); IT-from-BIT trajectory resonance; offline corpus pre-training |
| Support | `constants.py`, `hardware.py`, `diagnostics.py` | Centralized tunables; device detection; read-only per-tick 5W1H decomposition |

### Subpackages (`wheeler_memory/`)

| Package | Role |
|---------|------|
| `accel/` | GPU/HIP (ROCm) acceleration — batch-dispatch only, lazy `.so` load, falls back to CPU |
| `theories/` | Production topology helpers: `basin.py`, `metrics.py`, `synthesis.py` |
| `npu/` | Stub only (`coral/` placeholder) — no active source yet |

### Domain Chunks

Memories auto-route by keyword: `code`, `science`, `hardware`, `daily_tasks`, `meta`, `general` (fallback). Each chunk has independent index, attractors, bricks, and associations.

## CLI

All 19 commands registered in `pyproject.toml [project.scripts]`. Run `pip install -e .` after adding new entry points. Beyond the store/recall/temps/forget/sleep/agent/info/scrub/crystallize/primary core, the benchmark + eval commands are: `wheeler-bench`, `wheeler-bench-gpu`, `wheeler-baseline`, `wheeler-rerank`, `wheeler-recon-bench`, `wheeler-generate`, `wheeler-mmlu`, `wheeler-scm`, `wheeler-simlex`.

Common flags across commands: `--data-dir`, `--chunk`, `--encoder` (hash|hippocampus|embedding|blended|word), `--salience` (low|medium|high), `--verbose`.

## Autoresearch Protocol

See `docs/program.md` for full details. Summary:

1. Edit only `constants.py` (one or a few params)
2. Commit the change
3. Run `wheeler-bench --commit <hash7> --changed "<param>"`
4. Keep if improved, revert if score worsened >10%

Key parameters: `MAX_PUSH_STRENGTH` (attractor sharpness), `SLOPE_FLOW_STRENGTH` (mixing rate), `SALIENCE_THRESHOLD_MED` (convergence precision), `SALIENCE_MAX_ITERS_MED` (iteration cap).

## Project Tooling (`.claude/`)

- `.claude/skills/run-wheeler-memory/` — project skill to run, smoke-test, or benchmark the CA (`SKILL.md`, `smoke.sh`).
- `.claude/agents/` — repo subagents: `directory-organizer`, `testing-engineer`.
- `.claude/settings.local.json` — currently holds pytest permission allows only (commands run as `.venv/bin/pytest`).
- `ONBOARDING.md` — new-teammate onboarding guide.

## Hooks (recommended, not currently wired)

These PostToolUse hooks are *recommended* for this repo but are **not** presently
configured (no `hooks` key exists in `.claude/settings.json` or `~/.claude/settings.json`).
Set them up via `/update-config` or `/hooks` if you want them active:

1. **ruff format** — auto-format any edited `.py` file (system Python is externally-managed; format via the venv).
2. **pytest** — run the suite when `wheeler_memory/` or `tests/` files change.
3. **wheeler-bench** — run the quality benchmark when `constants.py` changes.

## Task Discipline

For any non-trivial request, break work into discrete tasks using TaskCreate before starting. This prevents scope creep and makes progress visible. Complete each task before moving to the next — do not batch unrelated changes into a single edit pass.

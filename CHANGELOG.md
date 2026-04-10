# Changelog

## Unreleased

### Bug Fixes
- **`build_hip.sh` path fix**: Updated `REPO_DIR` traversal (`../..` instead of `..`) and `GPU_DIR` target (`accel/hip` instead of `gpu/`) to match the accel/ directory migration.

### Features
- **GPU acceleration directory (`accel/`)**: Migrated HIP kernels from `gpu/` to `accel/hip/` with unified Makefile, shared ctypes helpers (`_common.py`), and clean Python bindings (`accel/ca.py`). `gpu_dynamics.py` is now a thin backwards-compatible shim.
- **Batch GPU evolution (`evolve_batch()`)**: New function in `dynamics.py` dispatches multiple frames to GPU in a single kernel launch. Wired into all serial call sites: SimLex-999 (`warm_batch()` pre-evolves ~1028 words), `wheeler-bench` (20 test inputs), and crystallization pipeline.
- **NPU scaffolding (`npu/`)**: Directory structure for Intel NPU (OpenVINO) and future Google Coral Edge TPU. Includes `npu_available()` device detection, `openvino_bridge.py` stub, and `coral/tpu_bridge.py` stub with dual-TPU pipeline support. All stubs raise `NotImplementedError` until hardware integration.
- **Context-window random indexing encoder** (`word_encoder.py`): Distributional semantics via context-window co-occurrence vectors trained on WikiText-103 (1.16M lines, 500K vocab, 384-dim). Available as `--encoder context` across CLI tools.
- **SimLex-999 benchmark** (`wheeler-simlex`): New CLI command for evaluating semantic similarity against the SimLex-999 gold standard. Supports all encoder types with Pearson/Spearman modes.

### Tuning
- **CONTEXT_RI_BLEND 0.5 → 0.9**: Higher blend weight toward context-RI vectors over hippocampus n-grams. SimLex-999 Spearman rho improved from +0.034 to +0.101.
- **Autoresearch infrastructure**: Overnight loop scripts for autonomous parameter sweeps (see `program.md`). 50 iterations logged in `results.tsv` driving quality score from 0.013 → 0.009.

### Organization
- **`accel/` directory**: All accelerator code consolidated under `wheeler_memory/accel/` — HIP kernel sources in `accel/hip/`, Python bindings in `accel/ca.py`, shared helpers in `accel/_common.py`
- **`npu/` directory**: Intel NPU and Google Coral stubs under `wheeler_memory/npu/` with context docs
- **`gpu/` deprecated**: Original `gpu/` directory kept read-only for reference, `CONTEXT.md` updated with migration notice
- **README docs restructure**: Documentation table reorganized into categorized sections (Getting Started, Core Guides, Reference, Project) with suggested reading order
- Moved `CORTEX_CLASSIFIER_SUMMARY.md`, `CORTEX_CLASSIFIER_FILES.txt` to `docs/reports/`
- Moved `VERSION_CHANGES.md` to `docs/`
- Added `.gitignore` patterns for overnight/autoresearch artifacts

### Tests
- Added `tests/test_accel_init.py` (9 tests): Module imports, device detection, shim compatibility for accel/ and npu/
- Added `tests/test_accel_ca.py` (10 tests): Batch evolution correctness (CPU), GPU vs CPU numerical match, `@pytest.mark.gpu` for GPU-specific tests
- Registered `gpu` pytest marker in `pyproject.toml`
- Test count: 757 → 776

## v0.3.1 (2026-04-01)

### Bug Fixes
- **Fixed 0-dominant convergence bug**: Frames with <5% alive cells no longer falsely declare CONVERGED. Added `alive_fraction` gate to all three convergence loops (`evolve_and_interpret`, `evolve_with_params`, trajectory). New DEGENERATE state for 0-dominant attractors.
- Updated decoder tests to mock interference path (matching new default)

### Changes
- **Three-grid interference is now the default recall path**: Flipped `use_interference` default from `False` to `True` in `WheelerAgent` and `WheelerPrimaryAgent`. Gracefully degrades to Pearson-only when no experiential data exists.
- `recall_with_interference` exported from public API (`wheeler_memory.__init__`)
- CLI `wheeler-recall` defaults to interference mode (`--no-interference` for Pearson-only fallback)
- New constants: `ALIVE_THRESHOLD` (0.33), `MIN_ALIVE_FRACTION` (0.05)

## v0.3.0 (2026-03-24)

### Features
- Three-grid interference architecture (Corpus/Experiential/SCM)
- SCM grid with hardening and annealing
- Self-consistency feedback loop
- Parameterized CA dynamics (`evolve_with_params`)
- Experiential-to-corpus consolidation during sleep
- Comprehensive test suite (+238 tests across 13 modules)

## v0.2.0 (2026-03-20)

### Organization
- Reorganized `scripts/` into `bench/`, `exploration/`, `tools/` subdirs
- Moved reports to `docs/reports/`, HTML demos to `docs/demos/`
- Removed `open_webui_setup/` (LLM infra, moved out of repo)
- Removed `wheeler_3d_viewer/` (no longer maintained)
- Added `docs/VISION.md` — Project Ralph architecture vision

### Features
- `wheeler-mmlu --mode learn` — full learn → consolidate → test cycle
- `wheeler-generate` — IT-from-BIT generative text engine
- `tests/test_hallucination.py` — hallucination classification tests
- Crystallization pipeline refactor (pipelined embed→CA→store)
- New planning docs committed (fractal cube addressing, ternary dynamics)

### Tuning
- Parameter experiments on SALIENCE_THRESHOLD_MED, constants updates

## v0.1.0 (Initial Release)

### Features

- **Cellular automata memory system** — 64x64 CA grid with 3-state dynamics, ~40-50 tick convergence
- **Chunked storage** — domain-specific routing (code, hardware, daily_tasks, science, meta, general) via keyword matching
- **Temperature dynamics** — access frequency + time decay (7-day half-life), hot/warm/cold/fading/dead tiers
- **Reconstructive recall (Darman)** — stored attractors blend with query context and re-evolve through CA
- **Sentence embedding** — `all-MiniLM-L6-v2` with Johnson-Lindenstrauss random projection to 64x64
- **Attention model** — salience-driven variable tick rates (low/medium/high CA budget)
- **Sleep consolidation** — temperature-tiered frame pruning within memory bricks
- **Eviction / forgetting** — graceful memory degradation with fade, evict, and capacity phases
- **Associative warming** — 2-hop spreading activation with fast-decay warmth
- **Oscillation detection** — detect and classify oscillating CA states
- **Cell polarity tracking** — polarity-based attractor avoidance
- **GPU acceleration** — HIP/ROCm kernel for AMD GPUs (CUDA compatible)
- **Wheeler-agent** — LLM chat agent with Wheeler context via Ollama (qwen3)
- **Wheeler-primary** — small model (Qwen 2.5-1.5B) as pure language renderer for Wheeler's attractor state
- **Corpus crystallization** — offline pre-training pipeline (JSONL, CSV, TXT, Parquet)
- **Web dashboard** — browser-based memory browser, chat interface, and interactive CA demo
- **Theory experiments** — basin analysis, Lichtenberg patterns, resonance, structured synthesis

### CLI Tools

- `wheeler-store`, `wheeler-recall`, `wheeler-forget`, `wheeler-temps`, `wheeler-sleep`
- `wheeler-agent`, `wheeler-primary`, `wheeler-crystallize`
- `wheeler-ui`, `wheeler-scrub`, `wheeler-info`, `wheeler-bench-gpu`

### Evaluation

- Semantic apple test (holdout topology validation)
- Decoder confidence gradient analysis
- Co-activation topology mapping
- Bridge sentence experiments

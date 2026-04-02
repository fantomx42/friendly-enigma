# Changelog

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

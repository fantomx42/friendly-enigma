# Version Changes

## v0.2.0 — 2026-03-20

- Reorganized `scripts/` into `bench/`, `exploration/`, `tools/`, `experiments/` subdirs
- Moved reports to `docs/reports/`, HTML demos to `docs/demos/`
- Removed `open_webui_setup/` and `wheeler_3d_viewer/` (no longer maintained)
- Added `docs/VISION.md` — Project Ralph architecture vision
- `wheeler-mmlu --mode learn` — full learn -> consolidate -> test cycle
- `wheeler-generate` — IT-from-BIT generative text engine
- Hallucination classification tests (`tests/test_hallucination.py`)
- Crystallization pipeline refactor (pipelined embed -> CA -> store)
- New planning docs: fractal cube addressing, ternary dynamics
- Parameter experiments on SALIENCE_THRESHOLD_MED, constants updates
- Cortex 3-tier system (L1 Graph, L2 Settlement CA, L3 Native Classifier) — replaces MiniLM dependency
- Word encoder with learned co-occurrence vectors (SVD on PMI matrix)
- Hippo-word blended encoder mode for MMLU benchmarking
- Autoresearch overnight loop infrastructure (`overnight_loop.py`, `run_autoresearch_loop.py`, `program.md`)
- MMLU baseline established: 27.5% on physics (488 questions), random chance = 25.0%

## v0.1.0 — Initial Release

- Cellular automata memory system — 64x64 CA grid with 3-state dynamics, ~40-50 tick convergence
- Chunked storage — domain-specific routing (code, hardware, daily_tasks, science, meta, general)
- Temperature dynamics — access frequency + time decay (7-day half-life)
- Reconstructive recall (Darman) — stored attractors blend with query context and re-evolve
- Sentence embedding — all-MiniLM-L6-v2 with JL random projection to 64x64
- Attention model — salience-driven variable tick rates
- Sleep consolidation — temperature-tiered frame pruning
- Eviction / forgetting — graceful memory degradation (fade, evict, capacity phases)
- Associative warming — 2-hop spreading activation with fast-decay warmth
- Oscillation detection — detect and classify oscillating CA states
- Cell polarity tracking — polarity-based attractor avoidance
- GPU acceleration — HIP/ROCm kernel for AMD GPUs (CUDA compatible)
- Wheeler-agent — LLM chat agent with Wheeler context via Ollama
- Wheeler-primary — small model as pure language renderer for attractor state
- Corpus crystallization — offline pre-training pipeline (JSONL, CSV, TXT, Parquet)
- Web dashboard — browser-based memory browser, chat interface, interactive CA demo
- Theory experiments — basin analysis, Lichtenberg patterns, resonance, structured synthesis
- CLI: wheeler-store, wheeler-recall, wheeler-forget, wheeler-temps, wheeler-sleep, wheeler-agent, wheeler-primary, wheeler-crystallize, wheeler-ui, wheeler-scrub, wheeler-info, wheeler-bench-gpu

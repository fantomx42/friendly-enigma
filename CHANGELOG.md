# Changelog

## v0.3.6 (2026-04-28)

### Two-Tier Recall API

- **Recognition / reconstruction split** (`wheeler_memory/recall_api.py`): `recognize(query)` performs a single-pass Pearson match against stored attractors with no CA convergence loop on the query frame — returns a `BasinSeed` if max similarity ≥ `RECOGNITION_THRESHOLD`, else `None`. `recognize_top_k(query, k)` returns up to k seeds. `reconstruct_from_seed(seed, query=None|str, alpha=0.3)` warm-starts CA from the stored attractor: `query=None` returns it as-is (ticks=0); a query string blends stored + raw query frame and re-evolves. Wraps `recall_memory()` from `storage.py` (sacred file, untouched).
- **Public re-exports** (`wheeler_memory/__init__.py`): `recognize`, `recognize_top_k`, `reconstruct_from_seed` exposed at package level.
- **Default path UNCHANGED**: `wheeler-recall` still uses three-grid interference. The new path is opt-in via `--recognize` flag (CLI) or direct API import.

### Per-Basin Temporal Stability (T)

- **New persistent state** (`wheeler_memory/t_metadata.py`): Each chunk's `index.json` now carries a per-basin `T` field. Fresh basins start at T=0 (fully plastic) and earn rigidity through repeated stable recalls. EMA update applied on `recognize()` when learning is enabled.
- **Drift rate**: Stored attractor drifts toward the observed pattern at rate `(1 - T) * BASIN_DRIFT_BASE_RATE` (default 0.02). Mature basins (T → 1) become near-rigid; fresh basins absorb input rapidly.
- **CLI flag**: `wheeler-recall --learn` enables T accumulation + drift. Off by default.
- **Trajectory** (5 consecutive recalls of the same query against a fresh basin): T 0 → 0.10 → 0.19 → 0.27 → 0.34 → 0.40, asymptoting toward observed_stability ~0.97-1.00.

### `_basin_stability` Replacement

- **Finding**: The original implementation called `cortex_scm.score_energy` for the per-basin stability signal. For converged attractors `score_energy` saturates to 0, which produced a degenerate T update — every recall reported stability ≈ 0 regardless of match quality, blocking T accumulation entirely.
- **Fix** (`recall_api.py`): Replaced with `1 - p99(|delta|) / 2`, mirroring the percentile-based metric used by CA convergence detection. Numerically stable on converged frames; matches observed signal.
- **D7 compliance**: `cortex_scm.py` itself was not modified — variable definitions in the SCM module are unchanged. The replacement lives in the calling site.

### New Constants

- `wheeler_memory/constants.py`: `RECOGNITION_THRESHOLD` (0.45), `T_INIT_DEFAULT` (0.0), `T_EMA_RATE` (0.1), `BASIN_DRIFT_BASE_RATE` (0.02). Four entries added.

### Caller Migration

- **`scripts/scm_ab_eval.py`**: Migrated from `recall_memory` to `recognize_top_k` for ranking arms. Behaviour-equivalent at default `k`.
- **`wheeler_memory/theories/structured.py`**: Migrated to `recognize_top_k`. Theory experiments unaffected.
- **`scripts/wheeler_recall.py`**: Added `--recognize` and `--learn` flags. Default behaviour unchanged.
- **Deferred sites** recorded in `plans/recall_migration_audit.csv` — call sites that retain the legacy path pending behaviour review (not bugs; intentional defer).

### Benchmark

- **`scripts/bench/bench_recall_warm_vs_cold.py`**: Three input-distance bands (near / mid / far from stored attractor). Compares warm-start (reconstruct from seed) against cold-start (full CA from query frame).
- **Result**: ~2x ticks reduction warm-vs-cold across all three distance bands. Quality (final-frame correlation to ground truth) within noise of cold path.

### Tests

- Added `tests/test_recall_api.py` (6 tests): `recognize` returns top-1 with stability, `recognize_top_k` ordering, `reconstruct_from_seed` converges from stored attractor, T accumulates under `--learn`, T stays at 0 without learning, drift respects `(1-T)` damping.
- Full suite: 758 passing, no regressions.

### Out of Scope

- Sleep Pass / drift consolidation deferred. T currently only updates on recall; there is no offline pass that consolidates accumulated T into the stored corpus state. Tracked for a later release.

## v0.3.5 (2026-04-27)

### SCM Cold-Start Spatial Alignment Test + Paraphrase A/B Rewrite

- **`test_cold_start_spatial_alignment`** added to `TestSCMGridRecallFeedback` (`tests/test_scm_grid.py`): Verifies that cold-start seeding (scm_grid.py:204-208) fires under negative advantage, seeds at least one cell, and aligns spatially to the top-quartile credit region. Setup: `kappa_base=0.8`, `kappa=0.2` (advantage = −0.6), top 32 rows of corpus/experiential attractors set to 0.8 (credit = 0.64), bottom 32 zero. Asserts: `seeded.any()`, all seeded values `>= SCM_HARDENING_FLOOR`, and `seeded == (credit >= p75_threshold)` (exact spatial match). Tighter than the existing `test_cold_start_negative_advantage_seeds_grid` — this one checks WHERE cells are seeded, not just that some are.
- **`scripts/scm_ab_eval.py` complete rewrite**: Two-phase evaluation design (warmup + paraphrase eval). Phase 1 runs 50 exact Q-part queries through `learning_scm.update_from_recall` to settle `kappa_base` to ~1.22 via EMA (rate=0.1) before eval begins. Phase 2 runs 50 paraphrase queries across all three arms. Warmup phase settles `kappa_base` so paraphrase kappas (0.45–0.94) produce negative advantage, bypassing the homeostasis ceiling and triggering cold-start seeding. Added `_paraphrase(text, rng)`: strips stop words, shuffles remaining content words. Character n-gram encoder (hippocampus) is sensitive to word-order and boundary changes — shuffling reduces Pearson similarity without requiring new corpus. Added `--no-warmup` flag. JSONL fields extended: `warmup_kappa_base`, `paraphrase_query`, `exact_query`, `kappa_base_before`.

### Architectural Finding: interference_score Collapses SCM to Global Scalar

- **Root cause of frozen/learning rank equivalence diagnosed**: `interference.py:158-161` computes `mean_openness = float((1.0 - np.abs(scm_grid)).mean())` and returns `score = (c_sim + e_sim) * mean_openness`. The openness multiplier is a **global scalar** — identical for every candidate in a given query regardless of the spatial SCM pattern. This makes rank ordering provably identical between frozen and learning arms for any query, no matter how much the SCM is seeded or how different the two grids are. The spatial 64×64 trust topology is irrelevant to candidate ranking under the current formulation.
- **Paraphrase A/B result**: Warmup correctly settles `kappa_base=1.2155`. Paraphrase kappas (0.45–0.94) produce negative advantage → seeding fires (confirmed by score ratio 0.677/0.696 = 0.974, implying mean|SCM| ≈ 0.026). But all three arms still produce R@1=1.000 and identical rank orderings. SCM openness reports 1.0000 throughout because seeded values (~0.001) are well below the 0.3 alive threshold — a misleading diagnostic.
- **Fix required for diagnostic A/B**: `interference_score` must use spatial product instead of global scalar: `score = mean((q_corpus * s_corpus) * (1-|SCM|)) + mean((q_exp * s_exp) * (1-|SCM|))`. This changes score semantics from Pearson r (normalized) to weighted mean (unnormalized) — calibration relative to existing recall paths needs thought before merging.

## v0.3.4 (2026-04-26)

### SCM Telemetry + Gradient Observability

- **Per-step JSONL telemetry** (`scm_telemetry.jsonl`): Every call to `SCMGrid.update()` or `SCMGrid.update_from_recall()` appends one row with `step`, `source`, `grad_mag_mean`, `grad_mag_max`, `scm_entropy` (20-bin histogram), `attractor_count` (BFS connected components on |SCM|>0.1), and `alive_fraction` (|SCM|>ALIVE_THRESHOLD). Telemetry path defaults to `<data_dir>/scm_telemetry.jsonl`; injectable via `SCMGrid.load_or_create(telemetry_path=...)`.
- **Shared monotonic step counter** (`_step_count`): Both update paths increment the same counter, giving a global ordering of grid-modifying events in the JSONL stream.
- **No-op paths still emit**: Empty-mask `update()` and homeostasis-ceiling-blocked `update_from_recall()` both emit a zero-delta row (grad_mag_max=0.0) so the stream is continuous.
- **Telemetry is fault-tolerant**: Write errors are silently caught — telemetry never crashes the engine.

### Gradient Sanity Test

- **`tests/test_scm_gradient_direction.py`**: One-step sanity check that `update_from_recall` moves the SCM toward its analytical fixed point under positive advantage. Uses S_perturbed = 0.5·sign(rng) + 0.1·noise to ensure openness < SCM_OPEN_FRACTION_CEIL (avoids homeostasis guard) and S_good = ε_floor·sign(S_perturbed) as the true attractor. Must pass before the closed-loop A/B is valid.

### Closed-Loop A/B Evaluation

- **`scripts/scm_ab_eval.py`**: 50-passage closed-loop A/B comparing three recall arms — Pearson baseline, frozen SCM (zeros, never updated), and learning SCM (accumulates from each recall outcome). Corpus from `datasets/arc.jsonl`, hippocampus encoder, Q-prefix queries.
- **Experiential storage**: Each passage stored in both corpus and experiential grids so the spatial answer equation `Answer(i,j) = Corpus(i,j) * Experiential(i,j) * (1 - |SCM(i,j)|)` activates. Without experiential, frozen/learning arms collapse to scalar Pearson-identical ranking.
- **Bug fix**: `_store_experiential` previously called `store_memory(..., grid='experiential')` which overwrote the corpus index entry with `"grid": "experiential"` — causing `recall_memory` to skip every passage (line 305-306 of storage.py skips experiential-tagged entries). Fixed by writing the experiential npy directly without touching `index.json`.
- **Metrics**: Recall@1, Recall@3, MRR, mean_top1_score, mean_correct_score, mean_score_gap, SCM openness trajectory. JSONL output to `results/scm_ab_eval_<timestamp>.jsonl`.
- **Design finding**: A fresh all-zeros SCM cannot be differentiated by `update_from_recall` alone — the `sign(M)` gate means only cells with existing opinions are adjusted. The learning SCM requires self-consistency feedback (`update()`) to seed initial opinions before recall-driven gradient can tune them.

### Tests

- Added `tests/test_scm_gradient_direction.py` (1 test): gradient direction sanity check
- Added `TestSCMTelemetry` class to `tests/test_scm_grid.py` (4 tests): self_consistency emission, recall_gradient emission, shared step counter, no-op path emission

## v0.3.3 (2026-04-13)

### Semantic Improvements
- **SimLex-999 rho +0.101 → +0.255 (+152%)**: Combined effect of decontamination tuning and richer training corpus. Now at 57% of MiniLM's ceiling (0.446) with zero external models.
- **Decontamination sweep**: Swept CONTEXT_RI_REMOVE_TOP_K across [0,1,2,3,4,5,6,8]. K=4 is the sweet spot — components 3-4 capture residual frequency bias without stripping semantic signal. Rho +0.220 → +0.227 (+3.1%).
- **OpenWebText corpus**: Added 500M-word subsample from OpenWebText (609K docs) to complement WikiText-103 (101M words). Combined 601M-word corpus improves noun rho by +18% and adjective rho by +18%.

### Bug Fixes
- **CLI default mismatch**: `wheeler_learn_words.py --remove-top-k` defaulted to 1, while `constants.py` specified 2. CLI now imports defaults from constants (single source of truth). Also fixed `--subsample-t` default.

### New Files
- `datasets/download_openwebtext.py` — Streaming HuggingFace download with word-count cap and deterministic seed
- `scripts/sweep_decontamination.sh` — Reusable sweep script for singular component removal tuning
- `sweep_decontamination_results.tsv` — Full sweep results (K vs rho)

## v0.3.2 (2026-04-09)

### Bug Fixes
- **Apple test encoder mismatch**: `apple_test_semantic.py` stored with hippocampus but queried with blended encoder — Pearson correlation was noise. Fixed to use hippocampus on both sides ("it from bit" — no external models). ML architecture now reaches WEAK_TOPOLOGY (0.173) honestly.
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

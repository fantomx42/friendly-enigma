# Changelog

## Documentation audit (2026-05-06)

Documentation-only pass. No code changes. Aligns user-facing docs with
`CANON.md` (the 2026-05-04 audit pass that became architectural
source-of-truth via the canon-precedence rule in `CLAUDE.md`).

### README

- Rewritten to canon's sober, status-tagged style
  (`[BUILT]` / `[DESIGNED]` / `[PARTIAL]` / `[OPEN]` / `[SPECULATIVE]` /
  `[ACTIVE RESEARCH]`).
- Canon pointer added at the top with explicit precedence rule.
- **GPU repositioned as batch acceleration only** (canon §1.4). HIP kernels
  accelerate crystallization and SimLex sweeps, not the recall CA. The 71×
  batch=1000 number now lives in a dedicated "Acceleration" section, properly
  contextualized.
- **MMLU framed as `[CHANCE FLOOR]`** per canon §8.2 — corpus-limited, not
  architecture-limited. Treat MMLU as a corpus-health metric, not a
  recall-quality metric.
- **SCM Map vs Measure** disambiguated explicitly per canon §3.5.1
  (`scm_grid.py` is the trust-topology Map; `cortex_scm.py` is the scoring
  Measure).
- **Encoder layer** reframed as `[ACTIVE RESEARCH]` per canon §1.2: primary
  surface (hash, hippocampus, embedding, blended) vs. research variants
  (in `scripts/wheeler_simlex.py:60` `ALL_ENCODERS`).
- **FCAS introduced** per canon §6 — (hash, depth) tuple addresses, SHA256
  triple-role, build status table.
- **Naming history** acknowledged: Ralph → Darman → Wheeler Memory (canon §14.1).
- **Open work** section mirrors canon §9 priorities.
- Stale claims fixed: removed `gpu/` from project structure (directory gone
  since v0.3.6 cleanup); test count 43 (was 44 — drift since cleanup);
  `notes/` (`exploration/`, `experiments/`, `theories/`) added as first-class;
  SimLex now cites canon's "ρ ≈ 0.22–0.26 and climbing" range rather than a
  pinned `+0.255`.
- Acknowledged the known `interference_score` global-scalar collapse issue.
- Dropped redundant "Suggested Reading Order" (lives in `docs/INDEX.md`).

### Other docs

- `CLAUDE.md` module map: removed deleted `gpu_dynamics.py` reference
  (CA Engine row).
- `docs/INDEX.md`: added "Canon" section pointing at `CANON.md` as the first
  thing readers should open.
- `docs/architecture.md`: SCM table now records both write paths
  (self-consistency erosion + recall-driven κ) per canon §3.3.1; added
  §3.3.5 closure note ("sleeping giant resolved"); GPU Backend section
  reframed as batch-acceleration-only per canon §1.4; SCM Map vs Measure
  callout in Cortex section; removed deleted `gpu/` directory from module
  tree.
- `docs/future.md`: now leads with canon §9 priorities (FCAS resolution,
  Wheeler-native eval, corpus population, cross-cube interference). GPU
  roadmap retained but properly framed as batch acceleration. Known
  `interference_score` issue surfaced.

## v0.3.6 — Cleanup pass (2026-04-29)

A separate cleanup commit on top of the v0.3.6 release. No behaviour change to the running system; everything tested green (775 tests passing, down from 808 only because three theory test files moved to the archive — the production paths still test the same code).

### Removed

- **`wheeler-ui` CLI and `scripts/wheeler_ui.py`**: orphaned since 2026-03-20 (its `UI_FILE` / `CHAT_FILE` paths pointed at a `ui/` directory that had been moved to `docs/demos/`). The script aborted on a missing-file check; nothing was rescuing it. Removed from `pyproject.toml` `[project.scripts]`. Static demo HTML remains under `docs/demos/`.
- **`wheeler_memory/gpu/` directory**: deprecated since 2026-04-08 (superseded by `wheeler_memory/accel/hip/`). Stale duplicate `.so` binaries and `.hip` sources removed. Five live doc/script references re-pointed at the active path.
- **`wheeler_memory/gpu_dynamics.py` shim**: only callers were two internal imports plus a single test that existed solely to verify the shim. Internal imports switched to `from .accel.ca import ...`; shim and shim-only test deleted.
- **5 dead constants in `wheeler_memory/constants.py`**: `CORTEX_CLASSIFIER_LR`, `CORTEX_CLASSIFIER_PATH`, `TRAJECTORY_CURVE_LEN`, `RECALL_ENCODER`, `EXPERIENTIAL_HIT_SATURATION` (zero callers in production code).

### Reorganised

- **`scripts/exploration/` → `notes/exploration/`** (9 research-notebook scripts; not pytest, not CLI entry points). The four scripts that produce diagrams in `docs/assets/reports/` are still invoked manually as `python notes/exploration/test_diversity*.py --output ...`; updated `docs/assets/README.md` accordingly.
- **`scripts/experiments/` → `notes/experiments/`** (6 per-theory exercises that pair with the theory modules).
- **3 theory modules archived**: `wheeler_memory/theories/lichtenberg.py`, `resonance.py`, `structured.py` → `notes/theories/`. Their pytest suites moved with them to `notes/theories/tests/`. The `__init__.py` was rewritten to drop the archived imports.
- **Production-supporting theories stayed live**: `wheeler_memory/theories/basin.py`, `metrics.py`, `synthesis.py` remain in the package because production code (agent, decoder, wheeler_mmlu, the apple-test benchmark) imports them. Their tests stayed under `tests/`.
- **`results/archive/`**: 6 March-2026 MMLU logs + 5 older `scm_ab_eval` JSONLs moved out of the active `results/` directory; latest run artifacts kept at the top level. `BASELINES.md` (the textual record) stays live.
- **`plans/archive/`**: all 5 March-2026 plan markdowns + the `planned-theories-test/` directory archived; `plans/recall_migration_audit.csv` (the live v0.3.6 deliverable) stays. New `plans/README.md` describes active vs archived.

### Documentation

- **Pitch pack rewritten for v0.3.6**: `BLUEPRINT.md`, `one_pager/darman_one_pager.md`, `slides/01_investor_pitch.md`, `slides/02_developer_pitch.md`, `slides/03_general_pitch.md`, and `demo_script/demo_script.md` fully rewritten. The 1324-line `whitepaper/wheeler_memory_whitepaper.md` was updated section by section: new abstract; revised contributions list with two-tier recall + per-basin Temporal Stability + three-grid interference + Context-RI as headline architectural moves; expanded §5 with five new validation subsections (warm-vs-cold benchmark, T trajectory, SimLex-999, Apple Test, MMLU). The previous "Symbolic Collapse Model" framing was renamed to "convergence-as-meaning" so the abbreviation **SCM** is reserved for the **Structural Coherence Map** (the trust-topology grid in three-grid interference).
- **Honest numbers**: pitch pack and README now cite 44 modules / 775 tests / 16 CLI commands; ~3 ms/tick CPU, 71× GPU speedup at batch=1000; MMLU all-57 cortex 24.3% / cortex+facts 25.3% / cortex+L3 25.9% (chance 25%); Context-RI ρ=+0.255 (57% of MiniLM ceiling); two-tier ~2× tick reduction. The L3-at-chance result is named honestly in every audience-facing file rather than buried.
- **Cross-doc consistency**: `README.md`, `docs/architecture.md`, `docs/cli.md`, `docs/install.md`, `docs/gpu.md`, `docs/assets/README.md`, `docs/demos/README.md`, `wheeler_memory/CONTEXT.md`, `scripts/CONTEXT.md` all updated for the post-cleanup project shape.
- **`scripts/wheeler_learn_words.py` docstring**: previously claimed a `wheeler-learn-words` CLI that was never registered. Updated to say `python -m scripts.wheeler_learn_words` to match reality.

### Verification

- `pytest -m "not slow"` → 729 passed + 46 deselected (775 total), no regressions.
- All `wheeler-ui` mentions outside historical version-history disclosures are gone.
- `pyproject.toml` version bumped 0.3.5 → 0.3.6 to match the CHANGELOG.

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

### Removed

- **`wheeler-ui` CLI and `scripts/wheeler_ui.py`**: The web dashboard entry point was already broken — `UI_FILE` and `CHAT_FILE` pointed at a `ui/` directory that was relocated to `docs/demos/` in March 2026 (commit `cadab70c`), so any invocation of `wheeler-ui` aborted immediately on a missing-file check. The orphaned script and its `pyproject.toml` entry have been removed; static demo HTML remains under `docs/demos/`. If a live dashboard is wanted again, it should be a fresh implementation against the current `recall_api` / `storage` surfaces.

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

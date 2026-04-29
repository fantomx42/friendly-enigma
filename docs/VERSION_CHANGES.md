# Version Changes

## v0.3.6 — 2026-04-28

Two-tier recall API (`recognize` / `reconstruct_from_seed`), per-basin Temporal Stability (T) with T-gated drift, and the `_basin_stability` p99 fix.

- **Two-tier recall API** (`wheeler_memory/recall_api.py`): new public surface that splits identity from content. `recognize(query)` (line 320) does a Pearson scan against stored attractors using the **raw** query frame and returns a `BasinSeed` (line 62) if max similarity reaches `RECOGNITION_THRESHOLD=0.45`, else `None`. Never calls `evolve_and_interpret` — one constant `apply_ca_dynamics` step probes the winning basin for the stability reading. `recognize_top_k(query, k)` (line 368) returns up to k seeds. `reconstruct(seed, query=None|str, alpha=0.3)` (line 414) is the warm-start content path: with `query=None` returns stored attractor unchanged (`ticks=0`), with a query string blends stored + raw query frame (no query-evolve!) and re-evolves through `evolve_and_interpret`. Module wraps `storage.py` from outside — `storage.py` is sacred and was not modified. `__init__.py:118` re-exports the new API; `recall_api.reconstruct` is exposed as `reconstruct_from_seed` to avoid clashing with `wheeler_memory.reconstruction.reconstruct`.
- **Per-basin Temporal Stability (T)** (`wheeler_memory/t_metadata.py`): NEW persistent state in each chunk's `index.json` under `metadata.t_stability` and `metadata.t_recall_count`. `ensure_t_fields` (line 13) backfills lazily on read; `update_t_stability` (line 27) is the EMA: `T_new = (1-r) * T_old + r * observed`. `T_INIT_DEFAULT=0.0` — basins earn rigidity through repeated stable recalls rather than starting rigid. `set_learning_enabled(bool)` (recall_api.py:52) toggles the module-level default; per-call `learning_enabled` arg takes precedence. When True, recognize() updates T via EMA and drifts the stored attractor in place: `new = stored + (1-T) * BASIN_DRIFT_BASE_RATE * (observed - stored)` where `observed = apply_ca_dynamics(query_frame)`. All writes happen under `chunks/<chunk>/index.json.lock` with `fcntl.LOCK_EX`. 4 new constants in `constants.py:195`: `RECOGNITION_THRESHOLD=0.45`, `T_INIT_DEFAULT=0.0`, `T_EMA_RATE=0.1`, `BASIN_DRIFT_BASE_RATE=0.02`.
- **`_basin_stability` p99 fix** (`wheeler_memory/recall_api.py:139`): initial design used `cortex_scm.score_energy(stored, one_step(stored))` for the per-recall stability reading, but `score_energy` reports `1 - max(|delta|)` clipped to `[0, 1]`. Converged attractors with sharp ±1 boundaries flip a few cells per CA step → max_delta=2 → score_energy=0; T could not accumulate organically. Replaced with `_basin_stability(stored, one_step) = 1 - p99(|delta|) / 2`, which mirrors the CA's own convergence detection (`CONVERGENCE_PERCENTILE=99.0` in constants.py). After the fix, both hash and hippocampus encoders show clean T accumulation across 5 recalls: `0 → 0.10 → 0.19 → 0.27 → 0.34 → 0.40`, asymptoting toward `observed_stability ~ 0.97-1.00`.
- **Migration audit** (`plans/recall_migration_audit.csv`): catalogued 36 call sites of `recall_memory` / `recall_with_interference`. Migrated as RECOGNITION_ONLY: `wheeler_memory/theories/structured.py:61` (basin-width measurement uses `measure_basin_width` on the file, not a re-evolved attractor), `scripts/scm_ab_eval.py:301` and `:342` (warmup_hits and pearson_hits — already used `readonly=True`). `wheeler_memory/agent.py`, `decoder.py`, `interference.py` and the rest stay on `recall_memory` (RECONSTRUCTION). CLI: `scripts/wheeler_recall.py` adds `--recognize` (fast path) and `--learn` (drift on) flags; defaults unchanged.
- **Warm-vs-cold benchmark** (`scripts/bench/bench_recall_warm_vs_cold.py`): three input-distance band benchmark (exact, near-paraphrase, far). ~2× ticks reduction across all bands. Recognition rate at production threshold (0.45) is encoder-dependent — for the synthetic hash-encoded fixture queries, recognition rate is 0%, which is honest reporting (those queries legitimately need cold reconstruction).
- **Tests** (`tests/test_recall_api.py`): 6 tests including a deliberately-seeded T trajectory test that asserts higher T → lower drift on a fresh basin, and a monkeypatch test that asserts `evolve_and_interpret` is never called from `recognize()`. 758 tests pass with no regressions.

**Out of scope** (deferred or unrelated): Sleep Pass / consolidation drift accumulation; encoder changes; slot/5W1H reconstruction; SCM variable definitions; three-grid coupling formulas. T is read on every recognition but only written when `learning_enabled=True`.

## v0.3.5 — 2026-04-27

Paraphrase A/B rewrite, cold-start spatial alignment test, and architectural diagnosis of global-scalar interference.

- **Cold-start spatial alignment test** (`tests/test_scm_grid.py`): `test_cold_start_spatial_alignment` added to `TestSCMGridRecallFeedback`. Pre-settles `kappa_base=0.8`, passes `kappa=0.2` (advantage = −0.6). Uses non-uniform attractors (top 32 rows = 0.8, bottom 32 = 0.0) to produce a credit pattern with a clear spatial structure. Asserts: seeding fires, all seeded values `>= SCM_HARDENING_FLOOR`, and seeded cells exactly match the credit-≥-p75 mask (top 32 rows only). Tighter than the existing negative-advantage test — validates spatial precision of the seeding path, not just presence.
- **Paraphrase A/B rewrite** (`scripts/scm_ab_eval.py`): Two-phase design. Phase 1 (warmup): 50 exact Q-part queries through `learning_scm.update_from_recall`; settles `kappa_base` to ~1.22 via EMA(rate=0.1). Phase 2 (eval): 50 content-word-shuffled paraphrase queries across all three arms. `_paraphrase()` strips stop words and shuffles remaining content words — word-order change degrades character n-gram Pearson without new corpus. Warmup ensures paraphrase kappas (0.45–0.94) fall below `kappa_base`, producing negative advantage and triggering cold-start seeding. Added `--no-warmup` flag and diagnostic JSONL fields (`warmup_kappa_base`, `paraphrase_query`, `kappa_base_before`).
- **Architectural finding: interference_score is a global scalar** (`wheeler_memory/interference.py:158-161`): `mean_openness = (1 - |SCM|).mean()` is a single float applied uniformly to every candidate's score in a query. Rank ordering between frozen and learning arms is provably identical regardless of SCM state. The 64×64 spatial trust topology currently has no effect on which candidate ranks first. Paraphrase A/B confirms: score ratio 0.974 (learning vs. frozen) proves SCM was seeded (mean|SCM| ≈ 0.026), but R@1=1.000 and rank ordering are unchanged across all arms. Fix: replace scalar mean_openness with spatial product `mean((q * s) * (1-|SCM|))` per candidate.

## v0.3.4 — 2026-04-26

SCM observability layer and closed-loop A/B evaluation infrastructure.

- **JSONL telemetry on every SCM grid-modifying event** (`scm_telemetry.jsonl`): `SCMGrid.update()` and `SCMGrid.update_from_recall()` both append a row with 7 fields: `step` (monotonic per-process counter), `source` (`"self_consistency"` or `"recall_gradient"`), `grad_mag_mean`, `grad_mag_max`, `scm_entropy` (20-bin entropy), `attractor_count` (BFS connected-components on |SCM|>0.1 via `dynamics._count_clusters`), `alive_fraction` (|SCM|>0.33). No-op paths (empty mask, homeostasis ceiling skip) emit a zero-delta row so the stream is gapless. Telemetry is fault-tolerant — OSError is silently swallowed.
- **Gradient direction sanity test** (`tests/test_scm_gradient_direction.py`): Constructs S_perturbed = 0.5·sign(rng) + 0.1·noise (high |M|, openness well below 0.95 ceiling), computes S_good = ε_floor·sign(S_perturbed) (analytical fixed point under positive advantage), runs one `update_from_recall` step with kappa=0.9, and asserts `||S_after - S_good|| < ||S_perturbed - S_good||`. This test must pass before the closed-loop A/B is valid.
- **Closed-loop A/B script** (`scripts/scm_ab_eval.py`): 50-passage evaluation of three recall arms over ARC passages. Pearson baseline gets R@1=1.000; interference arms (frozen SCM, learning SCM) confirm the spatial answer equation activates with experiential storage and that interference doesn't degrade recall quality. The learning SCM correctly shows that `update_from_recall` only tunes pre-existing opinions — a fresh all-zeros SCM cannot accumulate state from recall alone.
- **Experiential index bug fixed**: `_store_experiential` previously called `store_memory(..., grid='experiential')` which silently replaced corpus index entries with `"grid": "experiential"`, causing `recall_memory` to skip every stored passage. Fixed by writing the experiential npy directly to `chunks/<chunk>/experiential/<hex>.npy` without touching `index.json`.

## v0.3.1 — 2026-04-01

Activates the three-grid interference architecture as the default recall path and fixes a convergence bug that let 0-dominant attractors poison downstream scoring.

- **0-dominant convergence fix:** Added `alive_fraction` gate (`MIN_ALIVE_FRACTION=0.05`, `ALIVE_THRESHOLD=0.33`) to all convergence checks in `dynamics.py` and `trajectory.py`. Frames with <5% alive cells (|value| > 0.33) are rejected from convergence — they get DEGENERATE state instead of false CONVERGED. Fixes the p99 blind spot where zero-majority grids had zero delta across 99% of cells.
- **Interference as default recall:** `WheelerAgent` and `WheelerPrimaryAgent` now default to `use_interference=True`. `recall_with_interference()` is the live recall path — corpus Pearson recall + experiential re-scoring + SCM gating. Old Pearson-only path is the automatic fallback when no experiential counterpart exists (score = `c_sim * mean_openness`, with fresh SCM openness = 1.0, identical to pure Pearson).
- **Public API:** `recall_with_interference` added to `wheeler_memory.__init__.__all__`
- **CLI:** `wheeler-recall` defaults to interference mode; `--no-interference` disables it
- **SimLex-999 baseline (post-fix):** All native encoders rho ~ 0 (no semantic signal). Embedding encoder rho = +0.43 (proves CA pipeline preserves similarity when fed meaningful vectors).

## v0.3.0 — 2026-03-24

Three-Grid Interference Architecture — transforms Wheeler from content-addressed store into a system with emergent epistemic states.

- **Three-grid answer equation:** `Answer(i,j) = Corpus(i,j) * Experiential(i,j) * (1 - |SCM(i,j)|)`
- **SCM Grid** (`scm_grid.py`) — persistent 64x64 trust topology with hardening. Cells harden over time: `LR / (1 + hardening_count)`. Only written by self-consistency feedback loop.
- **Experiential storage** (`experiential.py`) — episodic memory with temporal context (preceding query, SCM snapshot hash). Loose CA dynamics (push=0.35, slope=0.70). 2-day half-life.
- **Interference engine** (`interference.py`) — pointwise three-grid multiplication with four epistemic states: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED
- **Self-consistency feedback loop** — re-encodes decoder output, checks basin convergence, sculpts SCM. The only writer to the trust topology. "It from bit" applied to epistemology.
- **Parameterized CA dynamics** — `evolve_with_params()` accepts per-call push/slope strengths. Same engine, two regimes (tight corpus vs loose experiential). GPU v2 kernel already supports this.
- **SCM annealing** — 10% hardening decay per sleep cycle to prevent permanent trust scarring
- **Experiential→corpus consolidation** — during sleep, cool experiential memories re-evolve under corpus rules and crystallize into permanent knowledge
- **CLI additions:** `wheeler-scm` (inspect/reset SCM), `--experiential` flag on `wheeler-store`, `--interference` flag on `wheeler-recall`, `--mode learn-interference` on `wheeler-mmlu`
- **Decoder integration** — `format_state()` now includes interference state labels and SCM openness in structured prompt
- **Backward compatible** — all existing attractors default to corpus (ABSORBED state), SCM starts as zeros (fully permissive). As of v0.3.1, `recall_with_interference()` is the default recall path
- 12 new constants in `constants.py` for the three-grid system

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

# wheeler_memory/ — Core Library

This is the main Python package. All modules are imported through `__init__.py`.

## Module Groups

### Encoding Pipeline (Text -> 64x64 Frame)
- **hashing.py** — SHA-256 deterministic encoding. **SACRED — do not modify.**
- **hippocampus.py** — Native n-gram semantic encoder. No pretrained models.
- **embedding.py** — Sentence-transformers + JL random projection. Optional (`.[embed]`).
- **word_encoder.py** — Word-level random indexing with learned co-occurrence vectors (SVD on PMI). Trains from stored corpus. Blended with hippocampus via `WORD_HIPPO_BLEND` in constants.py.

### CA Engine (Frame -> Attractor)
- **dynamics.py** — 3-state CA evolution. Local max -> +1, local min -> -1, slopes flow uphill. Von Neumann (4-neighbor) topology. Dispatches to GPU via `accel.ca` when available. v0.3.0 added `evolve_with_params()` for per-call push/slope injection (corpus vs experiential regimes).
- **accel/ca.py** — HIP/CUDA kernel loader. Calls compiled `.so` in `accel/hip/`. v0.3.0: `gpu_evolve_single()` accepts optional `push_strength`/`slope_strength` kwargs for v2 kernel.
- **oscillation.py** — Detects oscillating CA states that never converge.
- **rotation.py** — Rotation retry to escape bad attractor basins. **SACRED.**

### Storage & Recall
- **storage.py** — Store/recall with chunked Pearson correlation search. **SACRED.**
- **chunking.py** — Domain routing by keyword (code, science, hardware, etc). **SACRED.**
- **brick.py** — Memory brick format. Each brick is a `.npz` archive containing seed frame, keyframes, and attractor.
- **cache.py** — JSON file-based caching layer for expensive computations.

### Cortex System (3-Tier Semantic Scoring)
- **cortex.py** — Orchestration + L1 graph topology (Pearson adjacency, BFS clustering).
- **cortex_scm.py** — L2 Settlement CA. Opinion diffusion on correlation graph until convergence. Soft Constraint Satisfaction.
- **cortex_classifier.py** — L3 Native semantic classifier. Scores choices without external models.

### Memory Lifecycle
- **temperature.py** — Access frequency + time decay (7-day half-life). Tiers: hot/warm/cold/fading/dead.
- **warming.py** — 2-hop spreading activation. Fast-decay warmth primes associated memories.
- **consolidation.py** — Sleep consolidation. 3 phases: (1) brick pruning, (2) experiential→corpus re-projection, (3) SCM annealing (10% hardening decay per sleep).
- **eviction.py** — 3-phase graceful degradation when over capacity.
- **attention.py** — Salience-driven variable tick rates (low/med/high CA budget).

### Agents & Rendering
- **agent.py** — LLM chat agent wrapper. Seasons Ollama responses with Wheeler context.
- **decoder.py** — Language Wheeler / Wheeler-primary. Small model as pure decoder for CA state.
- **generation.py** — IT-from-BIT generative engine. Trajectory resonance.
- **language_wheeler.py** — Language-level Wheeler encoding.

### Three-Grid Interference (v0.3.0)
- **scm_grid.py** — SCM (Structural Coherence Map): persistent 64x64 float32 grid in [-1,1] + uint32 hardening counts. Trust topology — where interference is permitted. `load_or_create()`, `update(mask, direction)`, `gap_mask()`, `anneal()`, `stats()`. Persisted as `scm_grid.npy` + `scm_hardening.npy` with atomic save. Only the self-consistency loop writes to it.
- **experiential.py** — Episodic memory encoding. `ExperientialMeta` dataclass bundles temporal context (timestamp, preceding query hex, SCM snapshot hash). Stored under `chunks/{domain}/experiential/`. Loose CA dynamics (push=0.35, slope=0.70).
- **interference.py** — Three-grid interference engine. `compute_interference(corpus, experiential, scm)` → pointwise `C * E * (1 - |S|)`. Four states: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED. `self_consistency_check()` re-encodes text → re-evolves → Pearson against original → writes to SCM. `interference_score()` for ranked retrieval.

### Other
- **constants.py** — ALL tunable parameters. **Only file modified during autoresearch.** See `docs/program.md`. The three-grid architecture (corpus/experiential/SCM) introduced its own constant groups: `CORPUS_MAX_PUSH`, `CORPUS_SLOPE_FLOW`, `EXPERIENTIAL_MAX_PUSH`, `EXPERIENTIAL_SLOPE_FLOW`, `EXPERIENTIAL_HALF_LIFE_DAYS`, `SCM_LEARNING_RATE`, `SCM_HARDENING_FLOOR`, `SCM_GAP_THRESHOLD`, `SCM_ANNEAL_RATE`, `INTERFERENCE_PEAK_THRESHOLD`. v0.3.6 added the two-tier recall constants: `RECOGNITION_THRESHOLD`, `T_INIT_DEFAULT`, `T_EMA_RATE`, `BASIN_DRIFT_BASE_RATE`.
- **reconstruction.py** — Reconstructive recall (Darman). Blend stored attractor with query, re-evolve.
- **polarity.py** — Dual-polarity encoding (antipodal CA states).
- **trajectory.py** / **trajectory_cache.py** — Trajectory similarity for hybrid retrieval.
- **hardware.py** — System hardware detection.
- **crystallization.py** — Corpus pre-training pipeline (JSONL, CSV, TXT, Parquet).

## Key Variables & Constants
- Grid size: 64x64 (hard-coded in hashing.py)
- States: {-1, 0, +1} (ternary)
- Convergence: ~40-50 ticks typical, max 1000 (SALIENCE_MAX_ITERS_MED)
- Max capacity: 10,000 attractors across all chunks
- Encoders: hash, hippocampus, embedding, blended (default), word, hippo-word

## Data Flow
```
Text -> Encoder -> 64x64 float frame -> quantize to {-1,0,+1} -> CA evolution -> Attractor
                                                                                    |
                                                          Store in chunk/.npz brick
                                                                                    |
Query -> Same pipeline -> Query attractor -> Pearson correlation -> Top-K -> [Reconstruct]
```

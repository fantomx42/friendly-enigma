# wheeler_memory/ — Core Library

This is the main Python package. All modules are imported through `__init__.py`.

## Module Groups

### Encoding Pipeline (Text -> 64x64 Frame)
- **hashing.py** — SHA-256 deterministic encoding. **SACRED — do not modify.**
- **hippocampus.py** — Native n-gram semantic encoder. No pretrained models.
- **embedding.py** — Sentence-transformers + JL random projection. Optional (`.[embed]`).
- **word_encoder.py** — Word-level random indexing with learned co-occurrence vectors (SVD on PMI). Trains from stored corpus. Blended with hippocampus via `WORD_HIPPO_BLEND` in constants.py.

### CA Engine (Frame -> Attractor)
- **dynamics.py** — 3-state CA evolution. Local max -> +1, local min -> -1, slopes flow uphill. Von Neumann (4-neighbor) topology. Dispatches to GPU when available.
- **gpu_dynamics.py** — HIP/CUDA kernel loader. Calls compiled `.so` in `gpu/`.
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
- **consolidation.py** — Sleep consolidation. Prunes redundant keyframes from bricks.
- **eviction.py** — 3-phase graceful degradation when over capacity.
- **attention.py** — Salience-driven variable tick rates (low/med/high CA budget).

### Agents & Rendering
- **agent.py** — LLM chat agent wrapper. Seasons Ollama responses with Wheeler context.
- **decoder.py** — Language Wheeler / Wheeler-primary. Small model as pure decoder for CA state.
- **generation.py** — IT-from-BIT generative engine. Trajectory resonance.
- **language_wheeler.py** — Language-level Wheeler encoding.

### Other
- **constants.py** — ALL tunable parameters. **Only file modified during autoresearch.** See `program.md`.
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

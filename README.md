# Wheeler Memory

A cellular-automaton associative memory engine. Memory is reconstruction under perturbation, not lookup.

> *Darman doesn't retrieve. Darman reconstructs.*

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Architectural source-of-truth lives in [CANON.md](CANON.md).** This README is the public-facing introduction. When canon and this README disagree, canon wins on architecture; this file wins on framing for new readers.

Project also called **Project Darman** and **Project Ralph** — three names for the same repository (canon §14.1). Solo project; pure-Python core; no LLM in the loop.

---

## Core axiom

> Meaning is what survives symbolic pressure.

A stable attractor is what "remembering" means. Unstable patterns collapse and are forgotten. Recall is reconstruction: the system blends a stored attractor with current context and re-evolves under cellular-automaton dynamics. Two queries for the same memory return *similar but not identical* patterns — this is by design.

---

## Status tag legend

Used throughout this README, mirroring canon. A claim with no tag is descriptive prose; a claim with a tag describes a build state.

| Tag | Meaning |
|---|---|
| `[BUILT]` | Exists in code, working |
| `[PARTIAL]` | Implementation started, not complete |
| `[DESIGNED]` | Specified, not yet implemented |
| `[OPEN]` | Known unsolved problem with current best thinking |
| `[SPECULATIVE]` | Direction, not yet specified |
| `[ACTIVE RESEARCH]` | Surface under sweep — numbers and choices change |

---

## Quick start

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e ".[embed]"           # with sentence-transformers (MiniLM baseline)
pip install -e .                    # minimal: numpy, scipy, matplotlib, psutil
```

Python 3.11+. CPU is the target; the recall path does not require a GPU.

```bash
wheeler-store "self-attention computes relationships between all positions"
wheeler-recall "how does attention work in transformers"
```

Recall uses three-grid interference scoring by default. Add `--no-interference` for Pearson-only mode, or `--recognize` for the cheap recognition tier (no CA convergence loop on the query).

---

## Architecture

### State space — balanced ternary `[BUILT]`

Cells live on `{-1, 0, +1}`. `+1` is assertion, `-1` is negation, `0` is the **reconstruction root** — the state from which a pattern can settle into either polarity under interference. Quiescence is potential, not absence. Independently derived; later mapped post-hoc onto Setun (1958) and BitNet b1.58 (2024).

### The three grids `[BUILT]`

The architecture is a tensor product of three same-shaped 64×64 grids with different temporal dynamics. Implementation: `wheeler_memory/scm_grid.py`, `experiential.py`, `interference.py`.

| Grid | Temperature | Update rate | Role |
|---|---|---|---|
| Corpus | Cold | Slow / batch | Stable durable knowledge |
| Experiential | Hot | Fast / per-event | Recent activation, working memory |
| SCM (Map) | Glacial | Hardens with use | Trust topology — *where* interference is permitted |

**SCM Map vs Measure — acronym collision (canon §3.5.1).** The codebase has two unrelated objects sharing the SCM acronym:

- **`scm_grid.py`** — Structural Coherence **Map**. A 64×64 trust topology controlling where interference is permitted. This is the SCM in the answer equation.
- **`cortex_scm.py`** — Structural Coherence **Measure**. A scoring function classifying recall outputs as `SYNTHESIS / NOVEL / HALLUCINATION`. Operates post-recall.

Different objects, same acronym. Canon distinguishes by full name; code does not. When reading "SCM" in commit messages or comments, check which file is in scope.

### Recall — the interference formula `[BUILT]`

```
Answer(i, j) = Corpus(i, j) × Experiential(i, j) × (1 - |SCM(i, j)|)
```

The SCM acts as a **gate**, not a contributor. Hardened cells (`|SCM| → 1`) become opaque; quiescent cells (`|SCM| → 0`) let interference through. This is the waveguide interpretation: the SCM does not generate signal, it routes it (canon §4).

Per-cell state classification (`interference.py`):

| State | Condition |
|---|---|
| `GROUNDED` | Corpus peak + Experiential peak + SCM open |
| `ABSORBED` | Corpus peak + no Experiential + SCM open |
| `UNCONSOLIDATED` | No Corpus + Experiential peak + SCM open |
| `CONTESTED` | Corpus peak + Experiential peak + SCM closed |

### SCM feedback loop `[BUILT]`

Two pathways write into the SCM grid (canon §3.3.1):

1. **Self-consistency erosion** (`scm_grid.py:112` `update()`) — opens or closes cells based on output fidelity through the corpus rules.
2. **Recall-driven feedback** (`scm_grid.py:151` `update_from_recall()`) — adjusts gate magnitude (`κ`) based on recall quality.

The recall-driven `κ` path is the closed loop that earlier framings called the "sleeping giant problem". Canon §3.3.5 records it as resolved; gradient direction is verified by `tests/test_scm_gradient_direction.py`. SCM does not run autonomous CA dynamics — there is no evolution rule on the trust grid; it is feedback-driven only.

> **Known issue.** `interference.py:158` collapses the 64×64 SCM to a global scalar `mean_openness`, making rank ordering identical between frozen and learning arms regardless of spatial SCM state. Tracked under "Open work"; the spatial-product fix is straightforward but changes score semantics from normalized Pearson to weighted mean, so calibration vs. existing recall paths needs care before merging.

### Encoder layer `[ACTIVE RESEARCH]`

The encoder layer is plural and contested. Wheeler runs multiple encoder backends and treats their relative SimLex-999 performance as the live signal of architectural progress. The architectural claim is that meaning can be reconstructed natively if the encoder is good enough; MiniLM is the bar to clear, not the canonical answer (canon §1.2).

**Primary surface:**

| Encoder | Role |
|---|---|
| `hash` | Deterministic SHA-256 seed; default for `wheeler-recall` and reproducible benchmarks |
| `hippocampus` | Wheeler-native character n-gram random indexing; default for `wheeler-simlex`; active production target |
| `embedding` | MiniLM via sentence-transformers; the external baseline to clear (requires `.[embed]`) |
| `blended` | Convex combination; default for user-facing surfaces |

**Research variants** live in `scripts/wheeler_simlex.py:60` `ALL_ENCODERS`: `word`, `hippo-word`, `context`, `context-blended`, `word-blended`, `language`. They compete in sweeps; survivors get promoted.

For live SimLex numbers, run `wheeler-simlex --sweep` rather than trusting a pinned figure in this README.

### Cortex — three-tier semantic scoring `[BUILT]`

A scoring layer over retrieved attractors, structurally separate from the three grids (canon §3.5):

1. **L1 — Correlation graph** (`cortex.py`): Pearson correlation adjacency over the retrieved attractor set, with BFS clustering to identify coherent neighborhoods.
2. **L2 — Settlement CA** (`cortex.py`): Opinion diffusion on the correlation graph until convergence. This is a *second* CA in the system, distinct from the three-grid CA — it runs on graph topology, not the 64×64 grid.
3. **L3 — Native classifier** (`cortex_classifier.py`): ~11K-parameter numpy SGD network. Trained via `train_cortex_classifier.py`.

Output is a `SCMResult` (the *Measure*) classifying recall as `SYNTHESIS / NOVEL / HALLUCINATION` with ten layer scores and a net warrant.

### Two-tier recall `[BUILT]` (v0.3.6)

Default `wheeler-recall` is unchanged (three-grid interference). v0.3.6 added an opt-in API that splits identity from content (`wheeler_memory/recall_api.py`):

- **Recognition tier** — `recognize(query)` does a single-pass Pearson scan against stored attractors using the **raw** query frame. No CA convergence loop on the query. Returns a `BasinSeed` if max similarity ≥ `RECOGNITION_THRESHOLD`, else `None`.
- **Reconstruction tier** — `reconstruct_from_seed(seed, query=str)` warm-starts CA from the stored attractor blended with the raw query frame. The warm start is exactly the savings vs. cold path (~2× ticks reduction across near/mid/far input-distance bands per `scripts/bench/bench_recall_warm_vs_cold.py`).

```python
from wheeler_memory import recognize, recognize_top_k, reconstruct_from_seed

seed = recognize("how does attention work in transformers")
if seed is not None:
    pattern = reconstruct_from_seed(seed, query="how does attention work in transformers")
    print(pattern.text, pattern.convergence_ticks)
```

**Per-basin Temporal Stability (T)** — each basin carries a float in `[0, 1]` in `index.json` (`metadata.t_stability`). With `--learn`, recognition applies an EMA update to `T` and drifts the stored attractor toward the observed pattern at rate `(1 - T) × BASIN_DRIFT_BASE_RATE`. Mature basins (T → 1) are near-rigid; fresh basins absorb input rapidly.

### CA dynamics

3-state rule on a Von Neumann 4-neighborhood with wrapping boundaries. Local maxima push toward `+1` (`MAX_PUSH_STRENGTH`), minima toward `-1`, slopes flow uphill (`SLOPE_FLOW_STRENGTH`). Convergence detected by `percentile(|delta|, 99) < threshold` with an `alive_fraction ≥ 0.05` floor. All tunables — including the autoresearch parameter sweep targets — live in `wheeler_memory/constants.py`; current values move under `wheeler-bench`-driven tuning so are not pinned here. Evolution produces one of four terminal states: `CONVERGED`, `OSCILLATING` (period-2..10 role-space cycle), `DEGENERATE` (<5% alive cells, frame is 0-dominant), or `CHAOTIC` (max iterations exhausted).

---

## Acceleration

**CA semantics are CPU-targeted (canon §1.4).** No CUDA, no ROCm, no Vulkan paths inside the recall engine. The CA is the reasoning engine, not a GPU shader.

HIP kernels in `wheeler_memory/accel/hip/` accelerate **batch operations** — crystallization, SimLex sweeps, large-batch evolution. `evolve_batch()` in `dynamics.py` dispatches to GPU when available and falls back to serial CPU otherwise. Recall itself remains CPU.

Honest number, properly contextualized: ~3 ms/tick CPU; **71× speedup at batch=1000 on RX 9070 XT (RDNA4, gfx1201)** for batched offline operations. See [docs/gpu.md](docs/gpu.md) for setup.

---

## Empirical results

### MMLU `[CHANCE FLOOR]`

Currently sits near 25% (chance for 4-option multiple choice). **Diagnosis: corpus-limited, not architecture-limited (canon §8.2).** Treat MMLU as a *corpus health* metric, not a *recall quality* metric. The Corpus grid has not been populated with sufficient world-knowledge structure; MMLU will move when corpus does.

All 57 subjects, 14,042 questions (test split):

| Run | Mode | Score | Stored memories |
|---|---|---|---|
| Zero-shot Cortex | `--mode cortex` | 24.3% (3,418/14,042) | 0 |
| Cortex + Learned Facts | `--mode cortex` after `--mode learn` | 25.3% (3,557/14,042) | 1,812 science attractors |
| Cortex + L3 Classifier | `--mode cortex --classifier-weights cortex_classifier.npz` | 25.9% (3,643/14,042) | 1,812 + L3 trained |

Encoder: blended (hippocampus 0.7 + language wheeler 0.3). The L3 classifier is trained with numpy SGD (~11K params); loss barely moves from chance — needs more training data or richer features. Full logs in `results/`; recorded baselines in `results/BASELINES.md`. The previous MiniLM semantic baseline (27.5%) used an external pretrained model and is no longer the default encoder.

The right eval for an attractor-reconstruction memory is not a multiple-choice benchmark. Wheeler-native eval design — perturb a known attractor, measure settling time and final-state fidelity — is `[SPECULATIVE]` (canon §8.3).

### SimLex-999 `[ACTIVELY TRACKED]`

Live numbers come from `wheeler-simlex --sweep`. Don't pin them in docs. As of v0.3.3:

- `hippocampus` / `context-RI`: ρ ≈ 0.22–0.26 and climbing
- `MiniLM` (external ceiling): ρ ≈ 0.43

Context-RI is the first native (no pretrained models) encoder to show positive semantic signal. Trained on WikiText-103 + OpenWebText (1.77M docs, 601M words, 384-dim vectors). Decontamination via all-but-the-top singular component removal (K=4) plus Word2Vec-style subsampling. Per-POS: nouns lead, verbs remain the hard case for bag-of-words distributional methods.

### Apple test (semantic holdout)

Exclude a concept from its domain, crystallize neighbours, query for the excluded concept. Hippocampus encoder, no external models.

| Domain | Verdict | Top similarity | Embedding advantage |
|---|---|---|---|
| ML architecture | weak topology | 0.173 | +0.159 over hash control |
| Physics | silent | 0.077 | +0.069 over hash control |
| Biology | silent | 0.090 | +0.083 over hash control |

ML architecture shows the strongest topology: the holdout *"Transformer architecture combines self-attention with feed-forward layers and residual connections"* correctly fires feed-forward networks, layer normalization, and residual connections. The frontier is CA dynamics that preserve more of this structure through evolution.

---

## CLI reference

All 16 commands registered in `pyproject.toml [project.scripts]`. Common flags: `--data-dir`, `--chunk`, `--encoder`, `--salience`, `--verbose`.

### Core operations

| Command | Description |
|---|---|
| `wheeler-store "text"` | Store a memory |
| `wheeler-store "text" --experiential` | Store as episodic memory (loose attractors, 2-day half-life) |
| `wheeler-recall "text"` | Three-grid interference recall (default) |
| `wheeler-recall "text" --no-interference` | Pearson-only recall |
| `wheeler-recall "text" --recognize` | Recognition tier — single-pass match, no convergence loop |
| `wheeler-recall "text" --recognize --learn` | Recognition + per-basin T accumulation + drift |
| `wheeler-forget --text "text"` | Delete a specific memory |
| `wheeler-temps` | List memories with temperature/freshness |
| `wheeler-sleep` | Archive cold memories |

### Pre-training

| Command | Description |
|---|---|
| `wheeler-crystallize corpus.jsonl` | Crystallize a text corpus into the attractor landscape |
| `wheeler-crystallize --no-embed` | Hash encoding instead of embeddings |
| `wheeler-crystallize --max-items N` | Cap processing for validation runs |

### Diagnostics

| Command | Description |
|---|---|
| `wheeler-scrub --text "text"` | Brick inspector — visualise how a memory formed |
| `wheeler-info` | System info (hardware, GPU, paths) |
| `wheeler-bench` | CA quality benchmark (lower is better) |
| `wheeler-bench-gpu` | CPU vs GPU evolution-speed benchmark |
| `wheeler-generate` | Generative engine (IT-from-BIT mode) |
| `wheeler-scm` | Inspect SCM trust topology |
| `wheeler-simlex` | SimLex-999 semantic-similarity benchmark |

### Agents

| Command | Description |
|---|---|
| `wheeler-agent` | LLM chat agent with Wheeler context (requires Ollama) |
| `wheeler-primary` | Wheeler-primary — small LLM as pure decoder (requires Ollama) |

### Benchmark

| Command | Description |
|---|---|
| `wheeler-mmlu --subjects SUBJECT` | Run MMLU on specific subjects |
| `wheeler-mmlu --all` | Run all 57 MMLU subjects |
| `wheeler-mmlu --mode cortex` | Cortex L3 classifier scoring |
| `wheeler-mmlu --mode learn` | Learn dev+val → consolidate → test on test split |
| `wheeler-mmlu --classifier-weights cortex_classifier.npz` | Use trained L3 classifier |

See [docs/cli.md](docs/cli.md) for every flag.

---

## Project structure

```
wheeler_memory/                  Core library
  ENCODING                       hashing.py, hippocampus.py, embedding.py,
                                 word_encoder.py, brick.py
  CA ENGINE                      dynamics.py, oscillation.py, rotation.py
  STORAGE & RECALL               storage.py, reconstruction.py, recall_api.py,
                                 t_metadata.py, cache.py
  THREE-GRID INTERFERENCE        scm_grid.py, experiential.py, interference.py,
                                 similarity.py, trajectory.py, trajectory_cache.py
  CORTEX                         cortex.py, cortex_scm.py, cortex_classifier.py
  AGENTS & RENDERING             decoder.py, language_wheeler.py, agent.py,
                                 generation.py
  LIFECYCLE                      temperature.py, attention.py, warming.py,
                                 consolidation.py, eviction.py
  UTILITIES                      crystallization.py, chunking.py, hardware.py,
                                 polarity.py, constants.py
  theories/                      Production-supporting helpers [BUILT]:
                                 basin.py, metrics.py, synthesis.py
  accel/                         GPU acceleration (batch ops only)
    hip/                         HIP kernel sources + Makefile (RDNA4-aware)
    ca.py                        Python ctypes bindings
  npu/                           NPU/TPU scaffolding (future); OpenVINO + Coral stubs

scripts/                         CLI entry points + benchmarks
  bench/                         apple_test_semantic.py, eval_decoder.py,
                                 bench_associative.py, bench_recall_warm_vs_cold.py,
                                 train_projection.py, measure_separation.py
  tools/                         prepare_corpus.py, topology_map.py,
                                 generate_evolution_gif.py, build_hip.sh,
                                 install_hip_hook.sh, corpus_cleanup.py
  wheeler_simlex.py              ALL_ENCODERS lives here (line 60+)
  train_cortex_classifier.py     L3 cortex classifier training (numpy SGD)

notes/                           Research scratch — not part of pytest, not CLI
  exploration/                   9 research-notebook scripts (ex-scripts/exploration/)
  experiments/                   6 per-theory exercises (ex-scripts/experiments/)
  theories/                      Archived theory modules: lichtenberg.py,
                                 resonance.py, structured.py + their tests

tests/                           pytest suite (run `pytest --collect-only -q` for current count)
results/                         Benchmark logs; BASELINES.md is the textual record
docs/                            Technical documentation; INDEX.md is the entry point
plans/                           Active research & implementation plans
pitch_pack/                      Investor / developer pitch materials
datasets/                        Training corpora (gitignored, ~4 GB)
```

Note the live/archived split for `theories/`:

- `wheeler_memory/theories/` (production-supporting, imported by agent / decoder / wheeler_mmlu / apple-test)
- `notes/theories/` (archived: `lichtenberg.py`, `resonance.py`, `structured.py` — moved in v0.3.6 cleanup)

The `wheeler-ui` CLI was retired in v0.3.6 (orphaned since March 2026); static demo HTML remains under `docs/demos/`.

---

## Open work

In priority order (canon §9):

1. **FCAS address resolution** `[DESIGNED]` — wire `(hash, depth)` tuple keys into the recall path. See FCAS section below.
2. **Wheeler-native eval design** `[SPECULATIVE]` — reconstruction-fidelity benchmark to replace reliance on MMLU as architecture signal (perturb a known attractor, measure settling time and final-state fidelity).
3. **Corpus population strategy** `[OPEN]` — what gets ingested, how it gets ternarized, how to budget across the grid. Affects MMLU directly.
4. **Cross-cube interference semantics** `[SPECULATIVE]` — what does it mean for a nested cube³:0 to interfere with its parent? Speculative until FCAS resolution is done.
5. **`interference_score` spatial-product fix** `[OPEN]` — replace global `mean_openness` scalar with per-cell spatial product so frozen vs. learning SCM arms can differentiate. Requires score-semantics calibration.

---

## FCAS — Fractal Cube Address Space

`[DESIGNED]`. Hash primitives `[BUILT]`; address resolution and fractal nesting not yet wired.

Addresses are tuples `(hash, depth)`. The SHA256 of a terminal attractor serves *simultaneously* as:

1. **Coordinate** — the address at which this attractor lives.
2. **Reconstruction seed** — the initialization for re-instantiating the attractor under perturbation.
3. **Origin of a new cube³:0** — the (0, 0, 0) of a fresh sub-grid nested at this address.

This collapse-of-roles is the load-bearing trick. Every attractor is also a coordinate, which is also a new origin — the address space becomes fractal.

Build status (canon §6.3):

| Component | Status |
|---|---|
| Hash primitives | `[BUILT]` |
| Attractor identification | `[PARTIAL]` |
| Address resolution | `[DESIGNED]` |
| Fractal nesting | `[DESIGNED]` |
| Cross-cube interference | `[SPECULATIVE]` |

---

## Key concepts (short form)

- **Attractor landscape** — every stored memory is a fixed point in CA dynamics. The collection of all stored attractors forms a topology with structure that emerges, rather than is designed.
- **Temperature** — memories carry a temperature in `[0, 1]` based on access frequency and time decay (7-day half-life). Tiers: `hot ≥ 0.6 > warm ≥ 0.3 > cold ≥ 0.05 > fading ≥ 0.01 > dead`.
- **Reconstructive recall** — recall is not retrieval. The stored attractor blends with the query at a configurable α and re-evolves; the same memory comes back differently depending on what you're thinking about.
- **Chunked storage** — memories auto-route to domain chunks (`code`, `science`, `hardware`, `daily_tasks`, `meta`, `general`) by keyword. Capacity: 10,000 attractors.
- **Bridge sentences** — entries that discuss multiple concepts together create measurable attractor overlap. Effective corpus design needs textbook prose, not dictionary entries.

---

## Naming history

Canon §14.1: **Project Ralph** (original; began as a QR-code-to-Coral-TPU hardware idea) → **Project Darman** (middle-period name, used in some commits and the reconstructive-recall philosophy doc) → **Wheeler Memory** (current canonical name; honours Wheeler pregeometry). All three names refer to the same project.

---

## Documentation

[docs/INDEX.md](docs/INDEX.md) is the entry point.

| Guide | Description |
|---|---|
| [Canon](CANON.md) | Architectural source-of-truth — read this first |
| [Installation](docs/install.md) | Python venv setup, GPU notes, Ollama |
| [Architecture](docs/architecture.md) | CA dynamics, encoders, three-grid interference, cortex |
| [Concepts](docs/concepts.md) | Theoretical foundation, reconstructive recall |
| [Design Principles](docs/design.md) | The Darman philosophy |
| [CLI Reference](docs/cli.md) | Every command and flag |
| [API Reference](docs/api.md) | Python library usage |
| [GPU Acceleration](docs/gpu.md) | HIP/ROCm setup, benchmarks |
| [Vision](docs/VISION.md) | Project Ralph — full architecture vision |
| [Future / Roadmap](docs/future.md) | Active research and planned features |
| [Contributing](CONTRIBUTING.md) | Development setup, testing, code style |
| [Changelog](CHANGELOG.md) | Release history |

---

## License

CC BY-NC 4.0. Non-commercial use only. See [LICENSE](LICENSE).

---

*It from bit. The answer emerges from the attractor — not from lookup.*

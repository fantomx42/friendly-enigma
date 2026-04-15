# Wheeler Memory

**A cellular automaton-based associative memory system with real semantic topology** — no LLM, no external models. Pure generative architecture where meaning emerges from attractor dynamics.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU-green.svg)]()

---

## Overview

Wheeler Memory is a **learning architecture**, not a language model. It encodes text through native encoders (Hippocampus n-gram, Context-RI distributional, or blended) into a 64x64 cellular automaton (CA) grid, evolves it through 3-state dynamics until convergence (~5-14 ticks with tuned parameters), and stores the resulting pattern. A Cortex system with three tiers (L1 Graph topology, L2 Settlement CA, L3 Native Classifier) handles semantic scoring and reconstruction.

Similar concepts produce similar attractors. Query evolution followed by three-grid interference scoring (corpus + experiential + SCM trust gating) enables recall. The Language Wheeler component renders CA states as natural language without independent reasoning.

| Component | Role |
|-----------|------|
| **Hippocampus Encoder** | Native semantic embedding via character n-gram random indexing (no pretrained models) |
| **Context-RI Encoder** | Distributional semantics via context-window random indexing (trained on WikiText-103 + OpenWebText, 601M words) |
| **CA Dynamics** | 3-state evolution to stable attractors |
| **Cortex** | L1 semantic topology, L2 settlement stability, L3 classifier scoring |
| **Language Wheeler** | Renders attractor state as text (decoder, not LLM) |

Memories have **temperature** - frequently recalled memories stay warm, stale ones cool. Recall is **reconstructive**: stored attractors blend with query context and re-evolve.

**Benchmark goal:** MMLU against frontier models, measured as **learning gain** (train → consolidate → test), not as language modeling.

---

## Quick Start

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e ".[embed]"
```

**No GPU required.** CPU works fine. Python 3.11+ required.

### Store and recall memories

```bash
wheeler-store "self-attention computes relationships between all positions"
wheeler-recall "how does attention work in transformers"
```

Recall uses three-grid interference scoring by default (corpus + experiential + SCM gating). Add `--no-interference` for Pearson-only mode. Use `--encoder context` for distributional semantics or `--embed` for MiniLM sentence-transformer.

### Train context-RI vectors (distributional semantics)

```bash
# Download corpora (gitignored — ~3.5GB total)
python datasets/download_wikitext.py          # WikiText-103 (101M words)
python datasets/download_openwebtext.py       # OpenWebText subsample (500M words)
cat datasets/wikitext103.jsonl datasets/openwebtext_500m.jsonl > datasets/combined_corpus.jsonl

# Train context-RI vectors on combined corpus
python -m scripts.wheeler_learn_words --method context-ri --corpus datasets/combined_corpus.jsonl
```

### Pre-train from a corpus

```bash
wheeler-crystallize corpus.jsonl --verbose
```

### Run the Wheeler-primary agent (requires Ollama)

```bash
wheeler-primary --interactive --show-state --verbose
```

### Run the MMLU benchmark

```bash
# Zero-shot cortex evaluation (no stored knowledge)
wheeler-mmlu --all --mode cortex

# Full learn → consolidate → test cycle
wheeler-mmlu --subjects high_school_physics conceptual_physics --mode learn

# With trained L3 classifier
wheeler-mmlu --all --mode cortex --classifier-weights cortex_classifier.npz

# All 57 subjects, save results
wheeler-mmlu --all --mode cortex --output results.tsv
```

### Launch the web dashboard

```bash
wheeler-ui  # opens http://localhost:7437
```

---

## Architecture

```
ENCODING PIPELINE
-----------------
Text ---> Encoder (--encoder flag, default: blended)
            ├── hippocampus  character n-gram random indexing (native)
            ├── context      context-window RI (distributional, 601M words, 2M vocab)
            ├── blended      hippocampus(0.7) + language_wheeler(0.3) ← DEFAULT
            ├── embedding    MiniLM sentence-transformer (requires .[embed])
            └── hash/word/word-blended
                    |
                    v
          Random Projection (to 4096 dims) → Reshape to 64x64 grid
                    |
                    v
          CA Evolution (~5-14 ticks to convergence, tuned dynamics)
                    |
                    v
          Attractor (64x64 stable pattern)

THREE-GRID INTERFERENCE ARCHITECTURE (v0.3.1 — default recall path)
----------------------------------------------
Answer(i,j) = Corpus(i,j) * Experiential(i,j) * (1 - |SCM(i,j)|)

Grid 1: CORPUS (Cold)           Grid 2: EXPERIENTIAL (Hot)
  - Crystallized knowledge         - Episodic memory
  - Tight attractors (push=0.57)   - Loose attractors (push=0.35)
  - Barely decays                  - Aggressive decay (2-day half-life)
  - All existing memories          - Temporal context bundled
                    \                 /
                     v               v
            Grid 3: SCM (Structural Coherence Map)
              - 64x64 persistent trust topology
              - WHERE interference is permitted
              - Hardening: early updates have outsized influence
              - Only written by self-consistency feedback loop

Four Interference States:
  GROUNDED       Corpus peak + Experiential peak + SCM open
  ABSORBED       Corpus peak + no Experiential   + SCM open
  UNCONSOLIDATED No Corpus   + Experiential peak + SCM open
  CONTESTED      Corpus peak + Experiential peak + SCM closed

CORTEX SYSTEM (Semantic Scoring & Topology)
-------------------------------------------
Stored Attractors ---> L1 Graph: Semantic topology (Hippocampus native encoder)
                              |
                              v
                    L2 Settlement CA: Stability-driven settlement & consolidation
                              |
                              v
                    SCM: Soft Constraint Satisfaction for coherence
                              |
                              v
                    L3 Classifier: Native semantic scorer (trainable, 11K params)

QUERY & RECALL
--------------
Query ---> Same encoding pipeline ---> Query attractor
                    |
                    v
          Pearson correlation pre-filter (top-2K candidates)
                    |
                    v
          Three-grid interference re-scoring (default since v0.3.1):
            - Corpus Pearson similarity
            - Experiential Pearson similarity
            - SCM openness gating
            - Degrades to pure Pearson when no experiential data exists
                    |
                    v
          Top-K hits ranked by interference score
                    |
                    v
          [Optional] Reconstructive recall: blend stored attractor with query
          context, re-evolve through CA -> reconstructed memory
                    |
                    v
          Language Wheeler renders CA state as text

SELF-CONSISTENCY FEEDBACK LOOP
------------------------------
Decoder output ---> Re-encode ---> Re-evolve under corpus rules
        |                                    |
        v                                    v
  Compare to original               Pearson correlation
  corpus attractor                         |
        |                                    v
        +--- consistent ---> SCM opens gaps (trust increases)
        +--- inconsistent -> SCM closes gaps (trust decreases)
        +--- hardening accumulates: LR / (1 + hardening_count)

MMLU BENCHMARK MODES
--------------------
--mode cortex              : Cortex L3 classifier scoring (default, no LLM)
--mode semantic            : Pure CA attractor Pearson correlation
--mode recall-text         : Reconstruction + text decode
--mode decode              : Small model decoder for rendering
--mode learn               : Full cycle (learn → consolidate → test)
--mode learn-interference  : Learn + experiential storage + SCM sculpting
```

The CA uses a 3-state rule: local peaks push toward +1 (`MAX_PUSH_STRENGTH=0.57`), valleys toward -1, slopes flow uphill (`SLOPE_FLOW_STRENGTH=0.55`). Convergence takes ~3ms on CPU. Batch evolution via `evolve_batch()` dispatches to GPU when available (71x speedup at batch=1000 on RX 9070 XT); all major call sites (SimLex, benchmarks, crystallization) use batch dispatch. Evolution produces one of four terminal states: CONVERGED (stable attractor), OSCILLATING (epistemic uncertainty), DEGENERATE (<5% alive cells — 0-dominant frame rejected), or CHAOTIC (max iterations exhausted). Multiple native encoders are available — the Hippocampus encoder uses character n-grams (lexical similarity), the Context-RI encoder uses distributional co-occurrence vectors trained on WikiText-103 + OpenWebText (601M words, SimLex-999 rho = +0.255). Cortex eliminates all pretrained model dependencies; all semantic understanding is native to the architecture.

The three-grid interference system (default since v0.3.1) transforms Wheeler from a content-addressed store into a system with emergent epistemic states. Existing attractors are corpus by default (ABSORBED state). The SCM starts fully permissive (all zeros) and is sculpted only by the self-consistency feedback loop — no external reward signal. This is "it from bit" applied to epistemology: convergence IS ground truth.

---

## Empirical Results

Validated on a corpus of **2,711 memories** (26.9% grid saturation):

### MMLU Benchmark (learn → consolidate → test)

Wheeler has a full learning loop: store correct Q&A facts from dev+validation splits, run sleep consolidation, then test on the held-out test split. Results use the native Cortex encoder (no external models). Full logs in `results/`.

**All 57 subjects — 14,042 questions (test split):**

| Run | Mode | Score | Notes |
|-----|------|-------|-------|
| Zero-shot Cortex | `--mode cortex` (0 stored memories) | 24.3% (3,418/14,042) | At chance — no knowledge to retrieve |
| Cortex + Learned Facts | `--mode cortex` (1,812 science attractors stored) | 25.3% (3,557/14,042) | +1.0% over zero-shot |
| Cortex + L3 Classifier | `--mode cortex --classifier-weights cortex_classifier.npz` | 25.9% (3,643/14,042) | +1.6% over zero-shot |

Random chance for 4-choice MCQ is **25.0%**. Encoder: blended (hippocampus 0.7 + language wheeler 0.3). The L3 classifier is trained with numpy SGD (11K params) — loss barely moved from chance, needs more training data or richer features.

**Previous MiniLM semantic baseline (removed):** 27.5% — used external pretrained model (`all-MiniLM-L6-v2`), no longer the default encoder.

**Next step:** Reconstruction scoring. Evolve the query → let the CA settle → read back what the attractor is saying → compare that to the choices as text. This is the path where "it from bit" becomes the scoring mechanism, not just the storage philosophy.

### SimLex-999 (Semantic Similarity)

SimLex-999 measures how well an encoder captures genuine semantic similarity (not just relatedness). Evaluated with `wheeler-simlex --encoder <type> --mode pearson`.

| Encoder | Spearman rho | Notes |
|---------|:------------:|-------|
| Context-RI (evolved) | **+0.255** | Distributional RI, trained on WikiText-103 + OpenWebText (601M words, 2M vocab) |
| Context-RI (raw frames) | +0.046 | Before CA evolution — CA dynamics partially erode signal |
| Hippocampus | -0.032 | Character n-grams have no semantic signal (expected) |
| MiniLM (external ceiling) | +0.446 | Pretrained sentence-transformer, reference only |

The context-RI encoder is the first native (no pretrained models) encoder to show positive semantic signal — now at **57% of MiniLM's ceiling**. Trained on WikiText-103 + OpenWebText (1.77M documents, 601M words, 2M vocab, 384-dim vectors). Decontamination via all-but-the-top singular component removal (K=4) and Word2Vec-style subsampling. Per-POS: nouns rho=+0.331, adjectives rho=+0.267, verbs rho=+0.050 (verbs remain the hard case for bag-of-words distributional methods).

### Semantic Apple Test

Exclude a concept from its domain, crystallize all neighbours, then query for the excluded concept. Does the topology predict the missing node? Uses hippocampus encoder on both sides — no external models, pure "it from bit."

| Domain | Verdict | Top Similarity | Embedding Advantage |
|--------|---------|---------------|---------------------|
| ML Architecture | **weak topology** | 0.173 | +0.159 over hash control |
| Physics | silent | 0.077 | +0.069 over hash control |
| Biology | silent | 0.090 | +0.083 over hash control |

Hippocampus n-gram encoding produces real signal above the hash control in all domains. ML architecture shows the strongest topology: the holdout "Transformer architecture combines self-attention with feed-forward layers and residual connections" correctly fires feed-forward networks (0.173), layer normalization (0.135), and residual connections (0.106) — the three component concepts named in the holdout. The frontier is in CA dynamics that preserve more of this structure through evolution.

### Decoder Confidence Gradient

| Attractor Depth | Avg Top Similarity | Uncertain |
|----------------|-------------------|-----------:|
| Deep (crystallized) | 0.409 | 0% |
| Shallow (related) | 0.248 | 0% |
| Missing (out of domain) | 0.193 | 67% |

The system knows what it knows, knows what it doesn't, and knows the difference.

### Co-Activation Topology

Bridge sentences - text discussing multiple concepts together - create measurable attractor overlap:

| Concept Pair | Before Bridge | After Bridge |
|-------------|:------------:|:------------:|
| quantum superposition / entanglement | 1.000 | 1.000 (natural) |
| photosynthesis / mitochondria | 0.000 | 0.250 |
| self-attention / transformer | 0.000 | 0.111 |
| BERT / GPT | 0.000 | 0.111 |
| cross-domain (ML / physics) | 0.000 | 0.000 (correct isolation) |

---

## Corpus Crystallization

Pre-train Wheeler by feeding text corpora through the full pipeline at scale:

```bash
# Prepare a corpus (JSONL, CSV, TXT, or Parquet)
echo '{"text": "concept description here"}' > corpus.jsonl

# Crystallize with semantic embeddings
wheeler-crystallize corpus.jsonl --verbose --batch-size 64

# Resume support - re-running skips already-stored entries
wheeler-crystallize corpus.jsonl --verbose  # only processes new items
```

**Supported input formats:** JSONL (`{"text": "..."}` per line), CSV (column named `text`), TXT (one entry per line), Parquet (auto-detects text column).

The included corpus preparation script extracts from SWE-bench, mbpp, LongBench, and curated domain entries:

```bash
python scripts/tools/prepare_corpus.py
# -> datasets/corpus.jsonl (2711 entries)
```

---

## CLI Reference

### Core Operations

| Command | Description |
|---------|-------------|
| `wheeler-store "text"` | Store a memory (use `--encoder` to select encoding strategy) |
| `wheeler-store "text" --experiential` | Store as episodic memory (loose attractors, temporal context) |
| `wheeler-recall "text"` | Find similar memories (three-grid interference scoring by default) |
| `wheeler-recall "text" --no-interference` | Pearson-only recall (skip experiential + SCM gating) |
| `wheeler-forget --text "text"` | Delete a specific memory |
| `wheeler-temps` | View all memories with temperature/freshness |
| `wheeler-sleep` | Archive cold memories to save space |

### Agents

| Command | Description |
|---------|-------------|
| `wheeler-agent` | LLM chat agent with Wheeler context (requires Ollama) |
| `wheeler-primary` | Wheeler-primary agent - small model as pure decoder (requires Ollama) |
| `wheeler-primary --interactive` | Interactive conversation mode |
| `wheeler-primary --show-state` | Display attractor state alongside responses |

### Pre-Training

| Command | Description |
|---------|-------------|
| `wheeler-crystallize corpus.jsonl` | Crystallize a text corpus into attractor landscape |
| `wheeler-crystallize --no-embed` | Use hash-based encoding instead of embeddings |
| `wheeler-crystallize --max-items 1000` | Cap processing for validation runs |

### Diagnostics

| Command | Description |
|---------|-------------|
| `wheeler-ui` | Web dashboard at http://localhost:7437 |
| `wheeler-scrub --text "text"` | Visualise how a memory formed (brick inspector) |
| `wheeler-info` | System info (hardware, GPU, paths) |
| `wheeler-bench` | Quality score benchmark for CA dynamics tuning |
| `wheeler-bench-gpu` | Benchmark GPU vs CPU evolution speed |
| `wheeler-generate` | Generative text engine (IT from BIT mode) |
| `wheeler-scm` | Inspect SCM trust topology (openness, heatmap, reset) |
| `wheeler-simlex` | SimLex-999 semantic similarity benchmark |

### Benchmark

| Command | Description |
|---------|-------------|
| `wheeler-mmlu --subjects SUBJECT` | Run MMLU on specific subjects |
| `wheeler-mmlu --all` | Run all 57 MMLU subjects |
| `wheeler-mmlu --mode cortex` | Cortex L3 classifier scoring (default) |
| `wheeler-mmlu --mode learn` | Learn dev+val → consolidate → test on test split |
| `wheeler-mmlu --mode learn-interference` | Learn + experiential storage + SCM sculpting |
| `wheeler-mmlu --mode decode --model qwen2.5:1.5b` | Decoder mode (requires Ollama) |
| `wheeler-mmlu --classifier-weights cortex_classifier.npz` | Use trained L3 classifier |
| `wheeler-mmlu --list-subjects` | Print all 57 available subjects |

### Training

| Command | Description |
|---------|-------------|
| `python scripts/train_cortex_classifier.py` | Train the L3 cortex classifier (numpy SGD) |

### Evaluation Scripts

```bash
python scripts/bench/apple_test_semantic.py   # Semantic holdout test
python scripts/bench/eval_decoder.py          # Decoder quality by attractor depth
python scripts/bench/eval_decoder.py --decode # Also run small model (requires Ollama)
python scripts/bench/bench_associative.py     # Associative recall benchmarks
python scripts/tools/topology_map.py          # Co-activation adjacency map
python scripts/tools/generate_evolution_gif.py # Regenerate the CA evolution GIF
```

---

## Key Concepts

### Attractor Landscape

Every stored memory is a fixed-point in CA dynamics. The collection of all stored attractors forms a landscape with topological structure - semantically related concepts occupy nearby basins. This topology is not designed; it **emerges** from the interaction between semantic embeddings, random projection, and CA evolution.

### Temperature System

Memories have temperature based on access frequency and time decay (7-day half-life). Hot memories are prioritised during recall. Cold memories can be archived via `wheeler-sleep`.

Temperature tiers: `hot (≥0.6) > warm (≥0.3) > cold (≥0.05) > fading (≥0.01) > dead (<0.01)`

### Reconstructive Recall (Darman)

Recall is not retrieval. When a stored attractor is recalled, it blends with the query context at a configurable alpha and re-evolves through the CA. The result is a reconstruction shaped by current context - **the same memory comes back differently depending on what you're thinking about.**

### Chunked Storage

Memories are auto-routed to domain chunks (`code`, `science`, `hardware`, `daily_tasks`, `meta`, `general`) via keyword matching. Each chunk maintains its own attractor index, enabling domain-specific recall and capacity management. Maximum capacity: 10,000 attractors.

### Confidence Signal

In Wheeler-primary mode, confidence is derived from the top Pearson similarity of recalled attractors (floor: 0.18). The CA evolution compresses the similarity range relative to raw embedding space - thresholds are calibrated to CA-space values, not embedding-space intuitions.

### Bridge Sentences

Corpus entries that discuss multiple concepts together create measurable attractor overlap (co-activation bridges). Isolated definitions produce strong individual attractors but weak inter-concept connections. Effective corpus design requires connective tissue - textbook prose, not dictionary entries.

---

## Project Structure

```
wheeler_memory/          Core library
  ENCODING
    dynamics.py          CA engine (3-state evolution, GPU dispatch)
    embedding.py         MiniLM sentence embedding + JL random projection (optional)
    hippocampus.py       Native encoder: character n-gram random indexing (default)
    hashing.py           SHA-256 deterministic encoding
    word_encoder.py      Context-window random indexing (distributional semantics)
    brick.py             Memory brick format (.npz archives)
  CORTEX SYSTEM
    cortex.py            Cortex orchestration & L1 graph topology
    cortex_scm.py        L2 Settlement CA + Soft Constraint Satisfaction
    cortex_classifier.py L3 Native semantic classifier (trainable, numpy SGD)
  STORAGE & RECALL
    storage.py           Store/recall with chunked Pearson search
    reconstruction.py    Reconstructive recall (Darman philosophy)
    cache.py             JSON file-based caching layer
  THREE-GRID INTERFERENCE
    scm_grid.py          SCM persistent 64x64 trust topology with hardening
    experiential.py      Episodic memory encoding with temporal context
    interference.py      Three-grid interference engine + self-consistency loop
  AGENTS & RENDERING
    decoder.py           Language Wheeler decoder (text rendering)
    language_wheeler.py  Language Wheeler component (CA state → text)
    agent.py             LLM agent wrapper (Wheeler context seasoning)
    generation.py        Generative engine (IT from BIT)
  UTILITIES
    crystallization.py   Corpus pre-training pipeline (GPU batch-aware)
    temperature.py       Temperature/warmth tracking
    chunking.py          Domain routing (keyword-based)
    gpu_dynamics.py      Backwards-compatible shim → accel.ca
    hardware.py          Hardware detection & optimal device selection
    attention.py         Salience-weighted recall warming
    warming.py           Association tracking
    oscillation.py       Epistemic uncertainty via oscillation detection
    rotation.py          Rotation retry to escape bad attractor basins
    polarity.py          Dual-polarity encoding (antipodal CA states)
    consolidation.py     Sleep consolidation (prune redundant keyframes)
    eviction.py          Three-phase graceful degradation
    constants.py         Tunable system constants
  theories/              Theory experiments (basin, resonance, synthesis)
  accel/                 GPU acceleration (primary)
    __init__.py          gpu_available(), accel_info(), device routing
    _common.py           Shared ctypes helpers, buffer pool
    ca.py                Python bindings for HIP CA evolution kernel
    hip/                 HIP kernel sources + unified Makefile
      ca_evolve.hip      v2 kernel (variable grid, global memory)
      ca_evolve_v1.hip   v1 legacy kernel (64x64, shared memory)
      Makefile           Auto-detects GPU arch, builds all .so targets
  npu/                   NPU/TPU scaffolding (future)
    __init__.py          npu_available(), device_info()
    openvino_bridge.py   Intel NPU stub (OpenVINO INT8 inference)
    coral/               Google Coral Edge TPU (future hardware)
      tpu_bridge.py      PyCoral inference + dual-TPU pipeline stubs
  gpu/                   DEPRECATED — migrated to accel/hip/

scripts/                 CLI entry points
  bench/                 Benchmarks & evaluation
    apple_test_semantic.py   Semantic holdout test
    eval_decoder.py          Decoder quality by attractor depth
    bench_associative.py     Associative recall benchmarks
    train_projection.py      Learn an optimised JL projection matrix
  exploration/           Standalone exploration scripts
  tools/                 Data prep, corpus cleanup, HIP build utilities
    prepare_corpus.py    Corpus preparation (SWE-bench, mbpp, LongBench)
    topology_map.py      Co-activation adjacency map
    generate_evolution_gif.py   Regenerate docs/assets/diagrams/evolution.gif
    build_hip.sh / install_hip_hook.sh   HIP kernel build scripts
  train_cortex_classifier.py   L3 cortex classifier training (numpy SGD)
  wheeler_store.py / wheeler_recall.py / wheeler_forget.py / ...

tests/                   pytest suite (776 tests across 44 modules)
  test_accel_init.py     Accelerator module imports & device detection
  test_accel_ca.py       Batch evolution correctness, GPU vs CPU match
  test_cortex.py         Cortex system unit tests
  test_hallucination.py  Hallucination classification tests
  test_generation.py     Trajectory resonance tests
  ... (dynamics, storage, brick, chunking, consolidation, eviction, etc.)

results/                 Benchmark logs & baselines
  BASELINES.md           Recorded MMLU baseline runs with notes
  mmlu_cortex_*.log      Per-run MMLU logs

docs/                    Technical documentation
  VISION.md              Project Ralph — full architecture vision
  INDEX.md               Guide listing with suggested reading order
  architecture.md        CA dynamics, encoders, interference, cortex, chunked storage
  concepts.md            Theoretical foundation, reconstructive recall
  design.md              The Darman philosophy
  cli.md                 Every flag documented
  api.md                 Python library usage
  gpu.md                 HIP/ROCm and CUDA setup
  install.md             venv setup, platform-specific notes
  future.md              Active research and planned features
  assets/                Diagrams and media
  demos/                 Archived HTML demos (chat, dashboard, CA demo)
  reports/               Generated assessment reports

plans/                   Research & implementation plans
pitch_pack/              Investor/developer pitch materials
datasets/                Training corpora (gitignored, ~4GB: WikiText-103 + OpenWebText)
program.md               Autoresearch tuning program and constants guide
results.tsv              Latest benchmark summary (TSV)
```

---

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for a full guide listing with suggested reading order.

### Getting Started

| Guide | Description |
|-------|-------------|
| [Installation](docs/install.md) | Python venv setup, platform-specific GPU notes, Ollama |
| [Interactive Demo](docs/demos/demo.html) | See the CA engine in your browser (no server needed) |

### Core Guides

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | CA dynamics, encoders, three-grid interference, cortex pipeline, chunked storage |
| [Concepts](docs/concepts.md) | Theoretical foundation, reconstructive recall, semantic vs exact search |
| [Design Principles](docs/design.md) | The Darman philosophy — why recall is reconstruction, not retrieval |

### Reference

| Guide | Description |
|-------|-------------|
| [CLI Reference](docs/cli.md) | Every command and flag documented with examples |
| [API Reference](docs/api.md) | Python library usage — store, recall, reconstruct, crystallize, decode |
| [GPU Acceleration](docs/gpu.md) | HIP/ROCm and CUDA setup, benchmarks |

### Project

| Guide | Description |
|-------|-------------|
| [Vision](docs/VISION.md) | Project Ralph — full architecture vision |
| [Roadmap](docs/future.md) | Active research and planned features |
| [Contributing](CONTRIBUTING.md) | Development setup, testing, code style |
| [Changelog](CHANGELOG.md) | Release history |

### Suggested Reading Order

1. **Installation** — get running
2. **Concepts** — understand the theory
3. **Architecture** — understand the implementation
4. **CLI Reference** — start using it
5. **API Reference** — integrate into your code
6. **Design Principles** — understand the philosophy

---

*It from bit. The answer emerges from the attractor — not from lookup.*

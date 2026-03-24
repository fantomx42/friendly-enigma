# Wheeler Memory

**A cellular automaton-based associative memory system with real semantic topology** — no LLM, no external models. Pure generative architecture where meaning emerges from attractor dynamics.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU-green.svg)]()

---

## Overview

Wheeler Memory is a **learning architecture**, not a language model. It encodes text through a native Hippocampus encoder into a 64x64 cellular automaton (CA) grid, evolves it through 3-state dynamics until convergence (~40-50 ticks), and stores the resulting pattern. A Cortex system with three tiers (L1 Graph topology, L2 Settlement CA, L3 Native Classifier) handles semantic scoring and reconstruction.

Similar concepts produce similar attractors. Query evolution followed by Pearson correlation against stored attractors enables recall. The Language Wheeler component renders CA states as natural language without independent reasoning.

| Component | Role |
|-----------|------|
| **Hippocampus Encoder** | Native semantic embedding via character n-gram random indexing (no pretrained models) |
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

Add `--embed` to use the MiniLM sentence-transformer encoder instead of the native Hippocampus encoder.

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
Text ---> Hippocampus Encoder (native: character n-gram random indexing)
                    |
                    v
          [Optional: MiniLM via --embed flag]
                    |
                    v
          JL Random Projection (to 4096 dims)
                    |
                    v
          Reshape to 64x64 grid, quantize to {-1, 0, +1}
                    |
                    v
          CA Evolution (~40-50 ticks to convergence)
                    |
                    v
          Attractor (64x64 stable pattern)

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
          Cortex L1 scoring via Pearson correlation
                    |
                    v
          Top-K hits ranked by native classifier
                    |
                    v
          [Optional] Reconstructive recall: blend stored attractor with query
          context, re-evolve through CA -> reconstructed memory
                    |
                    v
          Language Wheeler renders CA state as text

MMLU BENCHMARK MODES
--------------------
--mode cortex       : Cortex L3 classifier scoring (default, no LLM)
--mode semantic     : Pure CA attractor Pearson correlation
--mode recall-text  : Reconstruction + text decode
--mode decode       : Small model decoder for rendering
--mode learn        : Full cycle (learn → consolidate → test)
```

The CA uses a 3-state rule: local peaks push toward +1, valleys toward -1, slopes flow uphill. Convergence takes ~3ms on CPU. The Hippocampus encoder uses character 3-grams and 4-grams with random indexing — lexical similarity produces similar frames; true semantic similarity emerges from attractor dynamics and Cortex layers. Cortex eliminates all pretrained model dependencies; all semantic understanding is native to the architecture.

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

### Semantic Apple Test

Exclude a concept from its domain, crystallize all neighbours, then query for the excluded concept. Does the topology predict the missing node?

| Domain | Verdict | Top Similarity | Embedding Advantage |
|--------|---------|---------------|---------------------|
| ML Architecture | **TOPOLOGY** | 0.251 | +0.251 over hash control |
| Physics | **TOPOLOGY** | 0.328 | +0.325 over hash control |
| Biology | weak topology | 0.149 | +0.147 over hash control |

**The system has real geometry, not just lookup.** Quantum entanglement (excluded) fires quantum superposition (top hit, 0.328). Transformer architecture (excluded) fires feed-forward networks and self-attention.

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
| `wheeler-store "text"` | Store a memory (add `--embed` for MiniLM semantic encoder) |
| `wheeler-recall "text"` | Find similar memories (add `--embed` for MiniLM semantic encoder) |
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

### Benchmark

| Command | Description |
|---------|-------------|
| `wheeler-mmlu --subjects SUBJECT` | Run MMLU on specific subjects |
| `wheeler-mmlu --all` | Run all 57 MMLU subjects |
| `wheeler-mmlu --mode cortex` | Cortex L3 classifier scoring (default) |
| `wheeler-mmlu --mode learn` | Learn dev+val → consolidate → test on test split |
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
    brick.py             Memory brick format (.npz archives)
  CORTEX SYSTEM
    cortex.py            Cortex orchestration & L1 graph topology
    cortex_scm.py        L2 Settlement CA + Soft Constraint Satisfaction
    cortex_classifier.py L3 Native semantic classifier (trainable, numpy SGD)
  STORAGE & RECALL
    storage.py           Store/recall with chunked Pearson search
    reconstruction.py    Reconstructive recall (Darman philosophy)
    cache.py             JSON file-based caching layer
  AGENTS & RENDERING
    decoder.py           Language Wheeler decoder (text rendering)
    language_wheeler.py  Language Wheeler component (CA state → text)
    agent.py             LLM agent wrapper (Wheeler context seasoning)
    generation.py        Generative engine (IT from BIT)
  UTILITIES
    crystallization.py   Corpus pre-training pipeline
    temperature.py       Temperature/warmth tracking
    chunking.py          Domain routing (keyword-based)
    gpu_dynamics.py      HIP/CUDA kernel dispatch
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
  gpu/                   HIP/CUDA kernel sources

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

tests/                   pytest suite (21 test modules, 230+ tests)
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
  architecture.md        CA dynamics, temperature system, chunked storage, the math
  concepts.md            Theoretical foundation, reconstructive recall
  design.md              The Darman philosophy
  cli.md                 Every flag documented
  api.md                 Python library usage
  gpu.md                 HIP/ROCm and CUDA setup
  install.md             venv setup, platform-specific notes
  future.md              Planned features and research directions
  assets/                Diagrams and media
  demos/                 Archived HTML demos (chat, dashboard, CA demo)
  reports/               Generated assessment reports

plans/                   Research & implementation plans
pitch_pack/              Investor/developer pitch materials
datasets/                Training corpora (gitignored, ~35GB)
program.md               Autoresearch tuning program and constants guide
results.tsv              Latest benchmark summary (TSV)
```

---

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for a full guide listing with suggested reading order.

| Guide | Description |
|-------|-------------|
| [Installation](docs/install.md) | venv setup, platform-specific notes, GPU acceleration, Ollama |
| [Interactive Demo](docs/demos/demo.html) | See the CA engine in your browser (open as a file, no server) |
| [Architecture](docs/architecture.md) | CA dynamics, temperature system, chunked storage, the math |
| [Concepts](docs/concepts.md) | Theoretical foundation, reconstructive recall, semantic vs exact search |
| [Design Principles](docs/design.md) | The Darman philosophy |
| [CLI Reference](docs/cli.md) | Every flag documented |
| [API Reference](docs/api.md) | Python library usage |
| [GPU Acceleration](docs/gpu.md) | HIP/ROCm and CUDA setup |
| [Vision](docs/VISION.md) | Project Ralph — full architecture vision |
| [Roadmap](docs/future.md) | Planned features and research directions |
| [Contributing](CONTRIBUTING.md) | Development setup, testing, code style |
| [Changelog](CHANGELOG.md) | Release history |

---

*It from bit. The answer emerges from the attractor — not from lookup.*

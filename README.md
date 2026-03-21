# Wheeler Memory

**A cellular automaton memory system with real semantic topology** — not a vector database, not a retrieval engine, but a self-organising attractor landscape that knows what it knows and knows what it doesn't.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU-green.svg)]()

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Looks](#how-it-looks)
- [Architecture](#architecture)
- [Empirical Results](#empirical-results)
- [Key Concepts](#key-concepts)
- [CLI Reference](#cli-reference)
- [Corpus Crystallization](#corpus-crystallization)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

---

## Overview

Wheeler Memory encodes text into a 64×64 cellular automaton (CA) grid, evolves it through 3-state dynamics until it converges to a stable attractor (~40–50 ticks), and stores the resulting pattern. Similar concepts produce similar attractors. When you query, your input is evolved the same way and matched against stored attractors via Pearson correlation.

The system operates in two modes:

| Mode | Description |
|------|-------------|
| **Wheeler-agent** | Wheeler provides context seasoning for a large LLM via Ollama |
| **Wheeler-primary** | Wheeler IS the cognitive system — a small model (Qwen 2.5-1.5B) acts as a pure language renderer for Wheeler's attractor state, with no independent reasoning |

Memories have **temperature** — frequently recalled memories stay warm, stale ones cool and can be archived. Recall is **reconstructive**: stored attractors blend with query context and re-evolve, so the same memory reconstructs differently depending on what you're thinking about.

The long-term goal is to benchmark Wheeler against frontier models on [MMLU](https://huggingface.co/datasets/cais/mmlu), [GPQA](https://arxiv.org/abs/2311.12022), AIME, and SWE-bench — not as a language model, but as a **learning system**: learn on training data, consolidate overnight, test on held-out splits, measure the delta.

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
wheeler-store --embed "self-attention computes relationships between all positions"
wheeler-recall --embed "how does attention work in transformers"
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
# Pure semantic evaluation (no LLM)
wheeler-mmlu --subjects high_school_physics conceptual_physics --mode semantic

# Full learn → consolidate → test cycle
wheeler-mmlu --subjects high_school_physics conceptual_physics --mode learn

# All 57 subjects, save results
wheeler-mmlu --all --mode semantic --output results.tsv
```

### Launch the web dashboard

```bash
wheeler-ui  # opens http://localhost:7437
```

---

## How It Looks

A random 64×64 grid converging to a stable attractor through 3-state CA dynamics.

---

## Architecture

```
ENCODING
--------
Text ---> Sentence Transformer (384-dim)
                    |
                    v
        JL Random Projection (384 -> 4096)
                    |
                    v
  Reshape to 64x64 grid, quantize to {-1, 0, +1}
                    |
                    v
         CA Evolution (~40-50 ticks to convergence)
                    |
                    v
           Attractor (64x64 stable pattern)
                    |
                    v
      Store: attractor + brick + metadata
                    |
                    v
  Chunked storage (code / science / general / ...)

RECALL
------
Query ---> Same encoding pipeline ---> Query attractor
                    |
                    v
  Pearson correlation against all stored attractors
                    |
                    v
         Top-K hits ranked by similarity
                    |
                    v
  [Optional] Reconstructive recall: blend stored attractor
  with query context, re-evolve through CA -> reconstructed memory

WHEELER-PRIMARY MODE
--------------------
Query ---> Recall from attractor landscape
                    |
                    v
    Extract state: confidence, co-activations, depth
                    |
                    v
    Format structured prompt with CA metadata
                    |
                    v
  Small model renders natural language from state
  (no independent reasoning — pure decoder)
```

The CA uses a 3-state rule: local peaks push toward +1, valleys toward -1, slopes flow uphill. Convergence takes ~3ms on CPU. The Johnson-Lindenstrauss random projection preserves semantic neighbourhoods — similar embeddings produce similar attractors.

---

## Empirical Results

Validated on a corpus of 2,711 memories (26.9% grid saturation):

### MMLU Benchmark (learn → consolidate → test)

Wheeler now has a full learning loop: store correct Q&A facts from dev+validation splits, run sleep consolidation, then test on the held-out test split.

**Physics subjects — 488 questions (test split):**

| Subject | Correct / Total | Accuracy |
|---------|-----------------|----------|
| conceptual_physics | 65/235 | 27.7% |
| college_physics | 36/102 | 35.3% |
| high_school_physics | 33/151 | 21.9% |
| **OVERALL** | **134/488** | **27.5%** |

> Random chance for 4-choice MCQ is 25.0%.

Current scoring uses Pearson correlation between CA attractors — it measures attractor shape similarity, not propositional content. The stored Q&A facts are in the index; the scoring mechanism can't yet read them.

**Next step:** Reconstruction scoring. Evolve the query → let the CA settle → read back what the attractor is saying → compare that to the choices as text. This is the path where "it from bit" becomes the scoring mechanism, not just the storage philosophy.

### Semantic Apple Test

Exclude a concept from its domain, crystallize all neighbours, then query for the excluded concept. Does the topology predict the missing node?

| Domain | Verdict | Top Similarity | Embedding Advantage |
|--------|---------|----------------|---------------------|
| ML Architecture | TOPOLOGY | 0.251 | +0.251 over hash control |
| Physics | TOPOLOGY | 0.328 | +0.325 over hash control |
| Biology | weak topology | 0.149 | +0.147 over hash control |

The system has real geometry, not just lookup. Quantum entanglement (excluded) fires quantum superposition (top hit, 0.328). Transformer architecture (excluded) fires feed-forward networks and self-attention.

### Decoder Confidence Gradient

| Attractor Depth | Avg Top Similarity | Uncertain |
|----------------|--------------------|-----------|
| Deep (crystallized) | 0.409 | 0% |
| Shallow (related) | 0.248 | 0% |
| Missing (out of domain) | 0.193 | 67% |

> The system knows what it knows, knows what it doesn't, and knows the difference.

### Co-Activation Topology

Bridge sentences — text discussing multiple concepts together — create measurable attractor overlap:

| Concept Pair | Before Bridge | After Bridge |
|-------------|---------------|--------------|
| quantum superposition / entanglement | 1.000 | 1.000 (natural) |
| photosynthesis / mitochondria | 0.000 | 0.250 |
| self-attention / transformer | 0.000 | 0.111 |
| BERT / GPT | 0.000 | 0.111 |
| cross-domain (ML / physics) | 0.000 | 0.000 (correct isolation) |

---

## Key Concepts

### Attractor Landscape

Every stored memory is a fixed-point in CA dynamics. The collection of all stored attractors forms a landscape with topological structure — semantically related concepts occupy nearby basins. This topology is not designed; it emerges from the interaction between semantic embeddings, random projection, and CA evolution.

### Temperature System

Memories have temperature based on access frequency and time decay (7-day half-life). Hot memories are prioritised during recall. Cold memories can be archived via `wheeler-sleep`.

Temperature tiers: **hot** (≥0.6) > **warm** (≥0.3) > **cold** (≥0.05) > **fading** (≥0.01) > **dead** (<0.01)

### Reconstructive Recall (Darman)

Recall is not retrieval. When a stored attractor is recalled, it blends with the query context at a configurable alpha and re-evolves through the CA. The result is a reconstruction shaped by current context — the same memory comes back differently depending on what you're thinking about.

### Chunked Storage

Memories are auto-routed to domain chunks (`code`, `science`, `hardware`, `daily_tasks`, `meta`, `general`) via keyword matching. Each chunk maintains its own attractor index, enabling domain-specific recall and capacity management. Maximum capacity: 10,000 attractors.

### Confidence Signal

In Wheeler-primary mode, confidence is derived from the top Pearson similarity of recalled attractors (floor: 0.18). The CA evolution compresses the similarity range relative to raw embedding space — thresholds are calibrated to CA-space values, not embedding-space intuitions.

### Bridge Sentences

Corpus entries that discuss multiple concepts together create measurable attractor overlap (co-activation bridges). Isolated definitions produce strong individual attractors but weak inter-concept connections. Effective corpus design requires connective tissue — textbook prose, not dictionary entries.

---

## CLI Reference

### Core Operations

| Command | Description |
|---------|-------------|
| `wheeler-store "text"` | Store a memory (add `--embed` for semantic) |
| `wheeler-recall "text"` | Find similar memories (add `--embed` for semantic) |
| `wheeler-forget --text "text"` | Delete a specific memory |
| `wheeler-temps` | View all memories with temperature/freshness |
| `wheeler-sleep` | Archive cold memories to save space |

### Agents

| Command | Description |
|---------|-------------|
| `wheeler-agent` | LLM chat agent with Wheeler context (requires Ollama) |
| `wheeler-primary` | Wheeler-primary agent — small model as pure decoder (requires Ollama) |
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
| `wheeler-bench` | CA dynamics quality benchmark (logs to results.tsv) |
| `wheeler-bench-gpu` | Benchmark GPU vs CPU evolution speed |
| `wheeler-generate` | Generative text engine (IT from BIT mode) |

### Benchmark

| Command | Description |
|---------|-------------|
| `wheeler-mmlu --subjects SUBJECT` | Run MMLU on specific subjects |
| `wheeler-mmlu --all` | Run all 57 MMLU subjects |
| `wheeler-mmlu --mode learn` | Learn dev+val → consolidate → test on test split |
| `wheeler-mmlu --mode decode --model qwen2.5:1.5b` | Decoder mode (requires Ollama) |
| `wheeler-mmlu --list-subjects` | Print all 57 available subjects |

### Evaluation Scripts

```bash
python scripts/bench/apple_test_semantic.py   # Semantic holdout test
python scripts/bench/eval_decoder.py          # Decoder quality by attractor depth
python scripts/bench/eval_decoder.py --decode # Also run small model (requires Ollama)
python scripts/tools/topology_map.py          # Co-activation adjacency map
```

---

## Corpus Crystallization

Pre-train Wheeler by feeding text corpora through the full pipeline at scale:

```bash
# Prepare a corpus (JSONL, CSV, TXT, or Parquet)
echo '{"text": "concept description here"}' > corpus.jsonl

# Crystallize with semantic embeddings
wheeler-crystallize corpus.jsonl --verbose --batch-size 64

# Resume support — re-running skips already-stored entries
wheeler-crystallize corpus.jsonl --verbose  # only processes new items
```

**Supported input formats:** JSONL (`{"text": "..."}` per line), CSV (column named `text`), TXT (one entry per line), Parquet (auto-detects text column).

The included corpus preparation script extracts from SWE-bench, mbpp, LongBench, and curated domain entries:

```bash
python scripts/tools/prepare_corpus.py  # -> datasets/corpus.jsonl (2711 entries)
```

---

## Project Structure

```
wheeler_memory/          Core library
  dynamics.py            CA engine (3-state evolution, GPU dispatch)
  embedding.py           Sentence transformer + JL random projection
  hashing.py             SHA-256 deterministic encoding
  storage.py             Store/recall with chunked Pearson search
  reconstruction.py      Darman reconstructive recall
  decoder.py             Wheeler-primary agent (small model as decoder)
  crystallization.py     Corpus pre-training pipeline
  temperature.py         Temperature/warmth tracking
  chunking.py            Domain routing (keyword-based)
  brick.py               Memory brick format (.npz archives)
  agent.py               LLM agent with Wheeler context
  gpu_dynamics.py        HIP/CUDA kernel dispatch
  attention.py           Salience-weighted recall
  warming.py             Association tracking
  oscillation.py         Epistemic uncertainty via oscillation detection
  rotation.py            Rotation retry to escape bad attractor basins
  polarity.py            Dual-polarity encoding (antipodal CA states)
  consolidation.py       Sleep consolidation (prune redundant keyframes)
  eviction.py            Three-phase graceful degradation
  generation.py          Generative engine (IT from BIT)
  cache.py               JSON file-based caching layer
  hardware.py            Hardware detection and capability flags
  constants.py           Tunable CA dynamics parameters
  theories/              Theory experiments (basin, resonance, synthesis)
  gpu/                   HIP/CUDA kernel sources

scripts/
  bench/                 Benchmarks & evaluation
    apple_test_semantic.py
    eval_decoder.py
    bench_associative.py
    train_projection.py
  exploration/           Standalone exploration scripts
  experiments/           Theory test harnesses
  tools/                 Data prep, corpus cleanup, GPU build utilities
    prepare_corpus.py
    topology_map.py
    generate_evolution_gif.py
    corpus_cleanup.py
    build_hip.sh / install_hip_hook.sh
  wheeler_store.py       CLI entry points (one per command)
  wheeler_recall.py
  wheeler_crystallize.py
  wheeler_primary.py
  wheeler_agent.py
  wheeler_ui.py
  wheeler_mmlu.py
  wheeler_generate.py
  wheeler_sleep.py
  wheeler_temps.py
  wheeler_forget.py
  bench_quality.py
  bench_gpu.py
  scrub_brick.py
  system_info.py

tests/                   pytest suite (~233 tests)
docs/
  INDEX.md               Guide index with suggested reading order
  VISION.md              Project Ralph architecture vision
  install.md             Installation guide
  architecture.md        CA dynamics, temperature system, the math
  concepts.md            Theoretical foundation, reconstructive recall
  design.md              The Darman philosophy
  cli.md                 Every flag documented
  api.md                 Python library usage
  gpu.md                 HIP/ROCm and CUDA setup
  future.md              Planned features and research directions
  assets/                Images and generated GIFs
  demos/                 Archived HTML demos (chat, dashboard)
  reports/               Generated assessment reports

plans/                   Research & implementation plans
pitch_pack/              Investor/developer pitch materials
datasets/                Training corpora (gitignored, ~35GB)
results.tsv              wheeler-bench CA dynamics tuning log
```

---

## Documentation

See [`docs/INDEX.md`](docs/INDEX.md) for a full guide listing with suggested reading order.

| Guide | Description |
|-------|-------------|
| [install.md](docs/install.md) | venv setup, platform-specific notes, GPU acceleration, Ollama |
| [architecture.md](docs/architecture.md) | CA dynamics, temperature system, chunked storage, the math |
| [concepts.md](docs/concepts.md) | Theoretical foundation, reconstructive recall, semantic vs exact search |
| [design.md](docs/design.md) | The Darman philosophy |
| [cli.md](docs/cli.md) | Every flag documented |
| [api.md](docs/api.md) | Python library usage |
| [gpu.md](docs/gpu.md) | HIP/ROCm and CUDA setup |
| [future.md](docs/future.md) | Planned features and research directions |
| [VISION.md](docs/VISION.md) | Project Ralph architecture vision |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, testing, code style |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

*It from bit. The answer emerges from the attractor — not from lookup.*

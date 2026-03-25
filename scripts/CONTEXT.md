# scripts/ — CLI Entry Points & Tools

All 16 CLI commands are registered in `pyproject.toml [project.scripts]` and point to functions in these files.

## Top-Level Scripts (CLI Entry Points)

### Core Operations
- **wheeler_store.py** — `wheeler-store` CLI. Encodes text and stores attractor.
- **wheeler_recall.py** — `wheeler-recall` CLI. Query -> encode -> Pearson search -> top-K.
- **wheeler_forget.py** — `wheeler-forget` CLI. Delete specific memory by text match.
- **wheeler_temps.py** — `wheeler-temps` CLI. Display all memories with temperature/freshness.
- **wheeler_sleep.py** — `wheeler-sleep` CLI. Archive cold memories, run consolidation.

### Agents
- **wheeler_agent.py** — `wheeler-agent` CLI. LLM chat with Wheeler context (requires Ollama).
- **wheeler_primary.py** — `wheeler-primary` CLI. Small model as pure CA decoder.

### Pre-Training & Generation
- **wheeler_crystallize.py** — `wheeler-crystallize` CLI. Corpus crystallization pipeline.
- **wheeler_generate.py** — `wheeler-generate` CLI. IT-from-BIT generative engine.

### Benchmarks & Diagnostics
- **wheeler_mmlu.py** — `wheeler-mmlu` CLI. MMLU benchmark runner. **Largest file in repo (~60KB).** Supports modes: semantic, cortex, recall-text, decode, learn.
- **bench_quality.py** — `wheeler-bench` CLI. CA quality score benchmark. TEST_INPUTS list is **SACRED**.
- **bench_gpu.py** — `wheeler-bench-gpu` CLI. GPU vs CPU speed comparison.
- **system_info.py** — `wheeler-info` CLI. Hardware and path diagnostics.

### Utilities
- **wheeler_ui.py** — `wheeler-ui` CLI. Web dashboard at localhost:7437.
- **scrub_brick.py** — `wheeler-scrub` CLI. Brick inspector / memory formation visualizer.
- **wheeler_learn_words.py** — Word vector training from stored corpus.
- **train_cortex_classifier.py** — L3 cortex classifier training script.
- **backfill_signatures.py** — Backfill trajectory signatures for existing bricks.

## Key Design Choice
CLI scripts are thin wrappers — they parse args and call into `wheeler_memory/` core library. Business logic lives in the library, not in scripts.

# scripts/ — CLI Entry Points & Tools

All CLI commands are registered in `pyproject.toml [project.scripts]` and bind directly to module paths like `scripts.wheeler_recall:main`. **Top-level wheeler_*.py files cannot be relocated without breaking installed CLI commands** — re-run `pip install -e .` after any entry-point change.

> Research notebooks (`exploration/`) and per-theory exercises (`experiments/`)
> used to live here. They were moved to `notes/` to fight feature creep — see
> `notes/README.md`.

Subdirectories (`bench/`, `tools/`) are NOT Python subpackages — they contain plain script files invoked by path or shell.

## Top-level (entry-point bound)

### Core operations
- **wheeler_store.py** — `wheeler-store`. Encode text, store attractor.
- **wheeler_recall.py** — `wheeler-recall`. Query → encode → Pearson search → top-K.
- **wheeler_forget.py** — `wheeler-forget`. Delete memory by text match.
- **wheeler_temps.py** — `wheeler-temps`. Display memories with temperature/freshness.
- **wheeler_sleep.py** — `wheeler-sleep`. Archive cold memories, run consolidation.

### Agents
- **wheeler_agent.py** — `wheeler-agent`. LLM chat with Wheeler context (requires Ollama).
- **wheeler_primary.py** — `wheeler-primary`. Small model as pure CA decoder.

### Pre-training & generation
- **wheeler_crystallize.py** — `wheeler-crystallize`. Corpus crystallization pipeline.
- **wheeler_generate.py** — `wheeler-generate`. IT-from-BIT generative engine.
- **wheeler_learn_words.py** — Word vector training from stored corpus.

### Benchmarks & diagnostics
- **wheeler_mmlu.py** — `wheeler-mmlu`. MMLU runner (~60 KB, largest file). Modes: semantic, cortex, recall-text, decode, learn, learn-interference.
- **bench_quality.py** — `wheeler-bench`. CA quality score. `TEST_INPUTS` list is **SACRED** (CLAUDE.md).
- **bench_gpu.py** — `wheeler-bench-gpu`. GPU vs CPU speed comparison.
- **system_info.py** — `wheeler-info`. Hardware and path diagnostics.

### Three-grid interference
- **wheeler_scm.py** — `wheeler-scm`. Inspect SCM trust topology.
- **wheeler_simlex.py** — `wheeler-simlex`. SimLex similarity benchmark (`ALL_ENCODERS` defined here).

### Utilities
- **scrub_brick.py** — `wheeler-scrub`. Brick inspector / memory-formation visualizer.

## bench/ — benchmarks & evaluation

Not entry-point bound; invoked as `python scripts/bench/<file>.py`.

- **apple_test_semantic.py** — Semantic holdout test. Excludes a concept, crystallizes neighbors, queries for the excluded concept. Tests emergent topology. Uses `wheeler_memory.theories.synthesis.apple_test`.
- **bench_associative.py** — Associative memory benchmark across domain chunks.
- **bench_recall_warm_vs_cold.py** — Warm/cold recall path comparison.
- **eval_decoder.py** — Decoder confidence gradient at different attractor depths.
- **measure_separation.py** — Inter-cluster separation measurement.
- **scm_ab_eval.py** — Closed-loop A/B over recall arms (Pearson baseline / frozen SCM / learning SCM).
- **train_projection.py** — JL random projection training/validation.

These validate that the CA topology has real geometric structure — the apple test is the primary proof that semantic relationships emerge from attractor dynamics.

## tools/ — data prep, builds, one-shot migrations

Invoked ad-hoc, not on the daily-CLI path.

- **corpus_cleanup.py** — Clean and deduplicate corpus files.
- **prepare_corpus.py** — Extract training data from SWE-bench, mbpp, LongBench, curated entries → `datasets/corpus.jsonl` (2711 entries).
- **generate_evolution_gif.py** — Generate CA evolution animation for `docs/assets/diagrams/evolution.gif`.
- **topology_map.py** — Co-activation adjacency map visualization.
- **train_cortex_classifier.py** — L3 cortex classifier training script.
- **backfill_signatures.py** — Backfill trajectory signatures for existing bricks (one-shot migration).
- **sweep_decontamination.sh** — Singular-component-removal sweep.
- **build_hip.sh** — Build HIP/CUDA kernels.
- **install_hip_hook.sh** — Install GPU kernel build as git hook.

## Design

CLI scripts are thin wrappers — they parse args and call into the `wheeler_memory/` core library. Business logic lives in the library, not in scripts.

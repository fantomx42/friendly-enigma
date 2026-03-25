# scripts/bench/ — Benchmarks & Evaluation

## Scripts

- **apple_test_semantic.py** — Semantic holdout test. Excludes a concept from its domain, crystallizes neighbors, queries for the excluded concept. Tests emergent topology. Uses `wheeler_memory.theories.synthesis.apple_test`.
- **bench_associative.py** — Associative memory benchmark. Tests recall quality across domain chunks.
- **eval_decoder.py** — Decoder confidence gradient analysis. Measures similarity at different attractor depths (deep/crystallized, shallow/related, missing/OOD). Add `--decode` flag to also run small model rendering via Ollama.
- **train_projection.py** — JL random projection training / validation.

## Why These Matter
These benchmarks validate that the CA topology has real geometric structure — not just lookup. The apple test is the primary proof that semantic relationships emerge from attractor dynamics.

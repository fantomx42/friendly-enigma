# tests/ — pytest Suite

~28 test files covering all core modules. Run with `pytest` from repo root.

## Markers
- `slow` — long-running tests (skip with `pytest -m "not slow"`)
- `embed` — requires sentence-transformers (skip with `pytest -m "not embed"`)

## Key Test Files
- **test_dynamics.py** — CA convergence, 3-state rule correctness
- **test_storage.py** — Store/recall round-trip, Pearson search accuracy
- **test_hashing.py** — Deterministic encoding invariants
- **test_cortex.py** — Cortex L1/L2/L3 pipeline
- **test_hallucination.py** — Hallucination classification (synthesis vs novel)
- **test_word_encoder.py** / **test_word_encoder_l2.py** — Word-level encoding
- **test_reconstruction.py** — Reconstructive recall (Darman blend + re-evolve)

## Fixtures
`conftest.py` provides shared fixtures for temporary data dirs and pre-stored memories.

## Sub-packages
- **theories/** — `test_hallucination_discrimination.py` (hallucination scoring edge cases)

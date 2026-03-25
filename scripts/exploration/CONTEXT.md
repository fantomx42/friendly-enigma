# scripts/exploration/ — Standalone Exploration Scripts

One-off investigation scripts for probing specific subsystems. Not part of the test suite.

- **test_attention.py** — Attention model behavior (variable tick budgets)
- **test_consolidation.py** — Sleep consolidation mechanics
- **test_diversity.py** / **test_diversity_math.py** — Attractor diversity measurement
- **test_eviction.py** — Eviction behavior under capacity pressure
- **test_paraphrase.py** / **test_paraphrase_embed.py** — Paraphrase similarity in CA space vs embedding space
- **test_reconstruction.py** — Reconstructive recall behavior (Darman blending)
- **test_warming.py** — Spreading activation / warmth propagation

These scripts are for generating data and insights, not for regression testing.

# wheeler_memory/theories/ — Compartmentalized Experiments

Isolated experiments on CA attractor topology. Each module wraps around core Wheeler Memory without modifying it. All imports are non-fatal (try/except in `__init__.py`).

## Modules

- **basin.py** — Basin width analysis. Measures how far you can perturb an attractor before it falls into a different basin. Functions: `measure_basin_width`, `find_basin_gaps`, `map_all_basins`.
- **synthesis.py** — Apple test (semantic holdout). Exclude a concept, crystallize neighbors, query for excluded concept. Tests whether topology predicts missing nodes. Functions: `apple_test`, `run_apple_battery`, `synthesize_from_gap`.
- **structured.py** — Theory building from attractor state. Constructs structured `Theory` objects from recalled context. Functions: `build_theory`, `theory_to_prompt`.
- **resonance.py** — Resonance querying across stored corpus. Functions: `query_corpus`, `ResonanceResult`.
- **metrics.py** — Topology quality metrics: `energy`, `basin_width`, `context_weight`, `hallucination_score`, `topology_consistency`.
- **lichtenberg.py** — Visualization. Plots topology maps and animates apple test results. Functions: `plot_topology`, `animate_apple_test`.

## Why This Exists
These are research experiments — not production code. They test hypotheses about the emergent geometry of the attractor landscape. Results feed back into parameter tuning and architecture decisions.

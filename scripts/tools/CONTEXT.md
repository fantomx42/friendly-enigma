# scripts/tools/ — Data Prep & Build Utilities

- **corpus_cleanup.py** — Clean and deduplicate corpus files
- **prepare_corpus.py** — Extract training data from SWE-bench, mbpp, LongBench, and curated entries -> `datasets/corpus.jsonl` (2711 entries)
- **generate_evolution_gif.py** — Generate the CA evolution animation (docs/assets/diagrams/evolution.gif)
- **topology_map.py** — Co-activation adjacency map visualization
- **build_hip.sh** — Build HIP/CUDA kernels
- **install_hip_hook.sh** — Install GPU kernel build as git hook

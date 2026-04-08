# Future Work

## Completed

- ~~**GPU acceleration** — HIP kernels for parallel chunk evolution on AMD GPUs (ROCm)~~ [Done](gpu.md)
- ~~**Embedding-based routing** — semantic similarity for chunk selection instead of keywords~~ Done
- ~~**Reconstructive recall** — memories influenced by current context (Darman architecture)~~ Done
- ~~**Associative warming** — spreading activation between related memories on recall~~ Done
- ~~**Eviction / forgetting** — graceful degradation of cold memories (fade bricks, evict dead, capacity limits)~~ Done
- ~~**Sleep consolidation** — prune redundant intermediate frames within bricks, keeping only salient keyframes~~ Done
- ~~**Variable tick rates (attention model)** — salience-driven CA budgets~~ Done
- ~~**Three-grid interference** — corpus/experiential/SCM trust gating with epistemic states~~ Done (v0.3.0)
- ~~**Cortex L1/L2/L3** — graph reasoning, settlement CA, native classifier~~ Done (v0.3.0)
- ~~**Context-RI encoder** — distributional semantics via context-window random indexing~~ Done
- ~~**SimLex-999 benchmark** — `wheeler-simlex` for semantic similarity evaluation~~ Done
- ~~**Autoresearch infrastructure** — overnight parameter sweeps, `program.md` protocol~~ Done

## Active Research

- **CA dynamics that preserve semantic signal** — current tuned dynamics (push=0.57, slope=0.55) erode distributional signal. SimLex-999 rho drops from +0.046 (raw) to +0.034 (evolved). Goal: dynamics that amplify rather than degrade semantic structure.
- **Per-encoder CA parameters** — different encoders may benefit from different CA regimes (e.g., softer dynamics for context-RI, tighter for corpus hippocampus).
- **Reconstruction scoring for MMLU** — evolve query → CA settle → read back attractor → compare to choices as text. "It from bit" as the scoring mechanism.

## Planned

- **Richer training corpora** — expand context-RI beyond WikiText-103 (academic papers, code documentation, domain-specific text)
- **Larger context windows** — `CONTEXT_RI_WINDOW` > 5 may capture longer-range dependencies
- **Fractal cube address space** — hierarchical 64×64 grids for multi-resolution memory (see `plans/fractal_cube_address_space.md`)
- **Parallel decoder** — K×4096 → tokens in parallel (no autoregressive generation)
- **Active learning** — system notices low-SCM gaps and seeks information to fill them
- **Cross-chunk interference** — three-grid scoring across domain boundaries
- **L3 classifier improvements** — batch training, Adam optimizer, regularization, richer features (see `docs/reports/CORTEX_CLASSIFIER_SUMMARY.md`)

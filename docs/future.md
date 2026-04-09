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

## GPU Acceleration Roadmap

Four-stage plan to move compute off the CPU onto the RX 9070 XT (RDNA4, 64 CUs, gfx1201). See `accel/hip/CONTEXT.md` for RDNA4-specific tuning notes.

| Stage | What | Status |
|-------|------|--------|
| 0 | Directory structure: `accel/` + `npu/` scaffolding, migrate `gpu/` → `accel/hip/` | **Done** |
| 1 | Batch GPU evolution: wire `evolve_batch()` into SimLex, bench, crystallization | **Done** |
| 2 | GPU-accelerated encoding: port hippocampus n-gram + context-RI matrix ops to HIP | Planned |
| 3 | GPU-accelerated similarity search: batch Pearson correlation on stored attractors | Planned |
| 4 | GPU-accelerated cortex: settlement CA + graph scoring on GPU (only if profiling shows need) | Planned |

Stages 2 and 3 are independent (can be done in parallel). Stage 4 depends on Stage 3. NPU scaffolding is independent.

## Planned

- **GPU encoding kernel (`hip_encode.hip`)** — sparse n-gram accumulation + (384→4096) projection on GPU. Batch B texts → B frames in one dispatch. RDNA4 WMMA for the projection GEMM.
- **GPU similarity kernel (`hip_similarity.hip`)** — persistent GPU buffer of stored attractors. On recall, dispatch batch Pearson correlation against all N stored frames. Wave32 = 32 attractors per wave.
- **Richer training corpora** — expand context-RI beyond WikiText-103 (academic papers, code documentation, domain-specific text)
- **Larger context windows** — `CONTEXT_RI_WINDOW` > 5 may capture longer-range dependencies
- **Fractal cube address space** — hierarchical 64×64 grids for multi-resolution memory (see `plans/fractal_cube_address_space.md`)
- **Parallel decoder** — K×4096 → tokens in parallel (no autoregressive generation)
- **Active learning** — system notices low-SCM gaps and seeks information to fill them
- **Cross-chunk interference** — three-grid scoring across domain boundaries
- **L3 classifier improvements** — batch training, Adam optimizer, regularization, richer features (see `docs/reports/CORTEX_CLASSIFIER_SUMMARY.md`)
- **Intel NPU integration** — offload cortex L3 classifier to NPU via OpenVINO (stubs in `npu/openvino_bridge.py`)
- **Google Coral Edge TPU** — dual M.2 TPU inference for INT8 classifier (stubs in `npu/coral/tpu_bridge.py`)

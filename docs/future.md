# Future Work

## Top priorities (canon §9)

In priority order. Source-of-truth: `CANON.md` §9.

1. **FCAS address resolution** `[DESIGNED]` — wire `(hash, depth)` tuple keys
   into the recall path. Hash primitives are `[BUILT]`; address resolution and
   fractal nesting are designed but not implemented. See canon §6 and
   `plans/fractal_cube_address_space.md`.
2. **Wheeler-native eval design** `[SPECULATIVE]` — reconstruction-fidelity
   benchmark to replace reliance on MMLU as architecture signal. The right eval
   for an attractor-reconstruction memory is not multiple choice; it is
   something like *perturb a known attractor, measure settling time and
   final-state fidelity*. Not yet specified.
3. **Corpus population strategy** `[OPEN]` — what gets ingested, how it gets
   ternarized, how to budget across the grid. MMLU is corpus-limited (canon
   §8.2); this is the lever that moves it.
4. **Cross-cube interference semantics** `[SPECULATIVE]` — what does it mean
   for a nested cube³:0 to interfere with its parent? Speculative until FCAS
   resolution is done.

## Known issue

- **`interference_score` spatial-product fix** `[OPEN]` —
  `interference.py:158` collapses the 64×64 SCM to a global scalar
  `mean_openness`, making rank ordering identical between frozen and learning
  arms regardless of spatial SCM state. The fix is straightforward
  (`score = mean((q_corpus * s_corpus) * (1-|SCM|)) + mean((q_exp * s_exp) * (1-|SCM|))`)
  but changes score semantics from normalized Pearson to weighted mean —
  calibration vs. existing recall paths needs care before merging.

## Active research

- **CA dynamics that preserve semantic signal** — current tuned dynamics
  (push=0.57, slope=0.55) erode distributional signal. Goal: dynamics that
  amplify rather than degrade semantic structure under evolution.
- **Per-encoder CA parameters** — different encoders may benefit from different
  CA regimes (e.g. softer dynamics for context-RI, tighter for corpus
  hippocampus).
- **Reconstruction scoring for MMLU** — evolve query → settle → read back
  attractor → compare to choices as text. "It from bit" as the scoring
  mechanism rather than just the storage philosophy.

## Acceleration roadmap

CA semantics are CPU-targeted (canon §1.4). The acceleration work below targets
**batch operations** — crystallization, SimLex sweeps, similarity scans — not
the per-query recall path.

| Stage | What | Status |
|---|---|---|
| 0 | `accel/` scaffolding; migrate `gpu/` → `accel/hip/` | Done |
| 1 | Batch GPU evolution: wire `evolve_batch()` into SimLex, bench, crystallization | Done |
| 2 | GPU-accelerated encoding: port hippocampus n-gram + context-RI matrix ops to HIP | Planned |
| 3 | GPU-accelerated similarity search: batch Pearson correlation on stored attractors | Planned |
| 4 | GPU-accelerated cortex: settlement CA + graph scoring on GPU (only if profiling shows need) | Planned |

Stages 2 and 3 are independent (parallelizable). Stage 4 depends on Stage 3.
See `accel/hip/CONTEXT.md` for RDNA4-specific tuning notes.

## Other planned items

- **GPU encoding kernel (`hip_encode.hip`)** — sparse n-gram accumulation +
  (384→4096) projection on GPU. Batch B texts → B frames in one dispatch.
  RDNA4 WMMA for the projection GEMM.
- **GPU similarity kernel (`hip_similarity.hip`)** — persistent GPU buffer of
  stored attractors. On batch recall, dispatch Pearson correlation against all
  N stored frames. Wave32 = 32 attractors per wave.
- **Richer training corpora** — expand context-RI beyond WikiText-103 +
  OpenWebText (academic papers, code documentation, domain-specific text).
- **Larger context windows** — `CONTEXT_RI_WINDOW > 5` may capture longer-range
  dependencies.
- **Parallel decoder** — K×4096 → tokens in parallel (no autoregressive
  generation).
- **Active learning** — system notices low-SCM gaps and seeks information to
  fill them.
- **L3 classifier improvements** — batch training, Adam optimizer,
  regularization, richer features.

## Completed

- ~~**GPU acceleration scaffolding** — HIP kernels for batch CA evolution~~ Done (`docs/gpu.md`)
- ~~**Embedding-based routing** — semantic similarity for chunk selection~~ Done
- ~~**Reconstructive recall** — memories influenced by current context (Darman)~~ Done
- ~~**Associative warming** — spreading activation between related memories~~ Done
- ~~**Eviction / forgetting** — graceful degradation of cold memories~~ Done
- ~~**Sleep consolidation** — prune redundant intermediate frames within bricks~~ Done
- ~~**Variable tick rates (attention model)** — salience-driven CA budgets~~ Done
- ~~**Three-grid interference** — corpus / experiential / SCM trust gating~~ Done (v0.3.0)
- ~~**Cortex L1/L2/L3** — graph reasoning, settlement CA, native classifier~~ Done (v0.3.0)
- ~~**Context-RI encoder** — distributional semantics via context-window RI~~ Done (v0.3.2)
- ~~**SimLex-999 benchmark** — `wheeler-simlex` for semantic similarity eval~~ Done
- ~~**Autoresearch infrastructure** — overnight parameter sweeps~~ Done
- ~~**SCM telemetry + closed-loop A/B**~~ Done (v0.3.4)
- ~~**SCM feedback loop closure ("sleeping giant" resolved)**~~ Done (v0.3.4–§3.3.5)
- ~~**Two-tier recall API** — recognize / reconstruct + per-basin Temporal Stability~~ Done (v0.3.6)

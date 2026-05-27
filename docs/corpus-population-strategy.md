# Corpus Population Strategy

> Resolves CANON open work item #3 (§9): *"what gets ingested, how it gets
> ternarized, how to budget across the grid."* Validated 2026-05-26.

## Context — the problem this addresses

The Corpus grid mechanism is `[BUILT]` (CANON §3.1) but **empty of
world-knowledge structure**. CANON §8.2 is explicit that MMLU sits at the 25%
chance floor for this reason — *"corpus-limited, not architecture-limited …
MMLU will move when corpus does."* Item #3 is therefore not a missing feature;
it is a decision about what to put into the cold grid and how. This document
makes those decisions and backs each with a measured proof run
(`scripts/bench/corpus_population_proof.py`).

## The strategy

### 1. What gets ingested
- **Unit:** one stored memory per fact, the full `Q: <question> A: <answer>`
  string. The question is the retrieval handle; the answer carries the payload.
- **First domain:** structured **science Q&A** from the local corpora
  (`datasets/arc.jsonl`, `datasets/sciq.jsonl` — ~21k unique facts), routed to
  the `science` chunk. Science is chosen because the raw material is already on
  disk and offline, and it maps onto MMLU science subjects for cross-checking.
- **Routing:** rely on the existing domain chunker (`chunking.py`), one chunk
  per domain, so each chunk keeps an independent index/cache. Other domains
  (history, law, …) follow the same recipe once their corpora are sourced.

### 2. How it gets ternarized
**Store the attractor snapped to {-1, 0, +1}** via
`dynamics.snap_to_ternary` (default `topological` mode), not the continuous
float32 attractor.

Justification is empirical, not aesthetic — the proof run measured both:

| storage mode | recall@1 | recall@5 | bytes/cell |
|--------------|----------|----------|------------|
| **ternary-snap** | 0% → **97%** | 0% → **100%** | 1 (int8) |
| float32      | 0% → 97% | 0% → 100% | 4 |

Ternary recall is **identical** to float at every cutoff, so the snap costs
nothing in retrieval quality while quartering attractor storage and aligning
the stored form with the CA's own 3-state output and the MMLU `ternary` scoring
path. The snap is free; take it.

### 3. How to budget across the grid
- **Batch population, eviction off.** Populate with `auto_evict=False`; the grid
  is a cold layer updated in infrequent batches (CANON §3.1), not under live
  capacity pressure. Run eviction (`eviction.py`) as a separate pass if a chunk
  exceeds its budget, *after* population.
- **Rebuild the cache once, at the end.** Recall uses the vectorized
  `AttractorCache`; build it once after a population batch rather than per-write.
- **Validated scale:** 1500 facts in a single `science` chunk retrieve cleanly
  (recall@1 97%). The 64×64 grid is the per-attractor substrate, not a global
  capacity ceiling — capacity is bounded by index/disk, managed per chunk.

### 4. Encoder consistency — the load-bearing prerequisite
Stored attractors and query attractors **must come from the same encoder**, or
Pearson similarity compares unrelated spaces and population yields nothing.

- Population uses the **embedding encoder** (`embedding.embed_to_frame`,
  MiniLM) for semantic generalization across paraphrases.
- **Known gap:** the stock `wheeler-mmlu` semantic path hash-encodes queries
  (`score_semantic` → `hash_to_frame`) and `_get_encoder_fns` has no
  `embedding` option, so it *cannot* consume an embedding-populated corpus. The
  proof run works around this by injecting embedding frames via
  `score_semantic(..., precomputed_frames=...)`. **Recommended follow-up:** add
  an `embedding` encoder to `_get_encoder_fns` so `wheeler-mmlu` can score
  against the populated corpus directly.

## Proof of effect

Run (offline; embedding encoder; 1500 science facts; seed 42):

```bash
python scripts/bench/corpus_population_proof.py            # recall, both snap modes
python scripts/bench/corpus_population_proof.py --skip-recall \
    --mmlu-subjects high_school_biology,conceptual_physics --mmlu-samples 60
```

**Primary — Wheeler-native recall (corpus health, measured directly):**
populating the corpus takes closed-set recall@1 from **0% → 97%** and recall@5
from **0% → 100%** over 300 held questions. Population creates a functioning,
retrievable knowledge structure.

**Secondary — MMLU science cross-check (embedding-consistent):** on 120 items
across `high_school_biology` + `conceptual_physics`, accuracy moves from
**24% → 32%** (chance 25%) — from the floor to +7pp above it, purely from
populating the corpus, no architecture change. This is the §8.2 prediction
realised: MMLU moves when corpus does.

## Recommended rollout
1. Land an `embedding` encoder in `wheeler-mmlu`'s `_get_encoder_fns` (closes
   the consistency gap above).
2. Promote the proof harness to a repeatable population pass over the full
   science corpus, with a post-population eviction + cache-rebuild step.
3. Source non-science corpora to extend coverage to other MMLU domains; the
   recipe (full-fact unit → embedding encode → ternary-snap → domain chunk) is
   domain-agnostic.
4. Track corpus health with the recall proof; track architecture health
   separately with `wheeler-recon-bench` (§8.3). Keep the two signals distinct.

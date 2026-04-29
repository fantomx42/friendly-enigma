# Wheeler Memory Concepts

## How It Works

```
Input Text
    ↓
Encoder (hash | hippocampus | blended | context | embedding | word)
    ↓
64×64 seed frame (values in [-1, +1])
    ↓
3-State CA Evolution (iterate until convergence)
    ├→ CONVERGED   → Store attractor + brick
    ├→ OSCILLATING → Epistemic uncertainty detected
    ├→ DEGENERATE  → 0-dominant frame (<5% alive cells), rejected
    └→ CHAOTIC     → Input needs rephrasing
```

## Encoder Taxonomy

Wheeler Memory supports multiple encoding strategies via `--encoder`:

1. **Hash (SHA-256)** — deterministic, exact match. Same text always produces the same frame. Changing one character completely changes the seed (avalanche effect).

2. **Hippocampus** — native character n-gram random indexing. Lexically similar text produces similar frames. No pretrained models required.

3. **Blended** (default) — hippocampus (0.7) + language wheeler (0.3). Combines lexical and structural signals.

4. **Context-RI** — distributional semantics via context-window random indexing. Trained on WikiText-103 (500K vocab, 384-dim). First native encoder with positive semantic signal (SimLex-999 rho = +0.101).

5. **Embedding** — sentence-transformer (`all-MiniLM-L6-v2`). Requires `pip install -e ".[embed]"`. Strongest semantic signal (SimLex rho = +0.446) but uses an external pretrained model.

6. **Word / Word-blended** — word-level random indexing with SVD on PMI matrix.

```

**The three cell roles** (von Neumann neighborhood):

| Role | Condition | Update | Meaning |
|------|-----------|--------|---------|
| Local Maximum | `cell >= all 4 neighbors` | Push toward +1 (0.35) | Attractor basin center |
| Slope | Neither max nor min | Flow toward max neighbor (0.20) | Transitional |
| Local Minimum | `cell <= all 4 neighbors` | Push toward -1 (0.35) | Repellor / valley |

Convergence typically happens in 39-49 ticks (~3ms on CPU). The result is a unique QR-code-like binary pattern per input.

## Theoretical Foundation

Wheeler Memory implements the **Symbolic Collapse Model (SCM)**:

1. **Meaning is what survives symbolic pressure** — stable attractors represent survived concepts
2. **Memory and learning are the same process** — each interaction reshapes the landscape
3. **Uncertainty is observable in dynamics** — convergence = clarity, oscillation = ambiguity, chaos = contradiction
4. **Time is intrinsic to memory** — convergence speed reflects concept complexity; full history is preserved

Named after John Archibald Wheeler's "It from Bit" — information emerges from physical-like dynamics.

---

## Reconstructive Recall

When `--reconstruct` is passed, recalled memories don't come back unchanged. Each stored attractor is **blended with the query attractor** and **re-evolved through the CA**:

```
blend = (1 - α) × stored + α × query   (default α = 0.3, memory-dominant)
reconstructed = evolve_and_interpret(blend)
```

The same stored memory reconstructs differently depending on the current query context. Querying "machine learning" vs. "web development" against the same stored memory yields different reconstructions — exactly like human episodic memory (see: Elizabeth Loftus research on reconstructive recall).

The result includes `correlation_with_stored` and `correlation_with_query`, showing how far the reconstruction drifted toward the query context.

---

## Philosophical Background

### Symbolic Compression and Meaning

**Axiom**: "Meaning is what survives symbolic pressure."

```
Input data → Symbolic pressure (compression, decay, competition)
                        ↓
            What survives = Meaning
            What evicts   = Noise
```

Wheeler Memory implements this directly:
- **Symbolic pressure** = temperature decay over time
- **Survival** = attractors that maintain high temperature through repeated access
- **Meaning** = stable attractors that resist decay

### The Irreversibility Requirement

The CA evolution is **not reversible**. Many initial seeds converge to the same attractor basin — information is lost. This is a feature:

1. **IIT (Integrated Information Theory)**: Consciousness requires time-irreversibility + information integration
2. **Wheeler Memory has both**: Attractor collapse is irreversible; CA neighbors integrate information from up to 4 directions per tick

The system makes no claims about Φ for the actual 64×64 working grid.

### Reconstructive Memory vs. Retrieval

```
Traditional (exact retrieval):
  Store:    Data → Address 0x1A4F
  Retrieve: Address 0x1A4F → Exact same data  (deterministic, lossless)

Wheeler Memory (reconstructive):
  Store:    Input → hash → CA evolution → Attractor A
  Recall:   Query → hash → CA evolution → Attractor Q
            Reconstruct: blend(A, Q, α=0.3) → re-evolve → Attractor A'
            A' ≠ A  (lossy, context-influenced)
```

### Temperature as Epistemic Humility

Current LLMs confabulate confidently because they have no mechanism to express uncertainty about memory. Wheeler Memory's temperature gives Darman the ability to calibrate confidence language based on how recently and often a memory has been accessed:

```
hot  (≥0.6):  "I remember discussing X..."
warm (≥0.3):  "I think we touched on X..."
cold (<0.3):  "I vaguely recall X, but I'm uncertain..."
```

This prevents the **recovered memory therapy** failure mode: filling reconstruction gaps with confident fabrication.

---

## Two-Tier Recall: Recognition vs. Reconstruction

The legacy single-call `recall_memory()` collapsed identity and content. The
two-tier API (v0.3.6+) separates them — the same separation human memory
makes between *I know that name* (recognition) and *let me tell you about
them* (reconstruction).

### Basins are identities

A stored attractor is a basin of attraction in CA state space. A basin's
identity is the set of seeds that fall into it under CA evolution.
Recognition asks a coordinate question: **does this query land in a known
basin?** It does not need to retrieve, blend, or re-evolve content; it only
needs to confirm that the query frame lies close enough to a stored
attractor (Pearson ≥ `RECOGNITION_THRESHOLD = 0.45`) to count as the same
basin.

Because the question is purely about location, `recognize()` evaluates it on
the **raw** query frame — no `evolve_and_interpret` on the query. A single
`apply_ca_dynamics` probe runs on the winning basin to read its current
stability, but that is constant 1-tick work, not a convergence loop.

### Seeds as cheap handles

`BasinSeed` is identity-only: `hex_key`, `chunk`, `similarity`,
`basin_depth`, `temperature`, `t_stability`, `text`, `data_dir`,
`basin_stability`. It is sufficient to look up the stored attractor and
warm-start CA settling later, without paying for content retrieval up front.
A seed answers *which basin* without committing to *what's inside*.

Reconstruction takes a seed and produces content. `reconstruct(seed)`
returns the stored attractor as-is (zero ticks). `reconstruct(seed, query)`
blends stored with the raw query frame and re-evolves — the warm start
skips the cold path's separate query-evolve cost.

### Temporal Stability (T) as earned rigidity

A fresh basin has no track record of being stable under recall. It might be
a chance correlation in the encoding, a transient pattern, or a real
memory waiting to crystallize. The system has no way to tell yet.

T encodes that uncertainty as plasticity. `T_INIT_DEFAULT = 0.0` — every new
basin starts fully plastic. Each successful recognition runs a one-step CA
probe and measures `observed_stability`. T is updated via EMA:

```
T_new = (1 - T_EMA_RATE) * T_old + T_EMA_RATE * observed_stability
       = 0.9 * T_old + 0.1 * observed_stability
```

Across 5 recalls of a stable basin (with `observed_stability ≈ 1.0`), T
climbs `0 → 0.10 → 0.19 → 0.27 → 0.34 → 0.40` — basins **earn** rigidity
through repeated stable recalls.

### Drift gated by T

When `learning_enabled=True`, the stored attractor drifts toward the
direction the query pulls it:

```
plasticity = (1 - T) * BASIN_DRIFT_BASE_RATE
stored_new = stored + plasticity * (observed - stored)
```

where `observed = apply_ca_dynamics(query_frame)` (one-step probe). T gates
the drift rate. A fresh basin (T=0) drifts at the full
`BASIN_DRIFT_BASE_RATE = 0.02`. A well-recalled basin (T=0.5) drifts at
half that rate. A rigid basin (T→1) is functionally frozen.

This separation is deliberate: recognition is cheap and side-effect-free by
default. Drift only happens when a caller explicitly opts in
(`set_learning_enabled(True)` or `--learn` on the CLI). The default behavior
matches the legacy semantics — stored attractors are immutable on read.

### Why split identity from content?

Two reasons map to two distinct usage patterns:

1. **Cheap basin probes.** Topology mapping, basin-width measurement, and
   A/B evaluation harnesses only need hex_key/chunk handles to load the
   right files — they don't need text, similarity-ranked text, or
   reconstructed attractors. `recognize_top_k()` returns just enough to
   point at the right files.
2. **Warm-start reconstruction.** When content **is** needed, callers can
   reconstruct from a seed without re-running recognition. This skips the
   cold path's query-evolve cost (~2× tick reduction in
   `scripts/bench/bench_recall_warm_vs_cold.py`).

The fast path is honest about its limits — recognition rate at the
production threshold is encoder-dependent, and queries that don't hit a
stored basin (similarity below threshold) legitimately need cold
reconstruction. `recognize()` returns `None` rather than guessing.

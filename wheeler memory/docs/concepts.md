# Wheeler Memory Concepts

## How It Works

```
Input Text
    ↓
SHA-256 → 64×64 seed frame (values in [-1, +1])
    ↓
3-State CA Evolution (iterate until convergence)
    ├→ CONVERGED   → Store attractor + brick
    ├→ OSCILLATING → Epistemic uncertainty detected
    └→ CHAOTIC     → Input needs rephrasing

## Semantic vs. Exact Recall

Wheeler Memory supports two modes of operation:

1. **Exact Recall (SHA-256)**:
   - The default mode.
   - Text is hashed to a seed frame. Changing even one character completely changes the seed (avalanche effect).
   - *Use case*: Exact password-like retrieval, "I want *exactly* this memory."

2. **Semantic Recall (Embedding)**:
   - Enabled via `--embed`.
   - Text is converted to a vector using a sentence-transformer model (e.g., `all-MiniLM-L6-v2`).
   - The 384-dimensional vector is projected onto the 64x64 grid using a fixed random matrix.
   - *Result*: Similar meanings produce similar seed frames.
   - *Use case*: Fuzzy search, "I want memories *like* this one."

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

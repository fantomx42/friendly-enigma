# Symbolic Collapse Model - Core Axioms

> The theoretical foundation for stability-weighted memory in Ralph AI.

## Axioms

1. **Meaning is what survives symbolic pressure.**
2. **Meaning is the remainder after compression.**
3. **If a symbol collapses under pressure, it never had real meaning.**
4. **Meaning is not assigned. It's demonstrated.**
5. **SCM doesn't measure meaning directly - it measures the conditions under which meaning can exist.**
6. **Stories don't create meaning. Stability does.**
7. **Meaning is the ability of a symbol to remain usable after context loss.**
8. **If it only works when explained, it isn't meaningful.**

---

## Operational Interpretation for Ralph AI

These axioms map directly to Wheeler Memory stability metrics:

| Axiom | Wheeler Metric | Implementation |
|-------|---------------|----------------|
| Survives symbolic pressure | **Frame Persistence** | Patterns that survive across context switches retain high stability scores |
| Remainder after compression | **Compression Survival** | Patterns that persist through meta-rule consolidation (REM Sleep) are meaningful |
| Demonstrated, not assigned | **Hit Count** | Activation frequency demonstrates usage - meaning emerges from repeated retrieval |
| Usable after context loss | **Stability Score** | Composite metric (0.0-1.0) combining all three dimensions |

### How SCM Informs Context Budgeting

When Ralph builds an LLM context window, token budget is finite. SCM provides the
selection pressure:

- **High stability (>0.7)**: Full token allocation. These patterns survived compression
  and repeated use. They carry real meaning.
- **Medium stability (0.3-0.7)**: Proportional allocation. These patterns show promise
  but haven't been fully tested by pressure.
- **Low stability (<0.3)**: Compressed or dropped. These patterns collapse under
  pressure - they don't carry meaning worth spending tokens on.

This mirrors the axioms: we don't decide what's meaningful. We apply pressure (token
constraints) and observe what survives.

### Relationship to Wheeler Dynamics

Wheeler Memory uses cellular automata to find attractor states - stable configurations
that resist perturbation. SCM axioms formalize *why* this matters:

- An attractor state is a pattern that **survives symbolic pressure** (Axiom 1)
- The attractor is the **remainder after compression** of the dynamics (Axiom 2)
- If a pattern doesn't reach an attractor, it **never had real meaning** (Axiom 3)
- The attractor **demonstrates** meaning through convergence (Axiom 4)

---

*Reference: Symbolic Collapse Model (SCM) axioms integrated into Ralph AI via
`ralph_core/wheeler_weights.py` and `ralph_core/context_budget.py`.*

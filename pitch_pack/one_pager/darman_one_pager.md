---
marp: true
theme: default
class: invert
paginate: false
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding: 30px 40px;
    columns: 2;
    column-gap: 30px;
  }
  h1 {
    color: #c084fc;
    font-size: 28px;
    margin: 0 0 10px 0;
  }
  h2 {
    color: #67e8f9;
    font-size: 16px;
    margin: 12px 0 6px 0;
  }
  h3 {
    color: #a78bfa;
    font-size: 14px;
    margin: 8px 0 4px 0;
  }
  p {
    margin: 4px 0;
    line-height: 1.3;
  }
  ul {
    margin: 4px 0;
    padding-left: 20px;
  }
  li {
    margin: 2px 0;
    line-height: 1.3;
  }
  code {
    background: #1e1b4b;
    padding: 1px 4px;
    font-size: 11px;
  }
  img {
    max-width: 100%;
    height: auto;
    margin: 6px 0;
  }
---

# Wheeler Memory (Darman)
## Cellular-Automaton Associative Memory — v0.3.6

---

## The Problem

Current AI systems have no persistent memory between conversations. Vector databases retrieve the same row every time. Humans don't — we remember imperfectly, hedge on uncertain memories, and reconstruct differently depending on context.

## The Solution

Wheeler Memory stores text as stable attractor patterns on a 64×64 cellular-automaton grid — no LLM, no pretrained models in the core. Recall isn't retrieval; it's **reconstruction**: find the closest stored pattern, blend it with query context, re-evolve through the CA. Same memory, different reconstructions in different contexts.

As of v0.3.6, recall is **two-tier**: a cheap recognition pass returns identity without engaging the CA loop on the query, and an opt-in reconstruction pass warm-starts the CA from a named basin.

---

## How It Works

```
Text → Encoder → 64×64 frame → CA evolution (~5–14 ticks)
                                       ↓
                            Stable attractor → Store
                                       ↓
Query → Recognition (Pearson scan, no CA on query) → BasinSeed | None
                                       ↓
                Reconstruction (warm-start CA from seed) → Pattern
```

**Three-grid interference (default recall path)**:
`Answer = Corpus × Experiential × (1 − |SCM|)`

- **Corpus**: crystallized knowledge, tight attractors
- **Experiential**: episodic memory, loose attractors, 2-day half-life
- **SCM**: 64×64 trust topology, sculpted by self-consistency feedback

Four interference states: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED.

---

## What's Native (No Pretrained Models)

- **Hippocampus encoder**: character n-gram random indexing (default)
- **Context-RI encoder**: distributional semantics, trained on WikiText-103 + OpenWebText (601M words). SimLex-999 ρ = +0.255 — **first native encoder with positive semantic signal**, 57% of MiniLM's pretrained ceiling (+0.446)
- **Cortex L1/L2/L3**: graph topology → settlement CA → numpy SGD classifier (11K params)
- **Two-tier recall**: per-basin Temporal Stability `T` accumulates via EMA on each `--learn` recall; high-T basins resist drift, low-T basins absorb new context

`sentence-transformers` is optional, only via `.[embed]`.

---

## Numbers (current)

- **Speed**: ~3 ms/tick CPU; **71× speedup** on RX 9070 XT at batch=1000
- **Repository**: 44 modules, 775 tests, 16 CLI commands
- **MMLU all 57 subjects (14,042 questions, test split)**:
  - zero-shot cortex: 24.3% (3,418/14,042) — chance is 25%
  - cortex + learned facts: 25.3% (3,557)
  - cortex + L3 classifier: 25.9% (3,643)
- **Two-tier recall**: ~2× fewer ticks warm-vs-cold across near/mid/far input distance bands (with band-dependent recognition rate)
- **Local-only**: all memories stay on disk, no cloud calls

---

## Honest Caveats

- L3 classifier "barely moved from chance" — needs more training data or richer features. We say so.
- Context-RI captures noun similarity well (ρ=+0.331) but verbs are still the hard case (+0.050). Bag-of-words distributional methods just are like that.
- The interference engine's spatial-vs-scalar SCM rank ordering has a known issue tracked in the roadmap.

The pitch isn't "we beat GPT" — the pitch is "this is real, here's what works, here's what doesn't, and the codebase tells the truth either way."

---

## Try It

```
git clone https://github.com/fantomx42/wheeler-memory.git
pip install -e .
wheeler-store "self-attention computes relationships between all positions"
wheeler-recall "how does attention work in transformers"
wheeler-recall "..." --recognize --learn  # two-tier path with T accumulation
```

Static demo: open `docs/demos/demo.html` in a browser.

**Repo**: https://github.com/fantomx42/wheeler-memory
**License**: CC BY-NC 4.0
**Citation**: Wheeler, J. A. — *Information, Physics, Quantum: The Search for Links* (1989). Loftus on reconstructive memory.

---

## Why It Matters

AI should remember the way humans do — imperfectly, associatively, context-dependent. Current systems retrieve; Wheeler Memory reconstructs. The architecture treats Wheeler's "It from Bit" literally: bits are the substrate, attractor dynamics are the meaning, epistemic states (GROUNDED / ABSORBED / UNCONSOLIDATED / CONTESTED) emerge from interference rather than getting hand-coded.

**"Convergence is ground truth."**

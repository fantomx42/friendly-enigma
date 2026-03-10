---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #c084fc;
  }
  h2 {
    color: #67e8f9;
  }
  code {
    background: #1e1b4b;
  }
---

# Wheeler Memory
## A Memory System That Remembers Like You Do

Darman: Cellular automata meet associative recall.

---

## Why Do You Remember Your First Kiss Differently Every Time?

Context shapes memory. The same event, reconstructed through different lenses.

Same memory, infinite recalls.

That's human memory.

---

## The Problem With Today's AI

ChatGPT has no memory between conversations.
- Never forgets
- Never uncertain
- Never hedges

But humans do all three.

They confabulate confidently. They should say "I'm not sure" and mean it.

---

## What If AI Could Actually Remember?

Forget things? ✓
Hedge on uncertain memories? ✓
Reconstruct context-dependently? ✓

That's Wheeler Memory.

---

## Cellular Automata: Inspired by Conway, Solving Memory

64×64 grid. Three simple rules per cell:
- **Peak** (local max) → push +1
- **Valley** (local min) → push −1
- **Slope** → flow uphill

The grid evolves. Chaos settles into stable patterns. Those patterns ARE memories.

---

## How It Works: Storing a Memory

1. Text enters the system
2. Text is hashed (SHA-256)
3. CA evolves for ~40–100 ticks
4. Chaos converges to a stable pattern (attractor)
5. Pattern is stored as numbers on disk

In ~3 milliseconds. GPU: 10x faster.

---

## How It Works: Recalling a Memory

1. You ask a question (query)
2. System finds the closest stored memory (Pearson correlation)
3. Blends it with your query context (30% query, 70% stored)
4. Re-evolves the blend through CA
5. Out comes a reconstructed answer

Context colors the recall.

---

## The Magic: Reconstruction

Same stored memory.
Different query contexts.
Different reconstructed outputs.

**Same memory. Infinite variations. Like your brain.**

---

## Temperature: Forgetting Built In

Memories have a lifespan:
- **New** (cool) — "I vaguely remember..."
- **Warm** (frequently used) — "I think..."
- **Hot** (recent) — "I distinctly remember..."

Half-life: 7 days without recall.
Decay matters. Age matters.

---

## Epistemic Humility: The LLM Mirrors Memory Quality

**Hot memory** (≥0.6 temperature):
> "I remember clearly that..."

**Warm memory** (≥0.3):
> "I believe... but I'm not entirely certain..."

**Cold memory** (<0.3):
> "I vaguely recall... it might have been..."

Confidence reflects freshness. The system admits uncertainty.

---

## Why This Matters

Current LLMs confabulate confidently.

Wheeler Memory lets AI say **"I'm not sure"** and **mean it**.

Grounded in theory (Wheeler's information dynamics).
Validated in practice (10K+ test cases).

---

## Not Just Search

**Vector DBs** (traditional): Find stored text that matches the query.
- Retrieval-based
- Exact-ish match
- No reconstruction

**Wheeler Memory**: Find the closest pattern, blend it with your context, re-evolve it.
- Reconstruction-based
- Context-colored
- Meaning emerges from dynamics

---

## Two Modes: Exact + Fuzzy Recall

**Exact**: Store "buy milk". Query "buy milk". Get the exact memory back.

**Fuzzy**: Store "grocery list". Query "shopping". Get a context-colored reconstruction.

Same system. Two recall philosophies.

---

## The Philosophical Foundation

Wheeler: "**It from Bit**" — Information emerges from dynamics, not storage.

A memory isn't retrieved. It's **computed**. Reconstructed. Born anew each time.

Like you. Every recall rewrites the memory.

---

## Design Philosophy: Engine ≠ Voice

**Engine**: The cellular automaton (the mind).
**Voice**: The LLM wrapper (the speaker).

Swap the LLM. The memory system stays the same.
The behavior persists.

Meaning comes from structure, not labels.

---

## Privacy & Ownership

All local. No cloud APIs. No third-party servers.

Your memories stay on your machine.

Your data. Your automaton. Your voice.

---

## Real Numbers

- **10K+** test cases across 6 domains
- **95%+** paraphrase coverage (semantic recall)
- **~3ms** CPU (convergence)
- **GPU**: 10× faster (HIP/ROCm + CUDA)
- **19 modules**, 167 tests, fully open-source

Validated. Tested. Ready.

---

## What's Running Today

✓ Web UI dashboard
✓ CLI tools (store, recall, temps, scrub)
✓ Darman chatbot agent
✓ Sleep consolidation (spreading activation)
✓ Semantic embeddings (optional)
✓ GPU auto-fallback

Everything works locally. Nothing leaves your machine.

---

## Live Demo

*[Walkthrough of Web UI → Darman agent → CLI tools → under the hood]*

Watch memories being stored, recalled, and reconstructed in real time.

---

## What's Next

- **Images & Audio** — Extend to multimodal memories
- **Concept Clustering** — Organize related memories
- **Federated Memory** — Distributed learning between trusted peers
- **WebAssembly** — Run in the browser

The foundation is proven. Now we build.

---

## Open Source

**GitHub**: [fantomx42/wheeler-memory](https://github.com/fantomx42/wheeler-memory)

**License**: CC BY-NC 4.0 (Non-Commercial Creative Commons)

**Community**: Contributions welcome.

---

## Why This Excites Us

First-of-a-kind memory architecture for AI.

Rooted in cellular automata theory.
Validated in practice.
Grounded in cognitive science (Elizabeth Loftus on reconstructive memory).

AI with a mind that remembers like you do.

---

## The Formula Is The Foundation

*Everything else is commentary.*

Wheeler Memory: Not a search engine.
Not a retrieval system.
A **reconstruction engine**.

A mind. An automaton. A voice.

Darman remembers.

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
## Cellular Automata Associative Recall

---

## The Problem

Current AI systems have no persistent memory between conversations. They retrieve confidently without admitting uncertainty, confabulating when they don't know. **Humans do the opposite**: we remember imperfectly, hedge on uncertain memories, and reconstruct differently depending on context.

## The Solution

Wheeler Memory is a cellular automata–based associative memory system that stores memories as stable attractor patterns on a 64×64 grid. Recall isn't retrieval—it's **reconstruction**: the system finds the closest stored pattern, blends it with query context (30% new, 70% stored), and re-evolves it through the automaton. Same memory, different reconstructions depending on context, just like human recall.

---

## How It Works

```
Text → Hash → CA Evolution (40–100 ticks)
  ↓
Stable Attractor (pattern) → Store
  ↓
Query → Find Closest (Pearson) → Blend (α=0.3)
  ↓
Re-Evolve → Reconstruct Answer
```

**Three CA Rules per Cell:**
- Peak (local max) → +1
- Valley (local min) → −1
- Slope → flow uphill

**Temperature-Based Decay:**
- Hot (≥0.6): "I remember clearly..."
- Warm (≥0.3): "I believe... but I'm not sure..."
- Cold (<0.3): "I vaguely recall..."
- Half-life: 7 days without recall

---

## Key Facts

- **Speed**: ~3ms CPU, 10× faster on GPU (HIP/ROCm + CUDA)
- **Local-Only**: No cloud APIs; all memories stay on your machine
- **19 Modules, 167 Tests**: Fully tested across 6 domains (code, hardware, daily_tasks, science, meta, general)
- **95%+ Paraphrase Coverage**: Semantic embeddings optional; core recall works without ML
- **Reconstructive, Not Retrieval**: Context-colored recall; same memory yields different outputs
- **Epistemic Humility**: Temperature reflects confidence; system admits uncertainty

---

## Validation

- **10K+ Test Cases**: Across MBPP, SWE-bench, BABILong, and domain-specific datasets
- **Diversity Report**: 95%+ coverage of paraphrases and semantic variations
- **Reconstruction Fidelity**: Blended memories preserve core semantics while adapting to query context
- **GPU Performance**: Demonstrated 10× speedup on ROCm; auto-fallback to CPU

---

## Roadmap

**Done:**
- Core CA engine & attractor storage
- Pearson correlation–based recall
- Temperature-based decay
- Web UI dashboard
- CLI tools (store, recall, temps, scrub)
- Darman chatbot agent
- GPU support (HIP/ROCm + CUDA)
- Semantic embeddings (optional)

**Next:**
- Multimodal (images & audio)
- Concept clustering
- Federated memory
- WebAssembly browser runtime

---

## Open Source

**GitHub**: [fantomx42/wheeler-memory](https://github.com/fantomx42/wheeler-memory)

**License**: CC BY-NC 4.0 (Non-Commercial Creative Commons)

**Citation**: Wheeler, J. A. (1989). *It from Bit*. In *A Brief History of Time* (Ed. S. Hawking). Elizabeth Loftus on reconstructive memory.

**Community**: Contributions welcome. All local, no cloud dependency.

---

## Why This Matters

AI should remember **like you do**—imperfectly, associatively, context-dependent. Current systems retrieve; Wheeler Memory reconstructs. It's the first architecture to combine cellular automata theory with associative recall, giving AI a mind that admits uncertainty and adapts recall to context.

**"The formula is the foundation. Everything else is commentary."**

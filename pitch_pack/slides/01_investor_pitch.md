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
    padding: 2px 6px;
    border-radius: 3px;
  }
  table {
    font-size: 0.9em;
  }
---

# Wheeler Memory
## Project Darman

**A memory system that remembers like you do — imperfectly, associatively, and influenced by context.**

---

## The Problem

### Why Memory Matters in LLMs

- **Confabulation**: Current LLMs hallucinate confidently with zero uncertainty signals
- **Statelessness**: No memory = no personalization, no context persistence across sessions
- **Information Loss**: Context windows close; conversational history is discarded
- **Enterprise Risk**: Mission-critical applications can't rely on models that don't know what they don't know

**Today's LLMs are amnesiacs who don't admit it.**

---

## Market Opportunity

### The $100B+ LLM Market Needs Memory

| Market Segment | Size | Pain Point |
|---|---|---|
| Enterprise Copilots | $30B+ | Need reliable recall for customer history, docs |
| AI Search/RAG | $20B+ | Vector DBs don't reconstruct; retrieval ≠ memory |
| Agentic Systems | $25B+ | Agents need persistent state, forgetting curves |
| Personalized LLMs | $25B+ | User models require associative recall |

**Memory is the missing layer.** Every major LLM platform (OpenAI, Anthropic, Google) is exploring it. The first production-ready system wins.

---

## Our Solution: Wheeler Memory

### A Cellular-Automaton Memory Engine

**Text → CA Attractor → Correlation Search → Context-Dependent Reconstruction**

- **Unique**: Uses 3-state cellular automaton instead of vectors
- **Reconstructive**: Every recall is context-dependent (like human memory)
- **Temperature-Aware**: Explicit confidence tiers prevent confabulation
- **Local-First**: Runs entirely on-device; no cloud dependency
- **GPU-Optional**: Works on CPU; accelerates on HIP/ROCm/CUDA

**"Darman doesn't retrieve. Darman reconstructs."**

---

## How It Works (Simplified)

```
Text Input
    ↓
SHA-256 hash (or semantic embedding)
    ↓
64×64 grid seed
    ↓
3-State CA Evolution (40-100 ticks)
    ↓
Stable Attractor Pattern
    ↓
Search: Pearson correlation against query
    ↓
Reconstruct: Blend stored + query, re-evolve
    ↓
Result: Context-dependent memory
```

**Convergence: ~3 ms on CPU. Deterministic. Reproducible.**

---

## Key Differentiator #1: Reconstructive Recall

### Darman Reconstructs, Not Retrieves

Same stored memory reconstructs differently depending on context:

```
blend = (1 - α) × stored + α × query    (α = 0.3)
reconstructed = evolve_and_interpret(blend)
```

- Query "machine learning" vs. "debugging" → different reconstructions
- **Like human memory** (Elizabeth Loftus, cognitive psychology)
- **Prevents database-like rigidity** — memories stay alive and contextual

**Competitors** (vector DBs, RAG): Static retrieval. Same result every time. No context dependency.

---

## Key Differentiator #2: Temperature System

### Epistemic Humility — Confidence Built In

```
temp = base_from_hits × decay_from_time
Half-life: 7 days
```

| Tier | Temp Range | Darman's Language |
|---|---|---|
| **Hot** | ≥ 0.6 | "I clearly remember…" |
| **Warm** | ≥ 0.3 | "I think we discussed…" |
| **Cold** | < 0.3 | "I vaguely recall, but I'm uncertain…" |

- **Prevents confabulation**: LLM can't sound confident about stale memories
- **Competitive advantage**: Customers know when to trust the system
- **Explainable AI**: Confidence is auditable, not a black box

---

## Key Differentiator #3: Local-First, Privacy-Preserving

### No Cloud. No API Keys. No Data Escape.

- **All computation on-device**: CPU or local GPU
- **All storage local**: `~/.wheeler_memory/` by default
- **No vendor lock-in**: Sentence-transformers runs locally; Ollama serves LLM locally
- **Enterprise-ready**: HIPAA, GDPR, CFAA compliance by architecture

**Market advantage**: Enterprises demand data sovereignty. We deliver it.

---

## Technical Validation

### Proof Points

- **167+ automated tests** covering CA dynamics, recall, reconstruction, temperature decay
- **3 major datasets**: MBPP (code), SWE-bench (software engineering), BABILong (long-context reasoning)
- **Diversity validation**: Rotation retry (0°/90°/180°/270°) ensures stability across seed angles
- **Reconstruction demo**: Semantic recall on embedding-based paraphrases
- **GPU benchmarks**: HIP/ROCm + CUDA with auto-fallback

**All tests reproducible. All benchmarks public.**

---

## Architecture: 19 Modules, Deep Stack

```
Agent Loop (Ollama integration)
    ↓
Chunking & Routing (6 domain-specific stores)
    ↓
Reconstruction Engine (context-dependent blending)
    ↓
Temperature Dynamics (decay + confidence tiers)
    ↓
Storage & Recall (Pearson correlation)
    ↓
3-State CA Dynamics (local max/min/slope rules)
```

- **Modular**: Swap any layer without touching others
- **Testable**: Each module has isolated test suite
- **Extensible**: Custom embeddings, hashing, GPU kernels

---

## Engine Philosophy

### "The Engine Is the Mind. The LLM Is the Voice."

**Key insight**: Swap the underlying model (GPT-4 → Mistral → local 7B) and Darman's *behavior* stays the same. Only *phrasing* changes.

**Why this matters**:
- Behavior is deterministic, auditable, provable
- Not dependent on model capability
- Cheaper to operate at scale (smaller models work)
- Personality emerges from memory, not from prompt engineering

**Enterprise value**: Darman's identity is stable regardless of LLM upgrade.

---

## Design Principles

### 7 Axioms That Drive Everything

1. **Engine is the mind** — behavior from CA, phrasing from LLM
2. **Reconstructive recall** — context-dependent, not static lookup
3. **Temperature is epistemic humility** — uncertainty is computed, not guessed
4. **Memories are suggestions** — LLM can disagree with stale memories
5. **Minimize LLM dependency** — engine handles search/decay/reconstruction
6. **Local-only by default** — cloud is optional, not required
7. **Formula is the foundation** — CA rule is load-bearing; UI is commentary

These principles prevent feature creep and ensure the system stays coherent as it scales.

---

## Competitive Positioning

### Wheeler vs. Alternatives

| Aspect | Vector DB | RAG | Wheeler Memory |
|---|---|---|---|
| Retrieval | Static | Static | **Context-Dependent** |
| Confidence Signal | None | None | **Temperature Tier** |
| Reconstruction | No | No | **Yes** |
| Local-First | Sometimes | Sometimes | **Always** |
| Forgetting | Manual | Manual | **Automatic (7d)** |
| Associative Recall | No | No | **Yes** |
| Cost at Scale | Linear | Linear | **Sublinear** |

**Market position**: Not an upgrade to existing memory systems. A new category.

---

## Traction & MVP

### Launch Status

- ✓ **Core CA engine**: 19 modules, production-ready
- ✓ **Automated testing**: 167+ tests across datasets (MBPP, SWE-bench, BABILong)
- ✓ **Web UI**: Wheeler dashboard with live recall, temperature display
- ✓ **Darman chatbot**: Live demo, integrated with Ollama
- ✓ **GPU support**: HIP/ROCm + CUDA with CPU fallback
- ✓ **Open source**: GitHub live, CC BY-NC 4.0, community-ready

**No VC funding yet. Built by founding team. Ready to accelerate.**

---

## Named After Wheeler's "It from Bit"

> "Every 'it' — every physical thing — derives its existence as an information system from answered physical questions."
> — John Archibald Wheeler

**Insight**: In our system, meaning emerges from dynamics. Information isn't stored; it's *computed*. Memories exist because the CA *interacts* with them through repeated recall.

**Philosophy**: Information emerges from physical process, not from static representation.

---

## Open Source, Commercial Ready

### GitHub: `fantomx42/wheeler-memory`

- **License**: CC BY-NC 4.0 (open source, non-commercial)
- **Quick start**: 3 commands to running instance
- **Documentation**: Architecture, CLI, API, design principles
- **Community**: Issues, discussions, contribution guidelines
- **Integration ecosystem**: Ollama, Open WebUI, Hugging Face

**Path to commercialization**: Dual licensing, managed service, enterprise integrations.

---

## Closing: "The Formula Is the Foundation"

### Call to Action

**We've built the engine. The market needs it. Let's scale it.**

- **For investors**: Memory is the missing layer in the $100B LLM market
- **For enterprises**: Privacy-first, on-device, auditable confidence signals
- **For the market**: A new category, not an incremental improvement

**Next steps**:
1. Seed round: 18-month runway to commercial product
2. Enterprise partnerships: First 3 pilot customers
3. Open source acceleration: Grow contributor base
4. Managed service: Wheeler-as-a-Service for API consumers

**"Darman doesn't retrieve. Darman reconstructs."**

---

# Demo + Questions

**Try it now**:
```bash
git clone https://github.com/fantomx42/wheeler-memory.git
pip install -e .
wheeler-ui
# Open http://localhost:7437
```

**Contact**: [Your email / website]

---

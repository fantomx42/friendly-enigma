# Wheeler Memory Pitch Pack Blueprint

> **This is the permanent reference for all pitch materials.**
> Use it across sessions to track what exists, what each deliverable contains, and how to build/rebuild PDFs.

---

## Quick Reference

| Deliverable | File | Build | Audience |
|---|---|---|---|
| Investor Deck | `slides/01_investor_pitch.md` | `make slides` | VCs, biz dev |
| Developer Deck | `slides/02_developer_pitch.md` | `make slides` | OSS contributors, researchers |
| General Deck | `slides/03_general_pitch.md` | `make slides` | Journalists, general tech |
| One-Pager | `one_pager/darman_one_pager.md` | `make one-pager` | Email attachment, handout |
| Whitepaper | `whitepaper/wheeler_memory_whitepaper.md` | `make whitepaper` | Academic, deep-dive |
| Demo Script | `demo_script/demo_script.md` | `make demo` | Presenter reference |
| **All PDFs** | `output/*.pdf` | **`make all`** | — |

### Build Prerequisites

```bash
npm install -g @marp-team/marp-cli   # or: npx marp (if installed locally)
sudo pacman -S pandoc-cli             # for whitepaper + demo script PDFs
```

---

## Key Messaging Cheat Sheet

Use these verbatim across all materials for consistency:

| Tag | Message |
|---|---|
| **Tagline** | "A memory system that remembers like you do — imperfectly, associatively, and influenced by context." |
| **Core thesis** | "The engine is the mind. The LLM is the voice." |
| **Differentiator** | "Darman doesn't retrieve. Darman reconstructs." |
| **Theory** | "Meaning is what survives symbolic pressure." |
| **Closing** | "The formula is the foundation. Everything else is commentary." |
| **Epistemic humility** | Hot: "I remember..." / Warm: "I think..." / Cold: "I vaguely recall..." |
| **Failure mode** | "LLMs confabulate confidently because they have no mechanism to express uncertainty about memory." |
| **Privacy** | "All local. No cloud APIs. Your memories stay on your machine." |

---

## Visual Asset Inventory

All paths relative to project root (`/home/tristan/projects/wheeler-memory/`).

| Asset | Path | Use In |
|---|---|---|
| CA evolution animation | `docs/assets/diagrams/evolution.gif` | All decks (title/intro) |
| Phase 2 verification | `docs/assets/diagrams/phase_2_verification.png` | Dev deck, whitepaper |
| Diversity report (10K GPU) | `docs/assets/reports/diversity_report_math_10k_gpu.png` | Validation slides, whitepaper |
| Diversity report (CPU) | `docs/assets/reports/diversity_report_math.png` | Whitepaper appendix |
| Paraphrase report | `docs/assets/reports/paraphrase_report.png` | Dev deck, whitepaper |
| Paraphrase + embedding | `docs/assets/reports/paraphrase_embed_report.png` | All validation slides |
| Reconstruction demo | `docs/assets/reports/reconstruction_demo.png` | Reconstruction slides, whitepaper |

### Missing Assets (create if needed)

- [ ] Architecture stack diagram (SVG): UI → Agent → Engine → CA
- [ ] Recall flow diagram: Query → Hash → Correlation → Reconstruction
- [ ] Temperature decay graph: temp vs. days since access
- [ ] UI screenshots: dashboard, chat, scrub tool

---

## Deck 1: Investor / VC Pitch (~17 slides)

**File:** `slides/01_investor_pitch.md`
**Tone:** ROI-forward, market opportunity, technical credibility
**Omitted:** Team, funding ask, business model (per user request)

### Slide-by-Slide Outline

| # | Title | Content |
|---|---|---|
| 1 | **Title** | Wheeler Memory • Project Darman • tagline • evolution.gif |
| 2 | **The Problem** | LLMs confabulate confidently; no memory uncertainty mechanism; no associative recall; humans remember contextually |
| 3 | **Market Opportunity** | $100B+ LLM inference market; memory layer is whitespace; AGI requires persistent episodic memory; enterprise copilots need accurate recall |
| 4 | **Our Solution** | CA-based memory engine; context-dependent reconstructive recall; temperature uncertainty; local-first, privacy-preserving |
| 5 | **How It Works** | Simplified pipeline: Input → CA engine → Attractor fingerprint → Correlation search → Reconstruct in context |
| 6 | **Differentiator: Reconstruction** | "Darman doesn't retrieve. Darman reconstructs." Same memory surfaces differently based on query context. Like human episodic memory. |
| 7 | **Differentiator: Epistemic Humility** | Temperature system prevents confabulation. Hot/warm/cold tiers guide LLM confidence language. No more "I clearly remember" on stale data. |
| 8 | **Differentiator: Local-First** | No cloud APIs required. GPU-optional. Privacy-respecting. Can run on edge devices. Full data sovereignty. |
| 9 | **Technical Validation** | 167+ tests passing; validated on MBPP, SWE-bench, BABILong; diversity reports show 95%+ hit rates; show report screenshots |
| 10 | **Architecture** | 19 core modules; 3-state CA on 64×64 grid; converges in ~3ms; HIP/ROCm + CUDA; Ollama-native LLM integration |
| 11 | **Engine Philosophy** | "The engine is the mind. The LLM is the voice." Behavior from engine, phrasing from model. Swap models; behavior stays. |
| 12 | **Design Principles** | 7 axioms (highlight top 3-4): engine is mind, reconstructive default, temperature = humility, formula is foundation |
| 13 | **Competitive Positioning** | vs Pinecone/Weaviate (vector DBs — retrieval, no reconstruction); vs RAG (no context blending, no forgetting); vs nothing (first-to-market for reconstructive memory) |
| 14 | **Traction** | MVP complete; 19 modules in production; 167 tests; Web UI shipped; Darman chatbot live with streaming; auto-recall/auto-store loop |
| 15 | **Origin Story** | Named after Wheeler's "It from Bit" — information emerges from dynamics. Implements Symbolic Collapse Model. Grounded in Loftus episodic memory research. |
| 16 | **Open Source** | GitHub: fantomx42/wheeler-memory; CC BY-NC 4.0; community contributions welcome; full documentation |
| 17 | **Closing** | "The formula is the foundation. Everything else is commentary." Call to action: demo invite, GitHub link |

---

## Deck 2: Developer / OSS Pitch (~20 slides)

**File:** `slides/02_developer_pitch.md`
**Tone:** Technical, collaborative, "build together"

### Slide-by-Slide Outline

| # | Title | Content |
|---|---|---|
| 1 | **Title** | Wheeler Memory: Open-Source Cellular-Automata Memory for LLMs |
| 2 | **Why This Matters** | LLM memory unsolved; vector DBs ≠ episodic memory; reconstruction is novel; publishing the whole stack |
| 3 | **What You Can Build** | Agentic systems with true forgetting; contextual chatbots; semantic search with meaning; fair memory evaluation |
| 4 | **Quick Start** | 3 commands: `git clone`, `pip install -e .`, `wheeler-ui` → localhost:7437. CPU works, no GPU needed. |
| 5 | **Two Recall Modes** | Exact (SHA-256, deterministic, avalanche); Semantic (sentence-transformers, fuzzy, meaning-based). Both pluggable. |
| 6 | **The Science** | Symbolic Collapse Model: "Meaning survives symbolic pressure." Irreversible CA evolution. IIT connections. |
| 7 | **3-State CA Rule** | Table: local max → +1 (0.35), local min → -1 (0.35), slope → flow uphill (0.20). Von Neumann neighborhood. Convergence in 39-49 ticks (~3ms). |
| 8 | **Temperature & Forgetting** | Formula: `base_from_hits × decay_from_time`. Half-life = 7 days. Hot/warm/cold tiers. Prevents confabulation. |
| 9 | **Reconstructive Recall** | `blend = (1-α)×stored + α×query` (α=0.3). Re-evolve through CA. Context-dependent like Loftus research. |
| 10 | **Validation** | 19 modules; 167+ tests; MBPP/SWE-bench/BABILong datasets; diversity + paraphrase reports. Show report images. |
| 11 | **Architecture Deep Dive** | Module dependency tree: dynamics.py → storage.py → reconstruction.py → chunking → polarity → agent. GPU dispatch via gpu_dynamics.py. |
| 12 | **Extensibility** | Pluggable hashing (SHA-256, embeddings, custom); custom chunks; rotation retry for edge cases; attention budgets |
| 13 | **Testing & Benchmarks** | Rotation stats; oscillation detection; brick visualization (wheeler-scrub); GPU vs CPU benchmarks |
| 14 | **Web UI & CLI** | 10 CLI tools listed; web dashboard screenshot; Open WebUI pipeline; streaming chat interface |
| 15 | **Integration Points** | Ollama (local LLM); Open WebUI (plugin); Hugging Face (embeddings); Discord/Slack bots; LangChain / agentic frameworks |
| 16 | **Roadmap** | Multimodal (images, audio); concept clustering; distributed memory (federation); browser-side (WebAssembly) |
| 17 | **Contribution Areas** | GPU kernels (CUDA, Metal); multimodal encoders; distributed protocols; more dataset benchmarks; language bindings |
| 18 | **Philosophy** | "The formula is the foundation." Don't fork, improve upstream. Breaking changes documented. Backward compat enforced. |
| 19 | **Community** | GitHub discussions; documentation at multiple levels; issue tracker; all code MIT-accessible |
| 20 | **Call to Action** | Star the repo; open an issue; contribute; read the whitepaper |

---

## Deck 3: General / Mixed Audience Pitch (~24 slides)

**File:** `slides/03_general_pitch.md`
**Tone:** Accessible, story-driven, philosophical, wonder-filled

### Slide-by-Slide Outline

| # | Title | Content |
|---|---|---|
| 1 | **Title** | Wheeler Memory: A Memory System That Remembers Like You Do |
| 2 | **Hook** | "Why do you remember your first kiss differently every time you think about it?" |
| 3 | **The Problem** | ChatGPT has no memory between conversations. It never forgets or gets confused. But you do. Humans remember contextually, fading with time. |
| 4 | **What If?** | What if AI could actually remember — forgetting unused things, hedging on uncertain memories, seeing old context color new ones? |
| 5 | **Enter: Cellular Automata** | Inspired by Conway's Game of Life, but solving memory. A 64×64 grid of evolving cells that settle into stable patterns called *attractors*. |
| 6 | **The CA Explained** | Three jobs per cell: peak → push to +1; valley → push to -1; slope → flow uphill. Converges in ~3 milliseconds. Final pattern = memory fingerprint. |
| 7 | **Storing a Memory** | Text → hash to seed → CA evolution → stable attractor → store as pattern |
| 8 | **Recalling a Memory** | Query → evolve to attractor → find closest stored pattern (Pearson correlation) → blend with context → re-evolve → reconstructed memory |
| 9 | **Key Insight: Reconstruction** | Same memory, different reconstructions. "Machine learning" vs "debugging" against the same stored thought → different shapes. Like *your* memories. |
| 10 | **Temperature: Forgetting** | New memories cool (~0.3), warm after use (~0.6), hot after frequent use (≥0.6). After 7 days without recall, temperature halves. |
| 11 | **Epistemic Humility** | Hot: "I *remember* discussing X..." / Warm: "I *think* we touched on X..." / Cold: "I *vaguely* recall X, but I'm uncertain..." |
| 12 | **Why This Matters** | Current LLMs confabulate confidently. This system lets AI say "I'm not sure" and actually mean it. |
| 13 | **Not Just Search** | Vector databases = retrieval (pull exact matches). Wheeler Memory = reconstruction (context-colored, memory-dominant, shaped by now). |
| 14 | **Two Modes** | Exact: same text always finds itself. Fuzzy: "grocery list" finds "remind me to buy food" because embeddings capture *meaning*. |
| 15 | **The Philosophical Foundation** | John Archibald Wheeler's "It from Bit" — information emerges from physical-like dynamics. |
| 16 | **Design Philosophy** | "The engine is the mind. The LLM is the voice." Swap models; behavior stays. Only phrasing changes. |
| 17 | **Privacy & Ownership** | All local. No cloud APIs. Your memories stay on your machine (or your company's servers). |
| 18 | **Real Numbers** | 10K+ test cases; 95%+ paraphrase coverage; ~3ms per memory on CPU; GPU = 10x faster |
| 19 | **What's Running Today** | 19 modules, 167 tests, Web UI, Darman chatbot live (remembers, forgets, hedges) |
| 20 | **Live Demo** | [Placeholder for live walkthrough — see demo script] |
| 21 | **What's Next** | Images & audio; concept clustering; federated memory; browser-side (WebAssembly) |
| 22 | **Open Source** | GitHub: fantomx42/wheeler-memory; CC BY-NC; community contributions welcome |
| 23 | **Why This Excites Us** | First-of-kind memory architecture for AI; rooted in theory, validated in practice; opens new categories |
| 24 | **Closing** | "The formula is the foundation. Everything else is commentary." |

---

## One-Pager

**File:** `one_pager/darman_one_pager.md`
**Format:** Single Marp slide, dense layout, fits on one printed page

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  WHEELER MEMORY: A Memory System That Remembers Like You Do │
│                       Project Darman                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  THE PROBLEM                                                 │
│  LLMs have no persistent memory and can't express            │
│  uncertainty. Humans remember imperfectly, contextually,     │
│  and forget over time. AI should too.                        │
│                                                              │
│  THE SOLUTION                                                │
│  Cellular automata engine evolves text into stable            │
│  "attractor" fingerprints. Memories fade unless reinforced.  │
│  Same memory reconstructs differently per context.           │
│                                                              │
│  HOW IT WORKS                                                │
│  Text → SHA-256 → 64×64 CA seed → Evolution → Attractor     │
│                                                              │
│  KEY FACTS                                                   │
│  • 64×64 CA grid, converges in ~3ms (CPU)                    │
│  • Temperature = freshness (hot/warm/cold tiers)             │
│  • Reconstructive: context-dependent recall                  │
│  • Semantic & exact search modes                             │
│  • Local-only, GPU-optional, privacy-first                   │
│  • 19 modules, 167+ tests, production-ready                  │
│                                                              │
│  VALIDATION                                                  │
│  MBPP, SWE-bench, BABILong datasets ✓                        │
│  95%+ paraphrase family coverage ✓                           │
│  Semantic embedding search validated ✓                       │
│  Reconstruction drift measured ✓                             │
│                                                              │
│  ROADMAP                                                     │
│  ✓ Core CA engine         → Multimodal (images, audio)       │
│  ✓ Web UI + CLI           → Distributed memory (federation)  │
│  ✓ Darman agent           → Browser-side (WebAssembly)       │
│                                                              │
│  github.com/fantomx42/wheeler-memory                         │
│  CC BY-NC 4.0 | Python 3.11+ | Ollama-native                │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Whitepaper

**File:** `whitepaper/wheeler_memory_whitepaper.md`
**Format:** Academic paper, ~30-40 pages, pandoc → PDF

### Chapter Outline

#### Title
"Wheeler Memory: A Cellular Automata Approach to Reconstructive Episodic Memory for Language Model Agents"

#### Abstract (200-300 words)
Problem → Approach → Key results → Implications

#### 1. Introduction (2-3 pages)
- Motivation: LLM memory is retrieval-based, not reconstructive
- The gap: no epistemic uncertainty in AI memory
- Contributions: SCM, CA architecture, temperature, reconstructive recall, validation
- Paper roadmap

#### 2. Related Work (3-4 pages)
- 2.1 Vector Databases (Pinecone, Weaviate, Qdrant)
- 2.2 Retrieval-Augmented Generation (RAG)
- 2.3 Memory in Neuroscience (Loftus, episodic vs semantic)
- 2.4 Cellular Automata in Computing (Conway, Wolfram)
- 2.5 Information Theory & Physics (Wheeler, IIT)

#### 3. Methodology (6-8 pages)
- 3.1 Symbolic Collapse Model (SCM): axiom, formal definition, irreversibility
- 3.2 3-State CA Rule: Von Neumann, gradient rule, convergence criteria
- 3.3 Memory Representation: MemoryBrick, .npy/.npz, JSON index
- 3.4 Similarity & Retrieval: Pearson correlation, temperature formula, ranking
- 3.5 Reconstructive Recall: blend formula, re-evolution, correlation metrics
- 3.6 Temperature System: base × decay, half-life derivation, tiers
- 3.7 Rotation Retry: algorithm, success tracking

#### 4. System Architecture (4-5 pages)
- 4.1 Module Structure (19 modules)
- 4.2 Chunked Storage (domain routing)
- 4.3 Spreading Activation (warmth propagation)
- 4.4 Dual-Polarity Encoding (geometric negation)
- 4.5 Sleep Consolidation
- 4.6 GPU Backend (HIP/ROCm, CUDA)
- 4.7 Agent Integration (Ollama, tool loop)

#### 5. Experimental Validation (6-8 pages)
- 5.1 Test Suite Overview
- 5.2 Convergence Properties
- 5.3 Diversity Validation
- 5.4 Paraphrase & Embedding Validation
- 5.5 Reconstruction Properties
- 5.6 Temperature Validation

#### 6. Discussion (3-4 pages)
- 6.1 Strengths
- 6.2 Limitations (O(n) search, scalability, no multimodal yet)
- 6.3 Theoretical Implications (SCM, IIT, neuroscience)
- 6.4 Comparison to Baselines

#### 7. Conclusion & Future Work (2-3 pages)

#### Appendices
- A: Algorithm Pseudocode
- B: Mathematical Details (temperature derivation, Pearson properties)
- C: Implementation Notes (NumPy, HIP kernel, chunking tables)
- D: References (Wheeler, Loftus, Wolfram, Tononi, Conway, Lewis et al.)

---

## Demo Script

**File:** `demo_script/demo_script.md`
**Format:** Plain markdown, two parts

### Part 1: Presentation Talking Points

Slide-by-slide notes (2-3 minutes per slide). For each slide:
- Slide title
- Key talking points (what to say)
- Transition to next slide
- Suggested visual aid

### Part 2: Live Walkthrough (10-15 minutes total)

| Section | Duration | Commands | Purpose |
|---|---|---|---|
| Web UI Dashboard | 2-3 min | `wheeler-ui` → localhost:7437 | Store a memory, recall it, show temperature |
| CLI Tools | 3-4 min | `wheeler-store`, `wheeler-recall`, `wheeler-temps`, `wheeler-scrub` | Demonstrate full CLI toolkit |
| Darman Agent | 3-4 min | `wheeler-agent --interactive` | Show auto-recall, conversation with memory |
| Under the Hood | 2-3 min | Show evolution.gif, explain convergence | Technical deep-dive for curious audience |

### Troubleshooting Notes

| Issue | Fix |
|---|---|
| `wheeler-ui` won't start | Check port 7437 isn't in use: `lsof -i :7437` |
| Ollama not responding | `ollama serve` in another terminal, check `http://localhost:11434` |
| No memories found | Store something first: `wheeler-store "test memory"` |
| GPU not detected | Expected — CPU fallback is automatic. Run `wheeler-info` to check. |

---

## Marp Theme Reference

All slide decks use this front matter for consistent dark theme:

```yaml
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
```

**Colors:** Violet headings (#c084fc), cyan subheadings (#67e8f9), dark code blocks (#1e1b4b).
**Font:** Segoe UI primary, monospace for code.
**Pagination:** Enabled, bottom-right.

---

## Source Material Cross-Reference

| Source Document | Path | Feeds Into |
|---|---|---|
| README.md | `README.md` | All decks, one-pager |
| Concepts | `docs/concepts.md` | Whitepaper ch.1-3, general deck |
| Design Principles | `docs/design.md` | All decks, whitepaper ch.3 |
| Architecture | `docs/architecture.md` | Dev deck, whitepaper ch.3-4 |
| CLI Reference | `docs/cli.md` | Dev deck, demo script |
| API Reference | `docs/api.md` | Dev deck, whitepaper appendix |
| GPU Guide | `docs/gpu.md` | Dev deck, whitepaper ch.4.6 |
| Install Guide | `docs/install.md` | Dev deck quick-start slide |
| Future Roadmap | `docs/future.md` | All decks roadmap slides |

---

## Checklist

- [ ] Investor deck written and renders to PDF
- [ ] Developer deck written and renders to PDF
- [ ] General deck written and renders to PDF
- [ ] One-pager written and renders to PDF
- [ ] Whitepaper written and renders to PDF (via pandoc)
- [ ] Demo script written and renders to PDF (via pandoc)
- [ ] `make all` succeeds with no errors
- [ ] All image references resolve (no broken images in PDFs)
- [ ] Key messaging is consistent across all materials
- [ ] BLUEPRINT.md committed to repo

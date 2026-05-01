---
title: Wheeler Memory Demo Script
author: Project Darman
date: April 2026 (v0.3.6)
---

# Wheeler Memory Demo Script

## Part 1: Presentation Talking Points

### Slide 1: Title
**"Wheeler Memory: A memory system that remembers like you do."**

**Talking Points (30 seconds):**
- Welcome. This is a story about memory — human memory, artificial memory, and how they're more similar than you'd think.
- Wheeler Memory is a cellular-automaton associative memory system. No LLM in the core, no pretrained models in the core. Pure dynamics.
- Named after John Wheeler's "It from Bit": information emerges from dynamics, not from static storage.
- Today: how it works, why it matters, what we've built — and what we have *not* built. We won't oversell.

---

### Slide 2: The Hook
**"Why do you remember the same event differently every time?"**

**Talking Points (45 seconds):**
- Context shapes memory. Same event, different recalls.
- You're not retrieving a fixed memory. You're **reconstructing** it, with current context as the lens.
- That's a feature, not a bug. Adaptive memory adapts.
- Wheeler Memory makes that explicit: same stored attractor, different reconstructions depending on the query.

---

### Slide 3: The Problem with Today's AI
**Talking Points (1 minute):**
- ChatGPT can't remember you between conversations.
- It also never forgets, and never says "I'm not sure" — it confabulates with full confidence.
- Vector databases are filing cabinets: query in, same row out.
- We want AI memory that admits uncertainty when uncertainty is warranted, and reconstructs differently in different contexts.

---

### Slide 4: The Substrate — Cellular Automata
**Talking Points (1.5 minutes):**
- A 64×64 grid. Each cell holds a continuous value in [−1, +1].
- Three rules per cell:
  - Local peak (surrounded by lower values) → push toward +1.
  - Local valley (surrounded by higher values) → push toward −1.
  - Slope → flow uphill toward the peak neighbor.
- Run those rules with current parameters, and chaos settles in **5–14 ticks** (~3 ms on CPU).
- The grid converges to a stable pattern — an **attractor**. *That* is the memory.
- Inspired by real neuroscience: brains store memories as patterns of activation, not as text files.

---

### Slide 5: Encoders — Native by Default
**Talking Points (1 minute):**
- We do not depend on pretrained models. The core is pure Python + numpy.
- Native encoders:
  - **Hippocampus**: character n-gram random indexing — captures lexical similarity.
  - **Context-RI**: distributional semantics, trained on WikiText-103 + OpenWebText (601M words). **First native encoder with positive SimLex-999 signal**: ρ = +0.255, vs MiniLM's pretrained ceiling of +0.446.
  - **Blended** (default): hippocampus(0.7) + language wheeler(0.3).
- `embedding` (MiniLM via sentence-transformers) is *optional*, available behind `pip install -e ".[embed]"`.

---

### Slide 6: Storing
**Talking Points (1 minute):**
- Take any text. Encode → 64×64 frame in [−1, +1]. Evolve through the CA.
- 5–14 ticks later, you have a stable attractor. We save the .npy file plus its metadata.
- Auto-routes to a domain chunk: `code`, `science`, `hardware`, `daily_tasks`, `meta`, `general`.
- On a recent AMD GPU (RX 9070 XT), batched evolution is **71× faster** at batch=1000. Auto-falls back to CPU.

---

### Slide 7: Recall, Two Tiers (the v0.3.6 headline)
**Talking Points (2 minutes):**
- Most "recall" requests don't need the whole CA loop. They just need to know **which basin** the query lands in.
- v0.3.6 splits recall in two:
  - **Recognition** — single-pass Pearson scan over stored attractors using the *raw query frame* (no CA convergence on the query). Returns a `BasinSeed` (id, similarity, basin stability) or `None`.
  - **Reconstruction** — warm-start CA from the stored attractor, blended with the query (α=0.3 by default). Re-evolve. Same memory, current context.
- Result on the warm-vs-cold benchmark: **~2× fewer ticks** with the warm path across near/mid/far input distance bands. Recognition rate is band-dependent (high on near-identical, low on substantial paraphrases — and that's correct, not a regression).

---

### Slide 8: Three-Grid Interference (default recall path)
**Talking Points (1.5 minutes):**
- Single-grid Pearson search misses something: epistemic state.
- v0.3.1 introduced three coupled 64×64 grids:
  - **Corpus**: crystallized knowledge, tight attractors (push=0.57). Barely decays.
  - **Experiential**: episodic memory, loose attractors (push=0.35). 2-day half-life.
  - **SCM** (Structural Coherence Map): permission topology. Sculpted by self-consistency feedback. *Permission, not content.*
- Score: `Answer(i,j) = Corpus(i,j) × Experiential(i,j) × (1 − |SCM(i,j)|)`.
- Four interference states fall out:
  - **GROUNDED**: corpus peak + experiential peak + SCM open
  - **ABSORBED**: corpus peak + no experiential + SCM open
  - **UNCONSOLIDATED**: no corpus + experiential peak + SCM open
  - **CONTESTED**: corpus peak + experiential peak + SCM closed
- v0.3.4 added per-event SCM telemetry to `scm_telemetry.jsonl`. v0.3.5 adds a closed-loop A/B eval.

---

### Slide 9: Temperature & Forgetting
**Talking Points (1 minute):**
- Memories have a temperature, computed from access frequency × recency (7-day half-life by default).
- Tiers: hot (≥0.6), warm (≥0.3), cold (≥0.05), fading (≥0.01), dead (<0.01).
- Temperature drives recall priority and decoder hedging.
- Sleep consolidation prunes redundant keyframes from cold bricks; dead memories evict (capacity = 10,000 attractors).

---

### Slide 10: Per-Basin Plasticity (the other v0.3.6 headline)
**Talking Points (1 minute):**
- Each stored basin carries a **Temporal Stability** `T` ∈ [0, 1].
- Fresh basins start at T=0 (fully plastic). Each `--learn`-enabled recall accumulates stability via EMA: `T_new = (1−rate)·T_old + rate·observed`.
- Drift rate is gated by T: `new = stored + (1−T)·base_rate·(observed − stored)`.
- Mature basins (T→1) become rigid; new basins absorb the input rapidly.
- `wheeler-recall "..." --recognize --learn` exercises the full path.

---

### Slide 11: Cortex — Native Semantic Scoring
**Talking Points (1 minute):**
- Three-tier scoring, all native:
  - **L1**: graph topology over retrieved attractors (Pearson adjacency, BFS clusters).
  - **L2**: settlement CA — opinion diffusion on the correlation graph until convergence.
  - **L3**: numpy SGD classifier (~11K params).
- Eliminates pretrained-model dependencies for semantic scoring.

---

### Slide 12: Honest MMLU Numbers
**Talking Points (1 minute):**
- 14,042 questions, 57 subjects, test split. Random chance is 25%.
- Zero-shot cortex (0 stored memories): **24.3%** — at chance, no knowledge to retrieve.
- Cortex + 1,812 learned facts: **25.3%** (+1.0%).
- Cortex + L3 classifier: **25.9%** (+1.6%).
- The L3 classifier loss barely moved from chance — needs more data or richer features. We say so out loud. The previous external-MiniLM baseline of 27.5% has been **removed** because it depended on a pretrained model.
- The next move is reconstruction scoring: evolve the query, settle the CA, read what the attractor says, compare to the choices.

---

### Slide 13: Privacy & Ownership
**Talking Points (45 seconds):**
- All local. No cloud APIs. No third-party servers. No subscription.
- Memories live on your disk. Optional sentence-transformers stays optional.
- You own the substrate.

---

### Slide 14: Real Numbers
**Talking Points (1 minute):**
- 44 modules, **775 tests**, 16 CLI commands.
- ~3 ms/tick CPU, 71× speedup on RX 9070 XT at batch=1000.
- Pure Python; numpy/scipy/matplotlib/psutil only in the core.
- CC BY-NC 4.0. Public on GitHub.

---

### Slide 15: What We Have *Not* Built
**Talking Points (1 minute):**
- No multimodal (no images, no audio).
- No federated memory.
- No browser runtime.
- No web dashboard right now (the prior `wheeler-ui` was retired in v0.3.6 because it had drifted out of date with the core).
- We mention this on purpose. The pitch is honest, not heroic.

---

### Slide 16: Why It Matters
**Talking Points (1 minute):**
- AI should remember the way you do — imperfectly, associatively, context-dependent.
- Vector DBs retrieve. Wheeler Memory reconstructs.
- Epistemic states are *emergent* from interference, not hand-coded. That's "It from Bit" applied to epistemology.
- **Convergence is ground truth.**

---

### Slide 17: Closing
**Talking Points (45 seconds):**
- This is real code, real tests, real numbers — pitched without inflation.
- We'd rather say "L3 was at chance" than fake the win.
- Wheeler Memory is open-source, local, native. The substrate is yours.

---

## Part 2: Live Walkthrough Script

### Prerequisites

```bash
cd /path/to/wheeler-memory
source .venv/bin/activate
pip install -e .                 # core only
# pip install -e ".[embed]"      # if you want MiniLM available
```

- Default data dir: `~/.wheeler_memory/`
- Static demo (browser-only): `docs/demos/demo.html` (open as a local file).
- Ollama (optional, for the agent): `ollama serve` and `ollama pull qwen2.5:1.5b`.

**Total demo time: ~8–10 minutes.**

> Note: there is no `wheeler-ui` web server in v0.3.6. The prior dashboard was retired because the script had drifted out of date with the core. If you want a visual, open `docs/demos/demo.html` as a static page.

---

### Section 1: Static Demo Page (1–2 minutes)

**Goal**: Give the audience something visual before any commands.

1. Open `docs/demos/demo.html` in a browser.
2. Walk through the embedded animations: chaos → convergence, three CA rules, attractor formation.
3. Talking point: "This is the substrate. Everything you see next is built on this."

**Timing: 1–2 minutes.**

---

### Section 2: Core CLI — Store, Recall, Two-Tier (3–4 minutes)

**Goal**: Show that the basic loop works and the v0.3.6 two-tier path is opt-in.

#### 2.1 Store a memory

```bash
wheeler-store "Self-attention computes weighted relationships between every pair of positions in a sequence."
```

Talking point: "We encoded that text into a 64×64 frame, ran ~5–14 ticks of CA evolution, and saved the resulting attractor. The default encoder is `blended` — hippocampus n-gram + language wheeler — no pretrained models."

#### 2.2 Recall (default three-grid interference path)

```bash
wheeler-recall "how does attention work in transformers"
```

Talking point: "That used the default recall path: Pearson pre-filter, then three-grid interference re-scoring (corpus × experiential × SCM-openness). With no experiential memories yet, it degrades cleanly to pure Pearson."

#### 2.3 Two-tier recognition (v0.3.6, opt-in)

```bash
wheeler-recall "how does attention work in transformers" --recognize
```

Talking point: "Recognition only. No CA loop on the query — single Pearson scan against stored attractors. Returns a `BasinSeed`. Cheap. We use this when we just need identity, not content."

#### 2.4 Recognition + per-basin learning

```bash
wheeler-recall "how does attention work in transformers" --recognize --learn
```

Talking point: "Same as above, but now the basin's per-basin Temporal Stability `T` accumulates via EMA, and the stored attractor drifts toward the observed pattern at rate `(1−T)·0.02`. Mature basins become rigid; fresh ones absorb new context. T persists in `index.json`."

#### 2.5 Inspect temperatures

```bash
wheeler-temps --top-n 5
```

Talking point: "Temperature is decay × access count. Hot at the top, dead at the bottom. Sleep consolidation prunes from below."

**Timing: 3–4 minutes.**

---

### Section 3: Crystallization & MMLU (2–3 minutes)

**Goal**: Show that the system can ingest a corpus and benchmark on real questions.

#### 3.1 Crystallize a small corpus

```bash
echo '{"text":"The mitochondrion is the powerhouse of the cell."}' > /tmp/demo.jsonl
echo '{"text":"Photosynthesis converts CO2 and water into glucose using light."}' >> /tmp/demo.jsonl
wheeler-crystallize /tmp/demo.jsonl --verbose
```

Talking point: "JSONL in, attractors out. Resume-safe — re-running skips already-stored entries."

#### 3.2 MMLU sanity check (one subject, fast)

```bash
wheeler-mmlu --subjects high_school_biology --mode cortex --samples 20
```

Talking point: "Three modes worth knowing: `--mode semantic` (default, pure Pearson), `--mode cortex` (L1/L2/L3 stack), `--mode learn` (full learn → consolidate → test cycle)."

**Timing: 2–3 minutes.**

---

### Section 4: Wheeler-Primary Agent (1–2 minutes, optional)

**Goal**: Show Wheeler driving a small LLM as a pure renderer.

Requires Ollama running with a small model:

```bash
wheeler-primary --interactive --show-state --model qwen2.5:1.5b
```

Talking point: "The CA decides what the answer should look like. The small model just renders that state into text. Swap the model — Wheeler stays the same. The CA is the mind; the LLM is the voice."

**Timing: 1–2 minutes.**

---

### Section 5: Under the Hood (1–2 minutes)

#### 5.1 Show the CA evolution GIF

Open `docs/assets/diagrams/evolution.gif`.

Talking point: "Frame-by-frame convergence. Chaos to order in tens of ticks."

#### 5.2 Show diversity / paraphrase reports

Open `docs/assets/reports/diversity_report.png`, `paraphrase_report.png`.

Talking point: "20 diverse inputs produce 20 distinct attractors with low cross-correlation. Paraphrases of the same text cluster together. The geometry is real."

**Timing: 1–2 minutes.**

---

## Troubleshooting & Contingencies

### `wheeler-*` not found
Activate the venv first: `source .venv/bin/activate`. Then `pip install -e .`.

### Ollama not responding (only matters for `wheeler-primary` / `wheeler-agent`)
Run `ollama serve` in another terminal. Verify with `ollama list`. If you don't have a model, pull one: `ollama pull qwen2.5:1.5b`.

### MMLU dataset not loaded
The MMLU loader pulls from Hugging Face on first run; expect a one-time download. Or pre-stage the dataset to `~/.cache/huggingface/`.

### GPU not detected
`wheeler-info` prints what was detected. If GPU is missing, the system auto-falls back to CPU (~3 ms/tick is fine for live demo).

### Image assets missing
`docs/assets/diagrams/evolution.gif` is generated by `python scripts/tools/generate_evolution_gif.py`. Diversity / paraphrase reports come from `notes/exploration/test_diversity*.py` and `test_paraphrase*.py` — run with `--output docs/assets/reports/...`.

### "I have no memories"
You haven't stored anything yet. Run a `wheeler-store "..."` first, or crystallize a small JSONL.

---

## Demo Script Summary

| Section | Duration | Key Points |
|---|---|---|
| Static demo page | 1–2 min | Visual substrate, no commands |
| Core CLI | 3–4 min | store, recall, two-tier `--recognize` / `--learn` |
| Crystallize + MMLU | 2–3 min | Corpus ingest + benchmark sanity |
| Wheeler-primary (optional) | 1–2 min | CA as mind, LLM as voice |
| Under the hood | 1–2 min | GIF + diversity / paraphrase reports |
| **Total** | **~8–12 min** | Live, honest, reproducible |

---

## Notes for Presenter

1. **Pacing**: Move quickly through each section. Park follow-up questions for after.
2. **Confidence without hype**: This is real code. Lean on that. We are not pitching "the next AGI" — we are pitching a reconstructive memory substrate that admits what it does and does not know.
3. **The reconstruction insight**: "Same memory, different reconstructions" is the point. If the audience walks away with that one idea, the demo worked.
4. **Privacy**: Reinforce that nothing leaves the machine. No cloud, no logs, no third parties.
5. **Two-tier is the v0.3.6 headline**. Don't bury it in the cold path slides — show `--recognize` and `--learn` directly. People understand "cheap probe vs expensive reconstruction" instantly.
6. **Honesty about MMLU**: If asked, lead with "L3 is at chance, here's why, here's the next move." Investors and engineers both reward that more than they reward inflated numbers.
7. **vs vector DBs**: "They retrieve, we reconstruct. Retrieval is fast and static. Reconstruction is slower and adaptive — like human memory."

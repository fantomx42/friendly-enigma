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
## A memory system that remembers like you do

Darman: cellular automata meet associative recall.
Open-source. Local. Native. v0.3.6.

---

## Why do you remember the same event differently every time?

Context shapes memory. Same event, reconstructed through different lenses.

Same memory, different recalls.

That's human memory. That's what we built.

---

## The problem with today's AI

Current AI systems:
- Never forget
- Never hedge
- Never say "I'm not sure" and mean it

Vector databases retrieve the same row every time. Same query → same answer, regardless of context.

Humans don't work that way.

---

## What if AI could actually remember?

- Forget gracefully? ✓ (temperature decay)
- Hedge on uncertain memories? ✓ (epistemic state from interference)
- Reconstruct context-dependently? ✓ (blend + re-evolve)
- Tell identity apart from content? ✓ (two-tier recall, v0.3.6)

That's Wheeler Memory.

---

## Cellular automata: the substrate

64×64 grid. Each cell holds a value in [−1, +1]. Three rules per cell:

- **Peak** (local max) → push toward +1
- **Valley** (local min) → push toward −1
- **Slope** → flow uphill

Run the rules. Chaos settles into a stable pattern in 5–14 ticks (~3 ms CPU).

That stable pattern *is* the memory.

---

## Storing a memory

1. Encode text → 64×64 frame (we have multiple native encoders, no pretrained models needed)
2. Run CA evolution
3. Save the resulting attractor (.npy on disk + metadata in `index.json`)

On a recent AMD GPU, batched evolution is **71× faster** at batch=1000. CPU works fine for live use.

---

## Recall, two tiers (v0.3.6 headline)

Identity is cheap. Content is expensive. v0.3.6 separates them.

- **Recognition**: single Pearson scan over stored attractors. Returns a `BasinSeed` or `None`. No CA loop on the query.
- **Reconstruction**: warm-start the CA from the stored attractor, blended with the query. Re-evolve. Same memory, current context.

`wheeler-recall "..." --recognize` → identity only.
`wheeler-recall "..."` → default three-grid interference (full content).

---

## The magic: reconstruction

Same stored attractor. Different queries. Different reconstructed outputs.

`new = stored × (1−α) + query × α`, then re-evolve.

Same memory. Different recalls. Like your brain.

---

## Three-grid interference (default recall)

```
Answer = Corpus × Experiential × (1 − |SCM|)
```

- **Corpus** — crystallized knowledge, tight attractors
- **Experiential** — episodic memory, loose attractors, 2-day half-life
- **SCM** — trust topology (permission, not content), sculpted by self-consistency

Four interference states fall out: GROUNDED, ABSORBED, UNCONSOLIDATED, CONTESTED.

---

## Temperature: forgetting built in

Every memory has a temperature.

- **Hot** (≥0.6) — recent or frequently recalled
- **Warm** (≥0.3) — middle ground
- **Cold** (≥0.05) — rarely accessed
- **Fading** / **Dead** — eviction candidates

Half-life: 7 days without recall. Decay is honest.

---

## Per-basin plasticity

Each basin carries a Temporal Stability `T` ∈ [0, 1].

- Fresh basins start at T=0 (fully plastic).
- `--learn` recalls accumulate T via EMA.
- Drift rate is gated: `(1 − T) × base_rate`.
- Mature basins become rigid; fresh ones absorb new context.

The brain analogue: rehearsal hardens the engram.

---

## Why this matters

Vector DBs retrieve. Wheeler Memory reconstructs.

- Retrieval is fast and **static**.
- Reconstruction is slower and **adaptive**.

Epistemic states emerge from interference rather than getting hand-coded. That's "It from Bit" applied to memory.

---

## What's native (no pretrained models)

- **Hippocampus encoder**: character n-gram random indexing
- **Context-RI encoder**: distributional semantics, trained on 601M words from WikiText-103 + OpenWebText. **First native encoder with positive SimLex-999 signal**: ρ = +0.255
- **Cortex L1/L2/L3**: graph topology + settlement CA + numpy SGD classifier (~11K params)

`sentence-transformers` is *optional*, behind `pip install -e ".[embed]"`.

---

## Honest numbers

- **44 modules, 775 tests, 16 CLI commands**
- ~3 ms/tick CPU, 71× speedup on RX 9070 XT
- MMLU all 57 subjects (14,042 questions, test split):
  - zero-shot cortex: 24.3%
  - cortex + learned facts: 25.3%
  - cortex + L3 classifier: 25.9%
  - random chance: 25.0%
- L3 classifier loss barely moved from chance. We say so.

---

## What we have NOT built

- No multimodal (no images, no audio)
- No federated memory
- No browser runtime
- No web dashboard right now (the prior `wheeler-ui` was retired in v0.3.6)

We list this on purpose. The pitch is honest, not heroic.

---

## Engine vs voice

**Engine**: the cellular automaton — the mind.
**Voice**: an optional small LLM (via Ollama) — the speaker.

These are *separate*. Swap the LLM and the memory stays the same. The CA decides what the answer should look like; the LLM only renders it.

`wheeler-primary --interactive --show-state` shows both.

---

## Privacy & ownership

All local. No cloud APIs. No third-party servers.

Memories stay on your disk. The optional embedding model stays optional.

You own the substrate.

---

## What's running today

✓ Native encoders (hippocampus, blended, context-RI)
✓ Three-grid interference recall (default path)
✓ Two-tier `--recognize` / `--learn` recall
✓ Sleep consolidation, eviction, temperature decay
✓ Cortex L1/L2/L3 native scoring
✓ HIP/ROCm GPU acceleration with CPU fallback
✓ MMLU benchmark runner (57 subjects, multiple modes)

Everything local. Nothing leaves your machine.

---

## Live demo

Walk through:
- Static demo page (`docs/demos/demo.html`) for the visual substrate
- `wheeler-store` / `wheeler-recall` / `--recognize` / `--learn`
- A small crystallization + MMLU sanity run
- Optional: `wheeler-primary` driving an Ollama model

See `pitch_pack/demo_script/demo_script.md`.

---

## What's next

- **Reconstruction scoring for MMLU**: evolve the query, settle the CA, read the attractor's answer, compare to choices. The "It from Bit" path to scoring.
- **Sleep consolidation of T**: an offline pass that consolidates accumulated Temporal Stability into the stored corpus state.
- **Spatial-product SCM scoring**: fix the known scalar-collapse issue in the interference engine.

The foundation is real. Each next step is named, scoped, and tracked.

---

## Open source

**GitHub**: [fantomx42/wheeler-memory](https://github.com/fantomx42/wheeler-memory)

**License**: CC BY-NC 4.0 (Non-Commercial Creative Commons)

**Citation**: John A. Wheeler, *Information, Physics, Quantum: The Search for Links* (1989). Elizabeth Loftus on reconstructive memory.

Contributions welcome.

---

## Why this excites us

Pure-Python cellular-automaton memory with two-tier recall, three-grid interference, and a native distributional encoder that hits 57% of MiniLM's pretrained ceiling — without ever touching a pretrained model in the core.

We're not pitching the next AGI. We're pitching a reconstructive memory substrate that admits what it does and does not know.

---

## Convergence is ground truth

*Everything else is commentary.*

Wheeler Memory isn't a search engine. Isn't a retrieval system.

It's a **reconstruction engine**. A mind. An automaton. An optional voice.

Darman remembers.

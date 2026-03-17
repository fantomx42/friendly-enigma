# Plan: Reconstruction Scoring — It From Bit

**Date:** 2026-03-18
**Status:** Ready to implement

---

## The Problem

`--mode learn` confirmed the loop works mechanically but yields 0% improvement. Pearson
correlation between CA attractors measures *shape similarity*, not *propositional content*.
The facts are in the index. The scoring mechanism can't read them.

Flat line: **27.5%** before and after learning 69 physics Q&A facts.

---

## The Idea

Wheeler stores `"Q: {question} A: {letter}. {choice_text}"` as attractors. If the recall
system works, querying with a test question should surface the correct Q&A fact near the
top. We just need to *read the letter out of the recalled text* instead of correlating
attractor shapes.

> evolve the query → let the CA settle → read back what the attractor is saying →
> compare that to the choices as text

This is "it from bit" as the scoring mechanism: the answer (it) emerges from the stored
information (bit) via recall + text matching, not from shape correlation.

---

## Two Scoring Modes to Implement

### Mode A: `--mode recall-text`  ← try this first

**Mechanism:**
1. Query Wheeler with just the question (no choice appended).
2. Get top-K recalled memories (already contains stored `"Q: ... A: X. ..."` facts).
3. Extract predicted letter from recalled text — look for `A: [A-D]` pattern.
4. Tally votes across top-K results; pick plurality letter.
5. Fall back to random if no letter found.

**Why this might work:**
The stored facts are verbatim `"Q: {question} A: {letter}. {text}"`. A strong recall hit
on the correct fact will contain the answer letter directly. No LLM required.

**Implementation — `score_recall_text()`:**
```python
import re

def score_recall_text(
    question: str,
    recall_k: int = 10,
    cache: AttractorCache | None = None,
    data_dir: Path | None = None,
) -> tuple[int, float]:
    """Recall top-K facts for the question, extract answer letter by vote."""
    from wheeler_memory.hashing import hash_to_frame
    from wheeler_memory.embedding import embed_to_frame

    # Use embedding if available for better recall
    try:
        from wheeler_memory.embedding import embed_to_frame
        frame = embed_to_frame(question)
    except Exception:
        from wheeler_memory.hashing import hash_to_frame
        frame = hash_to_frame(question)

    if cache is not None:
        hits = cache.search(frame, top_k=recall_k)
    else:
        return -1, 0.0

    votes = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    for hit in hits:
        text = hit.get("text", "")
        sim = hit.get("similarity", 0.0)
        m = re.search(r"A:\s*([A-D])\.", text)
        if m:
            votes[m.group(1)] += sim

    best = max(votes, key=votes.get)
    if votes[best] == 0.0:
        return -1, 0.0
    idx = CHOICES.index(best)
    return idx, votes[best]
```

Add `'recall-text'` to `--mode` choices. In `run_benchmark()`, build the attractor cache
as normal (semantic mode path), then route to `score_recall_text` instead of
`score_semantic`.

---

### Mode B: `--mode reconstruct`  ← if A is flat

**Mechanism:**
Use `recall_memory(..., reconstruct=True)` — this already exists in `storage.py`. It
returns `correlation_with_stored` and `correlation_with_query` per hit. Use
`correlation_with_stored` as the score rather than raw Pearson.

The hypothesis: reconstruction correlation is sensitive to *content match*, not just
*attractor shape*, because it measures how well the stored attractor reconstructs back
toward the query.

**Implementation:**
Add `score_reconstruct()` that calls `recall_memory` with `reconstruct=True` for each
`question + choice` and returns the best `correlation_with_stored` as score.

---

## Sequence for Tomorrow

1. **Implement `score_recall_text()`** and `--mode recall-text`.
2. **Run the physics baseline:**
   ```bash
   python scripts/wheeler_mmlu.py \
       --subjects high_school_physics conceptual_physics college_physics \
       --mode recall-text
   ```
3. **Compare to 27.5%.**
   - If improved → the recall loop is working. Tune K, try all 57 subjects.
   - If flat → the recall isn't surfacing the stored facts. Debug: print top-K hits for
     a few questions to see what's actually being recalled.
4. **If flat, implement `--mode reconstruct`** using the existing `reconstruct=True` path.

---

## Key Files

| File | What changes |
|------|-------------|
| `scripts/wheeler_mmlu.py` | Add `score_recall_text()`, `--mode recall-text` branch in `run_benchmark()` |
| `wheeler_memory/storage.py` | Already has `reconstruct=True`, `query_frame`, `readonly` — no changes needed |

---

## Debug Helper (if recall-text is flat)

Add a `--debug-recall` flag that prints the top-5 recalled texts for the first 3 questions
of each subject. This tells you immediately whether the stored `"Q: ... A: X."` facts are
appearing in recall at all.

---

## What This Proves

| Result | Meaning |
|--------|---------|
| recall-text > 27.5% | Recall is finding stored facts; letter extraction works |
| recall-text = 27.5% | Recall isn't surfacing stored facts near the top |
| reconstruct > 27.5% | Reconstruction correlation is content-sensitive |
| all flat | The CA attractor dynamics don't encode question-answer relationships at all; architecture question |

The baseline is **27.5%** (random chance for 4-choice is 25%).

"""Corpus builders for the hardened CA recall ablation (bench_ablation.py).

Three deterministic, seeded builders that produce the fact-dict shape the
ablation already consumes::

    {"text": "Q: <q> A: <L>. <choice>", "question": "<q>", "letter": "<L>"}

They exist to push the ablation out of the 100%-accuracy ceiling the original
69-fact physics corpus sat at:

  * ``load_mmlu_pool``        — SCALE. Thousands of real MMLU facts (uses the
                                large ``test`` split across all subjects), so the
                                capacity sweep can cross the Hopfield wall (N>565).
  * ``mine_hard_negatives``   — REALISTIC CROSSTALK. Groups of facts that are
                                mutual nearest-neighbours in the SAME representation
                                the RAW arm searches (un-evolved flattened frames),
                                so RAW is maximally confusable.
  * ``synthetic_minimal_pairs`` — CONTROLLED CROSSTALK. Templated near-duplicate
                                facts (identical surface text bar a few tokens),
                                an upper bound on interference, dataset-free.

Nothing here imports the CA: hard-negative mining deliberately uses the *raw*
embedding frames, because the point is to find what defeats the RAW arm and then
ask whether the CA cleans it up.
"""

from __future__ import annotations

import random

import numpy as np

_CHOICES = ("A", "B", "C", "D")
# Configs returned by cais/mmlu that are not per-subject QA sets.
_NON_SUBJECT_CONFIGS = {"all", "auxiliary_train"}


def _fact_from_item(item: dict) -> dict | None:
    """Build the standard fact dict from one MMLU row, or None if malformed."""
    try:
        ans = int(item["answer"])
        letter = _CHOICES[ans]
        choice = item["choices"][ans]
        q = item["question"]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return {"text": f"Q: {q} A: {letter}. {choice}", "question": q, "letter": letter}


def load_mmlu_pool(
    n: int,
    subjects: list[str] | None = None,
    splits: tuple[str, ...] = ("test", "validation", "dev"),
    seed: int = 42,
) -> list[dict]:
    """Return up to *n* unique MMLU facts, seeded-shuffled across many subjects.

    Unlike ``bench_associative._load_facts`` (dev+validation of 3 physics
    subjects, early-break → ~69 facts), this pulls the large ``test`` split
    across every subject config, so it scales into the thousands.
    """
    from datasets import get_dataset_config_names, load_dataset

    if subjects is None:
        try:
            subjects = [
                c for c in get_dataset_config_names("cais/mmlu")
                if c not in _NON_SUBJECT_CONFIGS
            ]
        except Exception:
            subjects = ["high_school_physics", "conceptual_physics", "college_physics"]

    rng = random.Random(seed)
    rng.shuffle(subjects)  # vary which subjects fill the quota across seeds

    facts: list[dict] = []
    seen: set[str] = set()
    for subject in subjects:
        for split in splits:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
            except Exception:
                continue
            for item in ds:
                fact = _fact_from_item(item)
                if fact is None or fact["text"] in seen:
                    continue
                seen.add(fact["text"])
                facts.append(fact)
                if len(facts) >= n:
                    rng.shuffle(facts)
                    return facts
    rng.shuffle(facts)
    return facts


def _centered_unit_matrix(frames: list[np.ndarray]) -> np.ndarray:
    """Flatten -> mean-center -> L2-normalise each frame (rows of the result).

    Mean-centering then cosine reproduces the centered-cosine / Pearson metric
    the RAW arm uses (``_search_raw``), so nearest neighbours here are exactly
    the items RAW will confuse. Degenerate (constant) frames become zero rows.
    """
    mat = np.stack([np.asarray(f).flatten().astype(np.float32) for f in frames])
    mat = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return mat / norms


def mine_hard_negatives(
    facts: list[dict],
    embed_fn,
    group_size: int = 8,
    n_groups: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Select facts that are mutual nearest-neighbours in RAW frame space.

    For each of *n_groups* seed facts (chosen at random, without reuse), take its
    ``group_size - 1`` nearest still-unused neighbours under centered-cosine on the
    un-evolved frames. The returned flat list therefore contains, for every member,
    several stored facts very close to it — the regime where RAW retrieval should
    start mis-ranking and CA basin cleanup could win.
    """
    if not facts:
        return []
    sim = _centered_unit_matrix(embed_fn([f["text"] for f in facts]))
    sim = sim @ sim.T
    np.fill_diagonal(sim, -np.inf)  # never pick self as its own neighbour

    rng = random.Random(seed)
    order = list(range(len(facts)))
    rng.shuffle(order)
    available = np.ones(len(facts), dtype=bool)

    picked: list[int] = []
    groups_made = 0
    for seed_idx in order:
        if groups_made >= n_groups or not available[seed_idx]:
            continue
        ranked = np.argsort(sim[seed_idx])[::-1]  # most-similar first
        group = [seed_idx]
        for cand in ranked:
            if len(group) >= group_size:
                break
            if available[cand]:
                group.append(int(cand))
        for idx in group:
            available[idx] = False
        picked.extend(group)
        groups_made += 1

    return [facts[i] for i in picked]


# Templates whose only varying tokens are the operands/answer, so stored frames
# are near-collinear — a deliberate upper bound on lexical crosstalk.
def _arith_add(a: int, b: int) -> dict:
    q = f"What is {a} plus {b}?"
    return {"text": f"Q: {q} A: {a + b}.", "question": q, "letter": None}


def _arith_mul(a: int, b: int) -> dict:
    q = f"What is {a} times {b}?"
    return {"text": f"Q: {q} A: {a * b}.", "question": q, "letter": None}


def synthetic_minimal_pairs(n: int, seed: int = 42) -> list[dict]:
    """Return *n* templated near-duplicate facts (distinct answers, ~identical text).

    Drawn from a large operand space across two templates, seeded-shuffled and
    deduped. Surface text differs only in a couple of digit tokens, so the RAW
    frames are highly collinear: the hardest crosstalk case in the suite.
    """
    rng = random.Random(seed)
    facts: list[dict] = []
    seen: set[str] = set()
    # Operand range sized so the pool comfortably exceeds any requested n.
    span = max(20, int(n ** 0.5) + 5)
    candidates: list[tuple] = []
    for a in range(2, 2 + span):
        for b in range(2, 2 + span):
            candidates.append(("add", a, b))
            candidates.append(("mul", a, b))
    rng.shuffle(candidates)

    for kind, a, b in candidates:
        fact = _arith_add(a, b) if kind == "add" else _arith_mul(a, b)
        if fact["text"] in seen:
            continue
        seen.add(fact["text"])
        facts.append(fact)
        if len(facts) >= n:
            break
    return facts

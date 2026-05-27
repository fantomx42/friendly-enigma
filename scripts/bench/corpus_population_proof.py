"""Corpus population proof (CANON open item #3, §8.2).

Validates the corpus-population strategy: does ingesting structured science
knowledge into the cold Corpus grid make that knowledge *retrievable*? Per
CANON §8.2, MMLU sits at the chance floor because the Corpus grid is empty of
world-knowledge structure — "MMLU will move when corpus does." This harness
measures the corpus-health side of that claim directly, offline.

Procedure
---------
1. Parse the local science Q&A corpora (datasets/arc.jsonl + sciq.jsonl) into
   (question, full-fact) pairs.
2. Baseline: against an empty Corpus grid, query a held set of questions and
   measure recall@1 / recall@5 (trivially ~0 — nothing is stored yet).
3. Populate: embed → evolve → (optionally ternary-snap) → store each fact into
   the science chunk of a fresh data dir.
4. After: re-query the same questions and re-measure recall@k. The delta is the
   proof that population creates a functioning retrieval structure.

Two population modes are run so the ternary-snap design choice can be justified
with data rather than asserted: stored attractors snapped to {-1,0,+1} vs left
as continuous float32.

Usage
-----
    python scripts/bench/corpus_population_proof.py
    python scripts/bench/corpus_population_proof.py --n-populate 1500 --n-query 300
    python scripts/bench/corpus_population_proof.py --json
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_batch, snap_to_ternary
from wheeler_memory.embedding import embed_to_frame_batch
from wheeler_memory.storage import store_memory

# datasets/ lives one level up from the repo root (shared corpus mount).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CORPUS_FILES = [
    _REPO_ROOT.parent / "datasets" / "arc.jsonl",
    _REPO_ROOT.parent / "datasets" / "sciq.jsonl",
]


def parse_corpus(paths=_CORPUS_FILES) -> list[tuple[str, str]]:
    """Parse science Q&A jsonl into (question, full_fact) pairs, deduped by question.

    Each line is {"text": "Q: <question> A: <answer...>"}. The full fact (Q+A)
    is what gets stored; the bare question is what we query with.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if " A: " not in text or not text.startswith("Q: "):
                    continue
                question = text.split(" A: ", 1)[0][len("Q: ") :].strip()
                if not question or question in seen:
                    continue
                seen.add(question)
                pairs.append((question, text))
    return pairs


def populate(facts: list[str], data_dir: Path, *, snap: bool) -> float:
    """Embed → evolve → (snap) → store each fact into the science chunk.

    Returns wall-clock seconds. Batches embedding and evolution for speed.
    """
    t0 = time.time()
    frames = embed_to_frame_batch(facts)
    results = evolve_batch(frames)
    for text, result in zip(facts, results):
        if snap:
            # Store the ternary-snapped attractor as the corpus coordinate.
            result = dict(result)
            result["attractor"] = snap_to_ternary(result["attractor"]).astype(np.float32)
        brick = MemoryBrick.from_evolution_result(result)
        store_memory(
            text, result, brick, data_dir=data_dir, chunk="science", auto_evict=False
        )
    return time.time() - t0


def measure_recall(
    queries: list[tuple[str, str]], data_dir: Path, *, k: int = 5
) -> dict:
    """Closed-set recall@1 / recall@k for a set of (question, target_fact) pairs.

    A query is recall@1 correct if the top hit's text equals the target fact,
    recall@k correct if the target appears anywhere in the top-k hits. Uses the
    vectorized AttractorCache (the same path MMLU semantic scoring uses).
    """
    from scripts.wheeler_mmlu import AttractorCache

    cache = AttractorCache(data_dir=data_dir)
    if not queries:
        return {"recall@1": 0.0, "recall@k": 0.0, "n": 0, "k": k}

    q_frames = embed_to_frame_batch([q for q, _ in queries])
    hit1 = 0
    hitk = 0
    for (_, target), frame in zip(queries, q_frames):
        hits = cache.search(frame, top_k=k)
        texts = [h["text"] for h in hits]
        if texts and texts[0] == target:
            hit1 += 1
        if target in texts:
            hitk += 1
    n = len(queries)
    return {
        "recall@1": round(hit1 / n, 4),
        "recall@k": round(hitk / n, 4),
        "n": n,
        "k": k,
    }


def run_mode(
    populate_facts: list[str],
    queries: list[tuple[str, str]],
    *,
    snap: bool,
    k: int,
    verbose: bool,
) -> dict:
    """Baseline (empty) → populate → after, for one storage mode."""
    mode = "ternary-snap" if snap else "float"
    with tempfile.TemporaryDirectory(prefix=f"wm_corpus_{mode}_") as tmp:
        data_dir = Path(tmp)

        if verbose:
            print(f"\n[{mode}] baseline recall on empty corpus...")
        baseline = measure_recall(queries, data_dir, k=k)

        if verbose:
            print(f"[{mode}] populating {len(populate_facts)} facts...")
        elapsed = populate(populate_facts, data_dir, snap=snap)

        if verbose:
            print(f"[{mode}] re-measuring recall after population...")
        after = measure_recall(queries, data_dir, k=k)

    return {
        "mode": mode,
        "n_populate": len(populate_facts),
        "populate_seconds": round(elapsed, 1),
        "baseline": baseline,
        "after": after,
    }


def run_mmlu_crosscheck(
    populate_facts: list[str],
    subjects: list[str],
    *,
    samples: int,
    verbose: bool,
) -> dict:
    """Embedding-consistent MMLU semantic cross-check: empty vs populated corpus.

    The stock wheeler-mmlu semantic path hash-encodes queries, which cannot
    consume an embedding-populated corpus. Here we encode "question + choice"
    with the same embedding encoder used for population and inject it into
    score_semantic via precomputed_frames, so the comparison is in one space.
    """
    from scripts.wheeler_mmlu import AttractorCache, load_mmlu, score_semantic

    items = list(load_mmlu(subjects, split="test", samples=samples))
    if not items:
        return {"subjects": subjects, "n": 0, "note": "no MMLU items loaded"}

    # Pre-embed every (question + choice) frame once.
    frame_cache: dict[str, np.ndarray] = {}
    flat_texts = []
    for _, q, choices, _ in items:
        for c in choices:
            flat_texts.append(f"{q} {c}")
    for text, frame in zip(flat_texts, embed_to_frame_batch(flat_texts)):
        frame_cache[text] = frame

    def _accuracy(cache: AttractorCache) -> float:
        correct = 0
        for _, q, choices, answer_idx in items:
            frames = [frame_cache[f"{q} {c}"] for c in choices]
            pred, _ = score_semantic(q, choices, cache=cache, precomputed_frames=frames)
            correct += int(pred == answer_idx)
        return correct / len(items)

    with tempfile.TemporaryDirectory(prefix="wm_mmlu_empty_") as empty_dir:
        if verbose:
            print(f"\n[mmlu] baseline (empty corpus) on {subjects}, {len(items)} items...")
        baseline_acc = _accuracy(AttractorCache(data_dir=Path(empty_dir)))

    with tempfile.TemporaryDirectory(prefix="wm_mmlu_pop_") as pop_dir:
        if verbose:
            print(f"[mmlu] populating {len(populate_facts)} facts (ternary-snap)...")
        populate(populate_facts, Path(pop_dir), snap=True)
        if verbose:
            print("[mmlu] re-scoring against populated corpus...")
        after_acc = _accuracy(AttractorCache(data_dir=Path(pop_dir)))

    return {
        "subjects": subjects,
        "n": len(items),
        "baseline_accuracy": round(baseline_acc, 4),
        "after_accuracy": round(after_acc, 4),
        "chance": 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus population proof (CANON §8.2)")
    parser.add_argument("--n-populate", type=int, default=1500)
    parser.add_argument("--n-query", type=int, default=300)
    parser.add_argument("--k", type=int, default=5, help="recall@k cutoff")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    parser.add_argument(
        "--mmlu-subjects",
        default="",
        help="Comma-separated MMLU subjects for the secondary cross-check (e.g. "
        "college_biology,high_school_biology). Empty skips the cross-check.",
    )
    parser.add_argument("--mmlu-samples", type=int, default=50)
    parser.add_argument(
        "--skip-recall", action="store_true", help="Run only the MMLU cross-check"
    )
    args = parser.parse_args()

    verbose = not args.json
    pairs = parse_corpus()
    if len(pairs) < args.n_populate:
        raise SystemExit(
            f"only {len(pairs)} unique facts parsed; need >= {args.n_populate}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    populate_pairs = pairs[: args.n_populate]
    # Closed-set: queries are a subset of what we populate.
    queries = rng.sample(populate_pairs, min(args.n_query, len(populate_pairs)))
    populate_facts = [fact for _, fact in populate_pairs]

    if verbose:
        print(
            f"Corpus population proof: {len(populate_facts)} facts, "
            f"{len(queries)} queries, recall@{args.k}"
        )

    results = (
        []
        if args.skip_recall
        else [
            run_mode(populate_facts, queries, snap=snap, k=args.k, verbose=verbose)
            for snap in (True, False)
        ]
    )

    mmlu = None
    if args.mmlu_subjects:
        subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
        mmlu = run_mmlu_crosscheck(
            populate_facts, subjects, samples=args.mmlu_samples, verbose=verbose
        )

    payload = {
        "n_populate": args.n_populate,
        "n_query": len(queries),
        "k": args.k,
        "seed": args.seed,
        "modes": results,
        "mmlu": mmlu,
    }

    if args.json:
        print(json.dumps(payload))
        return

    print(f"\n{'=' * 64}")
    print("  CORPUS POPULATION PROOF")
    print(f"{'=' * 64}")
    if results:
        print(f"  {'mode':<14} {'recall@1':>18} {'recall@' + str(args.k):>18}")
        print(f"  {'':<14} {'before → after':>18} {'before → after':>18}")
        print(f"  {'-' * 60}")
        for r in results:
            b, a = r["baseline"], r["after"]
            r1 = f"{b['recall@1']:.0%} → {a['recall@1']:.0%}"
            rk = f"{b['recall@k']:.0%} → {a['recall@k']:.0%}"
            print(f"  {r['mode']:<14} {r1:>18} {rk:>18}  ({r['populate_seconds']}s)")
    if mmlu and mmlu.get("n"):
        print(f"  {'-' * 60}")
        print(
            f"  MMLU cross-check {mmlu['subjects']} (n={mmlu['n']}, chance "
            f"{mmlu['chance']:.0%}): {mmlu['baseline_accuracy']:.0%} → "
            f"{mmlu['after_accuracy']:.0%}"
        )
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

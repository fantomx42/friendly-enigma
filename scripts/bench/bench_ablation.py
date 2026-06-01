"""Wheeler Memory — CA Ablation Benchmark (does the cellular automaton earn its keep?)

This is the experiment the project never ran: it isolates the *contribution of the
cellular-automaton evolution* to recall quality, holding everything else fixed.

Two arms, identical in every respect except one:

  CA arm  (the system as shipped):  embed -> evolve_and_interpret -> centered-cosine search
  RAW arm (the ablation):           embed ------------------------> centered-cosine search

Both arms use:
  - the SAME encoder (`hippocampus_to_frame_batch`, n-gram random indexing -> 64x64 frame)
  - the SAME retrieval metric (centered Pearson/cosine over the 4096-d flattened frame)
  - the SAME facts, queries, corruption seeds, and top-k

The ONLY difference is whether `evolve_and_interpret` (the CA) is applied to both the
stored items and the query before comparison. So the gap (CA - RAW) is a clean estimate
of what the CA buys you. If the gap is ~0 or negative, the CA is decorative for recall.

The CA arm reuses the exact helpers from `bench_associative.py` (no reimplementation, so
parity with the headline benchmark is guaranteed). The RAW arm is defined here.

Usage
-----
    python scripts/bench/bench_ablation.py
    python scripts/bench/bench_ablation.py --ns 5 10 22 50 --n-trials 3
    python scripts/bench/bench_ablation.py --skip-capacity
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Reuse the CA-arm helpers verbatim so the "with CA" path is identical to the
# shipped benchmark (bench_associative.py lives in this same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_associative import (  # noqa: E402
    _embed,
    _store,
    _build_cache,
    _search,  # CA arm: evolves the query, then centered-cosine
    _load_facts,
    _corrupt_text,
)

# ── RAW arm (the ablation): no CA, search the embedding frames directly ───────


def _build_cache_from_frames(texts: list[str], frames: list[np.ndarray]) -> dict:
    """Build a search cache straight from raw embedding frames (no evolve, no disk).

    Mirrors bench_associative._build_cache's output schema, but the vectors are the
    un-evolved 64x64 frames rather than CA attractors loaded from disk.
    """
    atts = [np.asarray(f).flatten().astype(np.float32) for f in frames]
    matrix = np.stack(atts)
    return {
        "matrix": matrix,
        "means": matrix.mean(axis=1),
        "stds": matrix.std(axis=1),
        "texts": list(texts),
    }


def _search_raw(cache: dict, frame: np.ndarray, top_k: int = 1) -> list[dict]:
    """Centered-cosine top-k over raw frames — identical math to _search, minus evolve."""
    q = np.asarray(frame).flatten().astype(np.float32)
    qm, qs = q.mean(), q.std()
    if qs < 1e-10:
        return []
    centered = cache["matrix"] - cache["means"][:, None]
    dots = centered @ (q - qm)
    valid = cache["stds"] > 1e-10
    sims = np.where(valid, dots / (4096 * cache["stds"] * qs), 0.0)
    k = min(top_k, len(sims))
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return [{"text": cache["texts"][i], "similarity": float(sims[i])} for i in idx]


# ── shared hit-counting: STRICT exact top-1 pattern retrieval ─────────────────
#
# The standard associative-memory criterion: a partial/corrupted cue must retrieve
# THE one stored pattern it came from. We score top-1 exact-text match rather than the
# shipped benchmark's "answer-letter in top-5" (which ceilings at 100% — only 4 letters,
# 5 slots — and so cannot discriminate the CA arm from the RAW arm).


def _count_hits(retrieve, query_facts, q_frames) -> float:
    """Fraction of queries whose #1 retrieved item is exactly the correct stored fact.

    `retrieve` is a callable (qframe, top_k) -> list[{"text": ...}]; CA and RAW arms
    share this loop so scoring can never diverge between them.
    """
    hits = 0
    for fact, qframe in zip(query_facts, q_frames):
        top = retrieve(qframe, 1)
        if top and top[0]["text"] == fact["text"]:
            hits += 1
    return hits / len(query_facts)


def _row(label, ca, raw, width=34):
    delta = ca - raw
    flag = "CA helps" if delta > 0.02 else ("CA hurts" if delta < -0.02 else "no effect")
    print(f"  {label:<{width}}  {ca:>8.1%}  {raw:>8.1%}  {delta:>+7.1%}   {flag}")


# ── Test 1: capacity curve ────────────────────────────────────────────────────


def test_capacity(facts, ns, n_trials):
    print("\n── Capacity: recall accuracy vs N stored (CA vs RAW) ───────────")
    print(f"  {'N (alpha=N/4096)':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    rng = random.Random(42)
    out = {}
    for n in ns:
        if n > len(facts):
            continue
        ca_trials, raw_trials = [], []
        for _ in range(n_trials):
            subset = rng.sample(facts, n)
            texts = [f["text"] for f in subset]
            frames = _embed(texts)
            # RAW cache: straight from frames, no disk
            raw_cache = _build_cache_from_frames(texts, frames)
            # CA cache: store (evolves + writes), then load attractors
            with tempfile.TemporaryDirectory() as tmp:
                data_dir = Path(tmp)
                for text, frame in zip(texts, frames):
                    _store(text, frame, data_dir)
                ca_cache = _build_cache(data_dir)
            if ca_cache is None:
                continue
            q_frames = _embed([f["question"] for f in subset])
            ca_trials.append(
                _count_hits(lambda qf, k: _search(ca_cache, qf, top_k=k), subset, q_frames)
            )
            raw_trials.append(
                _count_hits(lambda qf, k: _search_raw(raw_cache, qf, top_k=k), subset, q_frames)
            )
        if not ca_trials:
            continue
        ca, raw = float(np.mean(ca_trials)), float(np.mean(raw_trials))
        _row(f"N={n}  (alpha={n/4096:.3f})", ca, raw)
        out[n] = {"ca": ca, "raw": raw}
    return out


# ── Test 2: cue degradation ───────────────────────────────────────────────────


def test_degradation(facts, n_store, corruptions):
    print("\n── Cue degradation: accuracy vs % of query words dropped ───────")
    print(f"  {'corruption':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    subset = facts[:n_store]
    texts = [f["text"] for f in subset]
    frames = _embed(texts)
    raw_cache = _build_cache_from_frames(texts, frames)
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        for text, frame in zip(texts, frames):
            _store(text, frame, data_dir)
        ca_cache = _build_cache(data_dir)
    rng = random.Random(42)
    out = {}
    for c in corruptions:
        queries = [_corrupt_text(f["question"], c, rng) for f in subset]
        q_frames = _embed(queries)
        ca = _count_hits(lambda qf, k: _search(ca_cache, qf, top_k=k), subset, q_frames)
        raw = _count_hits(lambda qf, k: _search_raw(raw_cache, qf, top_k=k), subset, q_frames)
        _row(f"{c:.0%} words dropped", ca, raw)
        out[c] = {"ca": ca, "raw": raw}
    return out


# ── Test 3: interference ──────────────────────────────────────────────────────


def test_interference(n_store):
    print("\n── Interference: same-subject vs cross-subject (CA vs RAW) ─────")
    print(f"  {'condition':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    from datasets import load_dataset

    CHOICES = ["A", "B", "C", "D"]

    def load_subject(subject, n):
        out = []
        for split in ["dev", "validation"]:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
                for item in ds:
                    letter = CHOICES[int(item["answer"])]
                    out.append({
                        "text": f"Q: {item['question']} A: {letter}. {item['choices'][int(item['answer'])]}",
                        "question": item["question"],
                        "letter": letter,
                    })
                    if len(out) >= n:
                        return out
            except Exception:
                pass
        return out

    physics = load_subject("high_school_physics", n_store)
    history = load_subject("high_school_world_history", n_store)
    n = min(len(physics), len(history), n_store)
    physics, history = physics[:n], history[:n]

    out = {}
    for label, store_facts, query_facts in [
        ("same-subject (physics only)", physics, physics),
        ("cross-subject (physics+history)", physics + history, physics),
    ]:
        texts = [f["text"] for f in store_facts]
        frames = _embed(texts)
        raw_cache = _build_cache_from_frames(texts, frames)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for text, frame in zip(texts, frames):
                _store(text, frame, data_dir)
            ca_cache = _build_cache(data_dir)
        q_frames = _embed([f["question"] for f in query_facts])
        ca = _count_hits(lambda qf, k: _search(ca_cache, qf, top_k=k), query_facts, q_frames)
        raw = _count_hits(lambda qf, k: _search_raw(raw_cache, qf, top_k=k), query_facts, q_frames)
        _row(label, ca, raw)
        out[label] = {"ca": ca, "raw": raw}
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description="Wheeler CA ablation: evolve vs raw frames")
    p.add_argument("--ns", type=int, nargs="+", default=[5, 10, 22, 50, 100])
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--n-store", type=int, default=22, help="store size for degradation test")
    p.add_argument("--skip-capacity", action="store_true")
    p.add_argument("--skip-degradation", action="store_true")
    p.add_argument("--skip-interference", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    print("=" * 70)
    print("  WHEELER MEMORY — CA ABLATION  (CA = evolve-then-search,  RAW = embed-then-search)")
    print("  Positive delta => the cellular automaton improves recall. ~0 => decorative.")
    print("=" * 70)

    max_needed = max(max(args.ns, default=0), args.n_store, 60)
    print(f"\nLoading up to {max_needed} physics Q&A facts...")
    facts = _load_facts(max_needed)
    print(f"  Loaded {len(facts)} facts")
    if not facts:
        print("ERROR: no facts loaded (is 'datasets' installed?)")
        sys.exit(1)

    print("Warming up embedding model...")
    _embed([facts[0]["text"]])
    print("  Ready.")

    t0 = time.time()
    summary = []
    if not args.skip_capacity:
        cap = test_capacity(facts, ns=[n for n in args.ns if n <= len(facts)], n_trials=args.n_trials)
        if cap:
            d = np.mean([v["ca"] - v["raw"] for v in cap.values()])
            summary.append(("capacity", d))
    if not args.skip_degradation:
        deg = test_degradation(facts, n_store=min(args.n_store, len(facts)),
                               corruptions=[0.0, 0.2, 0.4, 0.6])
        if deg:
            d = np.mean([v["ca"] - v["raw"] for v in deg.values()])
            summary.append(("cue degradation", d))
    if not args.skip_interference:
        intf = test_interference(args.n_store)
        if intf:
            d = np.mean([v["ca"] - v["raw"] for v in intf.values()])
            summary.append(("interference", d))

    print(f"\n{'=' * 70}")
    print(f"  VERDICT  (mean CA - RAW per metric; positive = CA earns its keep)")
    print(f"{'-' * 70}")
    for name, d in summary:
        flag = "CA helps" if d > 0.02 else ("CA hurts" if d < -0.02 else "NO MEASURABLE BENEFIT")
        print(f"    {name:<22} {d:>+7.1%}   {flag}")
    overall = np.mean([d for _, d in summary]) if summary else 0.0
    print(f"{'-' * 70}")
    print(f"    {'OVERALL':<22} {overall:>+7.1%}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

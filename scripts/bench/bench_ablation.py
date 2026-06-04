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

The original 69-fact physics corpus ceilinged both arms at 100%, making the
+0.0% delta inconclusive. This hardened version scales the corpus (MMLU, all
subjects, ~thousands) and adds two adversarial regimes — hard-negative
crosstalk and synthetic minimal pairs — to push RAW into failure, where the
question "does attractor cleanup beat plain embedding cosine?" can finally be
answered. See scripts/bench/corpora.py for the corpus builders.

Usage
-----
    python scripts/bench/bench_ablation.py                     # full sweep N->2000, both adversarial axes
    python scripts/bench/bench_ablation.py --quick             # seconds: smoke-test wiring
    python scripts/bench/bench_ablation.py --max-n 565         # stop at the Hopfield wall
    python scripts/bench/bench_ablation.py --adversarial none  # capacity + legacy tests only
    python scripts/bench/bench_ablation.py --out results.json  # persist raw CA/RAW points
"""

from __future__ import annotations

import argparse
import json
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
    _corrupt_text,
)
from corpora import (  # noqa: E402
    load_mmlu_pool,
    mine_hard_negatives,
    synthetic_minimal_pairs,
)


def _trials_for(n: int, override: int | None) -> int:
    """Trial count for capacity at store-size *n* (1 CA evolve per stored item,
    so big N is expensive). An explicit --n-trials overrides the schedule."""
    if override is not None:
        return override
    if n <= 200:
        return 3
    if n <= 800:
        return 2
    return 1

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


def _ca_raw_accuracy(store_facts, query_facts):
    """Store *store_facts*, query each *query_fact* by its question, return
    (ca, raw) strict top-1 exact-text accuracy. Both arms share _count_hits so
    scoring cannot diverge. Returns (None, None) if the CA store is degenerate."""
    texts = [f["text"] for f in store_facts]
    frames = _embed(texts)
    raw_cache = _build_cache_from_frames(texts, frames)
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        for text, frame in zip(texts, frames):
            _store(text, frame, data_dir)
        ca_cache = _build_cache(data_dir)
    if ca_cache is None:
        return None, None
    q_frames = _embed([f["question"] for f in query_facts])
    ca = _count_hits(lambda qf, k: _search(ca_cache, qf, top_k=k), query_facts, q_frames)
    raw = _count_hits(lambda qf, k: _search_raw(raw_cache, qf, top_k=k), query_facts, q_frames)
    return ca, raw


def _raw_breaks(results: dict) -> tuple | None:
    """Smallest key at which RAW top-1 accuracy first drops below 90%.

    Returns (key, ca, raw, ca-raw) — the verdict the ceilinged run could not
    produce: what the CA buys *in the regime where embeddings start failing*."""
    for key in sorted(results):
        raw = results[key]["raw"]
        if raw < 0.90:
            ca = results[key]["ca"]
            return key, ca, raw, ca - raw
    return None


# ── Test 1: capacity curve ────────────────────────────────────────────────────


def test_capacity(facts, ns, n_trials=None):
    print("\n── Capacity: recall accuracy vs N stored (CA vs RAW) ───────────")
    print("  (Hopfield wall: alpha=0.138 ~ N=565 on a 4096-cell grid)")
    print(f"  {'N (alpha=N/4096)':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    rng = random.Random(42)
    out = {}
    for n in ns:
        if n > len(facts):
            continue
        ca_trials, raw_trials = [], []
        for _ in range(_trials_for(n, n_trials)):
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


# ── Test 4: hard-negative crosstalk (realistic adversarial) ───────────────────


def test_crosstalk(pool, group_sizes, n_groups):
    print("\n── Crosstalk: hard-negative groups (CA vs RAW) ─────────────────")
    print("  Each member has `group_size-1` near-twins stored alongside it.")
    print(f"  {'group_size (stored=gs*groups)':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    out = {}
    for gs in group_sizes:
        facts = mine_hard_negatives(pool, _embed, group_size=gs, n_groups=n_groups)
        if len(facts) < 2:
            continue
        ca, raw = _ca_raw_accuracy(facts, facts)
        if ca is None:
            continue
        _row(f"gs={gs}  (stored={len(facts)})", ca, raw)
        out[gs] = {"ca": ca, "raw": raw, "stored": len(facts)}
    return out


# ── Test 5: synthetic minimal pairs (controlled-extreme crosstalk) ────────────


def test_minimal_pairs(densities):
    print("\n── Minimal pairs: templated near-duplicates (CA vs RAW) ────────")
    print("  Surface text differs only in a few digit tokens -> collinear frames.")
    print(f"  {'N stored (minimal pairs)':<34}  {'CA':>8}  {'RAW':>8}  {'delta':>7}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*7}")
    out = {}
    for d in densities:
        facts = synthetic_minimal_pairs(d)
        if len(facts) < 2:
            continue
        ca, raw = _ca_raw_accuracy(facts, facts)
        if ca is None:
            continue
        _row(f"N={len(facts)}", ca, raw)
        out[len(facts)] = {"ca": ca, "raw": raw}
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description="Wheeler CA ablation: evolve vs raw frames")
    p.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 565, 800, 1200, 2000],
                   help="capacity store sizes (default crosses the Hopfield wall to N=2000)")
    p.add_argument("--max-n", type=int, default=2000, help="cap the capacity sweep at this N")
    p.add_argument("--n-trials", type=int, default=None,
                   help="override the per-N trial schedule (default: 3/2/1 by size)")
    p.add_argument("--n-store", type=int, default=22, help="store size for the legacy easy tests")
    p.add_argument("--adversarial", choices=["none", "hardneg", "minimal", "both"], default="both")
    p.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8, 16],
                   help="hard-negative group sizes (confusability density)")
    p.add_argument("--n-groups", type=int, default=20, help="number of hard-negative groups")
    p.add_argument("--minimal-densities", type=int, nargs="+", default=[50, 200, 500, 1000])
    p.add_argument("--quick", action="store_true", help="tiny sweep to smoke-test wiring")
    p.add_argument("--out", type=str, default=None, help="write raw results JSON to this path")
    p.add_argument("--skip-capacity", action="store_true")
    p.add_argument("--skip-degradation", action="store_true")
    p.add_argument("--skip-interference", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.quick:
        args.ns, args.max_n = [50, 100], 100
        args.group_sizes, args.n_groups = [4], 5
        args.minimal_densities, args.n_store = [50], 20

    print("=" * 70)
    print("  WHEELER MEMORY — CA ABLATION  (CA = evolve-then-search,  RAW = embed-then-search)")
    print("  Positive delta => the cellular automaton improves recall. ~0 => decorative.")
    print("=" * 70)

    ns = sorted(n for n in args.ns if n <= args.max_n)
    # Pool sized for the capacity sweep plus headroom for hard-negative mining.
    need_adv = args.adversarial in ("hardneg", "both")
    pool_n = max(ns, default=0)
    if need_adv:
        pool_n = max(pool_n, max(args.group_sizes) * args.n_groups * 8, 3000 if not args.quick else 300)
    print(f"\nLoading up to {pool_n} MMLU facts (test+validation+dev, all subjects)...")
    facts = load_mmlu_pool(pool_n)
    print(f"  Loaded {len(facts)} facts")
    if not facts:
        print("ERROR: no facts loaded (is 'datasets' installed?)")
        sys.exit(1)

    print("Warming up embedding model...")
    _embed([facts[0]["text"]])
    print("  Ready.")

    t0 = time.time()
    summary = []        # (name, mean delta) for the headline table
    breaks = []         # (name, where-RAW-breaks tuple) — the real verdict
    raw_results = {}    # for --out

    if not args.skip_capacity:
        cap = test_capacity(facts, ns=[n for n in ns if n <= len(facts)], n_trials=args.n_trials)
        if cap:
            summary.append(("capacity", np.mean([v["ca"] - v["raw"] for v in cap.values()])))
            breaks.append(("capacity (N)", _raw_breaks(cap)))
            raw_results["capacity"] = cap
    if not args.skip_degradation:
        deg = test_degradation(facts, n_store=min(args.n_store, len(facts)),
                               corruptions=[0.0, 0.2, 0.4, 0.6])
        if deg:
            summary.append(("cue degradation", np.mean([v["ca"] - v["raw"] for v in deg.values()])))
            raw_results["degradation"] = deg
    if not args.skip_interference:
        intf = test_interference(args.n_store)
        if intf:
            summary.append(("interference", np.mean([v["ca"] - v["raw"] for v in intf.values()])))
            raw_results["interference"] = intf
    if need_adv:
        ct = test_crosstalk(facts, group_sizes=args.group_sizes, n_groups=args.n_groups)
        if ct:
            summary.append(("crosstalk (hardneg)", np.mean([v["ca"] - v["raw"] for v in ct.values()])))
            breaks.append(("crosstalk (group_size)", _raw_breaks(ct)))
            raw_results["crosstalk"] = ct
    if args.adversarial in ("minimal", "both"):
        mp = test_minimal_pairs(args.minimal_densities)
        if mp:
            summary.append(("minimal pairs", np.mean([v["ca"] - v["raw"] for v in mp.values()])))
            breaks.append(("minimal pairs (N)", _raw_breaks(mp)))
            raw_results["minimal_pairs"] = mp

    print(f"\n{'=' * 70}")
    print("  VERDICT  (mean CA - RAW per metric; positive = CA earns its keep)")
    print(f"{'-' * 70}")
    for name, d in summary:
        flag = "CA helps" if d > 0.02 else ("CA hurts" if d < -0.02 else "NO MEASURABLE BENEFIT")
        print(f"    {name:<24} {d:>+7.1%}   {flag}")
    overall = np.mean([d for _, d in summary]) if summary else 0.0
    print(f"{'-' * 70}")
    print(f"    {'OVERALL':<24} {overall:>+7.1%}")

    print(f"\n  WHERE RAW BREAKS  (first point RAW top-1 < 90%; CA-RAW delta there)")
    print(f"{'-' * 70}")
    any_break = False
    for name, br in breaks:
        if br is None:
            print(f"    {name:<24} RAW never dropped below 90% (regime still too easy)")
            continue
        any_break = True
        key, ca, raw, d = br
        verdict = "CA RESCUES" if d > 0.05 else ("CA also fails" if d <= 0.02 else "CA helps a little")
        print(f"    {name:<24} at {key}: CA={ca:.1%} RAW={raw:.1%} delta={d:>+.1%}  [{verdict}]")
    if not any_break:
        print("    Nothing broke RAW — escalate --ns / --group-sizes / --minimal-densities.")
    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")
    print("=" * 70)

    if args.out:
        Path(args.out).write_text(json.dumps(raw_results, indent=2))
        print(f"  Raw results written to {args.out}")


if __name__ == "__main__":
    main()

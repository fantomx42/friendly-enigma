#!/usr/bin/env python3
"""Realized-topology sweep — find a strong capture to grade direction on.

The CA's surviving decode contribution lives in the *pairwise* attractor
correlations (leave-one-out at the decode layer; see tests/topology_isolation.py
--field-isolation).  Pairwise structure is strongest on dense, well-clustered
recalls — exactly the captures a directional judge should be aimed at, NOT the
thin ISOLATED captures where the signal is weakest.

This sweep runs many real recall queries through the SAME path as run_ablation
(recall_with_interference -> extract_state), bins each capture by its REALIZED
landscape (TIGHT/SPREAD/ISOLATED/EMPTY, computed from the actual recalled
attractors — never from the intended query), and selects the highest-mean|r|
capture among the high-confidence ones.  Each capture is persisted in full so it
replays deterministically into the grader.

Usage:
    python scripts/bench/topology_sweep.py
    python scripts/bench/topology_sweep.py --data-dir ~/.wheeler_memory
    python scripts/bench/topology_sweep.py --corpus my_queries.jsonl   # {"query": ...} per line
    python scripts/bench/topology_sweep.py --recall-k 6 --min-confidence 0.40

After selecting a capture, grade its direction with:
    python tests/topology_isolation.py --ablation --field-isolation --grade \
        --judge-backend claude --query "<selected query>" [--data-dir ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable when run as a file: sys.path[0] is this script's
# directory (scripts/bench), not the repo root. Harmless when wheeler_memory is
# already installed / on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wheeler_memory.decoder import extract_state  # noqa: E402
from wheeler_memory.interference import recall_with_interference  # noqa: E402

# Default probe set.  Deliberately includes clusters of closely-related queries
# (e.g. several attention/transformer variants) that tend to recall tightly
# correlated attractors -> TIGHT, alongside spread/science/missing queries, so the
# realized-landscape histogram spans the range.  Override with --corpus/--queries.
DEFAULT_QUERIES = [
    # tight cluster: closely-related ML/attention concepts
    "How does self-attention work in neural networks?",
    "What is the attention mechanism in transformers?",
    "How do query, key and value vectors work in attention?",
    "What is multi-head attention?",
    "What is the transformer architecture?",
    # ML, related but more spread
    "What is backpropagation and how do neural networks learn?",
    "What is gradient descent?",
    "What is the difference between supervised and unsupervised learning?",
    "How do convolutional neural networks process images?",
    # physics cluster
    "Explain quantum superposition in physics",
    "What is quantum entanglement?",
    "What is the uncertainty principle?",
    # biology
    "How does DNA replication work?",
    "What is the structure of DNA?",
    "How does CRISPR gene editing work?",
    # likely-missing (ISOLATED/EMPTY controls)
    "What is the best recipe for sourdough bread?",
    "How do I fix a leaking faucet?",
    "Who won the 2024 presidential election?",
]


def _mean_abs_r(pairs: list) -> float:
    return sum(abs(r) for _, _, r in pairs) / len(pairs) if pairs else 0.0


def _coerce(v):
    """Make a recall-hit value JSON-safe (numpy scalars -> python; long text trimmed)."""
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except (ValueError, TypeError):
            return str(v)
    if isinstance(v, str):
        return v[:240]
    return v


_HIT_KEYS = (
    "text", "similarity", "hex_key", "chunk", "temperature", "temperature_tier",
    "state", "cluster_count", "neg_cluster_count", "grid_entropy", "alive_fraction",
    "correlation_with_stored", "boundary_length", "energy",
)


def _hit_summary(h: dict) -> dict:
    return {k: _coerce(h[k]) for k in _HIT_KEYS if k in h}


def _slug(query: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in query.lower())
    return "-".join(filter(None, keep.split("-")))[:60]


def _load_queries(args) -> list[str]:
    if args.queries:
        return [q.strip() for q in args.queries.split("||") if q.strip()]
    if args.corpus:
        out: list[str] = []
        for line in Path(args.corpus).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["query"])
            except (ValueError, KeyError):
                out.append(line)  # treat as a bare query line
        return out
    return list(DEFAULT_QUERIES)


def sweep(queries: list[str], recall_k: int, data_dir: Path | None) -> list[dict]:
    """Run each query through the run_ablation recall path; return capture records."""
    captures: list[dict] = []
    for i, query in enumerate(queries, 1):
        try:
            hits, interference_state, scm_openness = recall_with_interference(
                query, top_k=recall_k, data_dir=data_dir,
                encoder="blended", use_embedding=True,
            )
        except Exception as e:  # noqa: BLE001 — keep the sweep going, record the failure
            print(f"[{i:2d}/{len(queries)}] {query!r:50.50} -> recall ERROR: {e}")
            continue

        state = extract_state(
            query, hits, interference_state=interference_state,
            scm_openness=scm_openness, data_dir=data_dir,
        )
        mar = _mean_abs_r(state.pairwise_distances)
        cap = {
            "query": query,
            "recall_k": recall_k,
            "data_dir": str(data_dir) if data_dir else None,
            "n_hits": len(hits),
            "confidence": round(float(state.confidence), 4),
            "landscape": state.landscape,
            "mean_abs_r": round(mar, 4),
            "n_pairs": len(state.pairwise_distances),
            "pairwise_distances": [[int(a), int(b), float(r)]
                                   for a, b, r in state.pairwise_distances],
            "interference_state": interference_state,
            "scm_openness": round(float(scm_openness), 4),
            "hits": [_hit_summary(h) for h in hits],
        }
        captures.append(cap)
        print(f"[{i:2d}/{len(queries)}] {query!r:50.50} -> "
              f"{state.landscape:8} conf={cap['confidence']:.3f} "
              f"mean|r|={mar:.3f} ({len(hits)} hits)")
    return captures


def select(captures: list[dict], min_confidence: float, min_mean_abs_r: float) -> dict | None:
    """Highest realized mean|r| among high-confidence captures clearing the floor."""
    eligible = [
        c for c in captures
        if c["confidence"] >= min_confidence and c["mean_abs_r"] > min_mean_abs_r
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["mean_abs_r"])


def _report(captures: list[dict], selected: dict | None,
            min_confidence: float, min_mean_abs_r: float) -> None:
    print("\n" + "=" * 70)
    print("REALIZED-LANDSCAPE HISTOGRAM (binned by actual recall, not intent)")
    print("=" * 70)
    bins: dict[str, int] = {}
    for c in captures:
        bins[c["landscape"]] = bins.get(c["landscape"], 0) + 1
    for label in ("TIGHT", "SPREAD", "ISOLATED", "EMPTY"):
        n = bins.get(label, 0)
        print(f"  {label:9} {'#' * n} {n}")

    print("\n" + "=" * 70)
    print("CAPTURES sorted by mean|r| x confidence (grading strength)")
    print("=" * 70)
    print(f"  {'mean|r|':>8}{'conf':>7}{'land':>10}{'hits':>6}  query")
    for c in sorted(captures, key=lambda c: (c["mean_abs_r"], c["confidence"]), reverse=True):
        print(f"  {c['mean_abs_r']:>8.3f}{c['confidence']:>7.3f}"
              f"{c['landscape']:>10}{c['n_hits']:>6}  {c['query'][:44]}")

    print("\n" + "=" * 70)
    print(f"SELECTION  (confidence >= {min_confidence}, mean|r| > {min_mean_abs_r}, "
          f"max mean|r|; TIGHT preferred)")
    print("=" * 70)
    if selected is None:
        print("  NONE — no capture cleared the floor. Store a denser/related corpus, "
              "lower --min-confidence, or widen --corpus, then re-run.")
        return
    print(f"  query      : {selected['query']!r}")
    print(f"  landscape  : {selected['landscape']}   "
          f"mean|r|={selected['mean_abs_r']:.3f}   conf={selected['confidence']:.3f}")
    print(f"  hits/pairs : {selected['n_hits']} hits, {selected['n_pairs']} pairs")
    dd = f" --data-dir {selected['data_dir']}" if selected["data_dir"] else ""
    print("\n  GRADE DIRECTION ON IT (pairwise-isolated, Claude judge, fully blind):")
    print(f"    python tests/topology_isolation.py --ablation --field-isolation "
          f"--grade --judge-backend claude \\\n"
          f"        --recall-k {selected['recall_k']}{dd} "
          f"--query {selected['query']!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=None, help="Wheeler data dir (default: ~/.wheeler_memory)")
    p.add_argument("--recall-k", type=int, default=5)
    p.add_argument("--corpus", default=None,
                   help="JSONL ({'query': ...} per line) or plain one-query-per-line file")
    p.add_argument("--queries", default=None, help="inline queries separated by '||'")
    p.add_argument("--min-confidence", type=float, default=0.40,
                   help="capture must reach this top-hit similarity to be selectable")
    p.add_argument("--min-mean-abs-r", type=float, default=0.10,
                   help="capture must clear this mean|r| (the SPREAD floor) to be selectable")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: <repo>/results/topology_sweep)")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[2] / "results" / "topology_sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(args)
    print(f"Sweeping {len(queries)} queries (recall_k={args.recall_k}, "
          f"data_dir={data_dir or '~/.wheeler_memory'}) -> {out_dir}")
    captures = sweep(queries, args.recall_k, data_dir)
    if not captures:
        print("No captures produced — is the store populated? "
              "(scripts/wheeler_store.py / --data-dir)", file=sys.stderr)
        return 1

    # Dead-store guard: captures exist but EVERY query recalled nothing. This is
    # the silent trap this harness fell into — recall against an empty/wrong store
    # yields all-EMPTY landscapes and a vacuous "NONE selected" that reads like a
    # topology result. It is not. Fail loudly instead.
    total_hits = sum(c["n_hits"] for c in captures)
    if total_hits == 0:
        print(
            f"\nDEAD STORE: recall returned 0 hits for all {len(captures)} queries "
            f"(data_dir={data_dir or '~/.wheeler_memory'}).\n"
            "  The store is empty, wrong, or its index/embeddings are missing — this is\n"
            "  NOT a topology finding. Point --data-dir at a populated store or\n"
            "  re-crystallize (scripts/wheeler_crystallize.py), then re-run.\n"
            "  No captures written.",
            file=sys.stderr,
        )
        return 2

    selected = select(captures, args.min_confidence, args.min_mean_abs_r)
    _report(captures, selected, args.min_confidence, args.min_mean_abs_r)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for c in captures:
        (out_dir / f"capture_{_slug(c['query'])}.json").write_text(json.dumps(c, indent=2))
    summary = {
        "timestamp": stamp,
        "recall_k": args.recall_k,
        "data_dir": str(data_dir) if data_dir else None,
        "min_confidence": args.min_confidence,
        "min_mean_abs_r": args.min_mean_abs_r,
        "n_captures": len(captures),
        "landscape_histogram": {
            lbl: sum(1 for c in captures if c["landscape"] == lbl)
            for lbl in ("TIGHT", "SPREAD", "ISOLATED", "EMPTY")
        },
        "selected_query": selected["query"] if selected else None,
        "captures": captures,
    }
    (out_dir / f"sweep_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    if selected:
        (out_dir / "selected.json").write_text(json.dumps(selected, indent=2))
    print(f"\nWrote {len(captures)} captures + sweep_summary_{stamp}.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

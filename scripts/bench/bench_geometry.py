"""Wheeler Memory — CA pairwise-geometry faithfulness test (non-LLM, bounded).

The retrieval ablation (bench_ablation.py) settled that the CA does not help —
and mildly hurts — *which* memory comes back. The one surviving question was the
DECODE path: the CA changes the *pairwise correlations* among recalled memories
(~0.075 divergence), and those numbers are fed to the LLM. Is that altered
geometry *better*?

The earlier attempt graded decode output with an LLM judge and was inconclusive
(llama3-judging-llama3 = position bias). This is the non-LLM replacement my notes
flagged as the alternative: a deterministic, judge-free, single-run characterization.

Operationalisation of "better pairwise geometry"
------------------------------------------------
"Better" = more faithful to true semantic relatedness. For a group of memories we
build three pairwise similarity matrices over the group:

  REF  — cosine over MiniLM (all-MiniLM-L6-v2) sentence embeddings  [semantic yardstick]
  RAW  — centered-cosine over the un-evolved hippocampus frames     [the embedding arm]
  CA   — centered-cosine over the CA-evolved attractors             [the system as shipped]

then measure how well RAW and CA rank-align (Spearman over off-diagonal pairs) with
REF. If CA aligns *better* than RAW, the CA sharpens the geometry toward semantic
truth (the thesis survives, narrowly). If worse, the CA distorts it — consistent with
the retrieval finding that the attractor quantization discards information.

This is bounded by construction: fixed #groups × group-size, one deterministic run,
a paired Wilcoxon test, and we report the result whatever it is. No escalation.

Usage
-----
    python scripts/bench/bench_geometry.py
    python scripts/bench/bench_geometry.py --groups 80 --k 12
    python scripts/bench/bench_geometry.py --quick --out geometry.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_associative import _embed  # noqa: E402  (hippocampus n-gram frames)
from corpora import load_mmlu_pool  # noqa: E402


def _offdiag(mat: np.ndarray) -> np.ndarray:
    """Upper-triangle (k>j) entries of a square matrix as a flat vector."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _centered_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise centered cosine (== Pearson r) over rows — the system's metric."""
    v = vectors.astype(np.float32)
    v = v - v.mean(axis=1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-10] = 1.0
    v = v / n
    return v @ v.T


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Plain pairwise cosine over rows — standard MiniLM semantic similarity."""
    v = vectors.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-10] = 1.0
    v = v / n
    return v @ v.T


def _flatten_frames(frames) -> np.ndarray:
    return np.stack([np.asarray(f).flatten().astype(np.float32) for f in frames])


def _evolve_all(frames) -> np.ndarray:
    from wheeler_memory.dynamics import evolve_and_interpret

    atts = []
    for f in frames:
        res = evolve_and_interpret(np.asarray(f))
        att = res["attractor"] if isinstance(res, dict) else res
        atts.append(np.asarray(att).flatten().astype(np.float32))
    return np.stack(atts)


def run(groups: int, k: int, pool_n: int, seed: int) -> dict:
    from scipy.stats import spearmanr, wilcoxon
    from wheeler_memory.embedding import embed_available, embed_text_batch

    if not embed_available():
        print("ERROR: MiniLM embedding model unavailable (sentence-transformers).")
        sys.exit(1)

    print(f"Loading pool of {pool_n} MMLU facts...")
    pool = load_mmlu_pool(pool_n, seed=seed)
    if len(pool) < k:
        print(f"ERROR: pool too small ({len(pool)} < k={k}).")
        sys.exit(1)
    print(f"  Loaded {len(pool)} facts. Sampling {groups} groups of {k}.")

    rng = random.Random(seed)
    align_ca, align_raw = [], []
    spread_ca, spread_raw, spread_ref = [], [], []

    for gi in range(groups):
        grp = rng.sample(pool, k)
        texts = [f["text"] for f in grp]

        ref = _cosine_matrix(embed_text_batch(texts))            # MiniLM semantic
        frames = _embed(texts)
        raw = _centered_cosine_matrix(_flatten_frames(frames))    # un-evolved frames
        ca = _centered_cosine_matrix(_evolve_all(frames))         # CA attractors

        ref_v, raw_v, ca_v = _offdiag(ref), _offdiag(raw), _offdiag(ca)
        a_raw = spearmanr(raw_v, ref_v).correlation
        a_ca = spearmanr(ca_v, ref_v).correlation
        if np.isnan(a_raw) or np.isnan(a_ca):
            continue
        align_raw.append(float(a_raw))
        align_ca.append(float(a_ca))
        spread_ref.append(float(ref_v.std()))
        spread_raw.append(float(raw_v.std()))
        spread_ca.append(float(ca_v.std()))

        if (gi + 1) % max(1, groups // 10) == 0:
            print(f"  [{gi + 1:3d}/{groups}] align CA={np.mean(align_ca):+.3f} "
                  f"RAW={np.mean(align_raw):+.3f}")

    n = len(align_ca)
    m_ca, m_raw = float(np.mean(align_ca)), float(np.mean(align_raw))
    delta = m_ca - m_raw
    ca_wins = int(np.sum(np.array(align_ca) > np.array(align_raw)))
    # Paired test: is the per-group CA-RAW alignment difference != 0?
    try:
        w = wilcoxon(align_ca, align_raw)
        p = float(w.pvalue)
    except ValueError:
        p = float("nan")

    return {
        "n_groups": n, "k": k, "pool": len(pool), "seed": seed,
        "align_ca_mean": m_ca, "align_raw_mean": m_raw, "delta": delta,
        "ca_wins": ca_wins, "ca_win_rate": ca_wins / n if n else 0.0,
        "wilcoxon_p": p,
        "spread_ca": float(np.mean(spread_ca)),
        "spread_raw": float(np.mean(spread_raw)),
        "spread_ref": float(np.mean(spread_ref)),
    }


def _verdict(r: dict) -> str:
    sig = r["wilcoxon_p"] < 0.05 if r["wilcoxon_p"] == r["wilcoxon_p"] else False
    if not sig:
        return "NO DIFFERENCE (CA geometry neither more nor less semantically faithful)"
    if r["delta"] > 0:
        return "CA SHARPENS geometry toward semantic truth (thesis survives, narrowly)"
    return "CA DISTORTS geometry away from semantic truth (consistent with the retrieval kill)"


def main():
    p = argparse.ArgumentParser(description="CA pairwise-geometry faithfulness (non-LLM)")
    p.add_argument("--groups", type=int, default=80)
    p.add_argument("--k", type=int, default=10, help="memories per group")
    p.add_argument("--pool", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.quick:
        args.groups, args.pool = 10, 200

    print("=" * 70)
    print("  WHEELER MEMORY — CA PAIRWISE-GEOMETRY FAITHFULNESS (non-LLM, bounded)")
    print("  Spearman alignment of CA / RAW pairwise structure vs MiniLM semantics.")
    print("=" * 70)
    t0 = time.time()
    r = run(args.groups, args.k, args.pool, args.seed)

    print(f"\n{'=' * 70}")
    print("  RESULT  (Spearman of pairwise structure vs MiniLM semantic similarity)")
    print(f"{'-' * 70}")
    print(f"    groups={r['n_groups']}  k={r['k']}  (pairs/group={r['k']*(r['k']-1)//2})")
    print(f"    RAW (un-evolved frames)  alignment: {r['align_raw_mean']:+.4f}")
    print(f"    CA  (evolved attractors) alignment: {r['align_ca_mean']:+.4f}")
    print(f"    delta (CA - RAW):                   {r['delta']:+.4f}")
    print(f"    CA wins {r['ca_wins']}/{r['n_groups']} groups ({r['ca_win_rate']:.0%})"
          f"   Wilcoxon p={r['wilcoxon_p']:.4g}")
    print(f"    off-diag spread  REF={r['spread_ref']:.3f} "
          f"RAW={r['spread_raw']:.3f} CA={r['spread_ca']:.3f}")
    print(f"{'-' * 70}")
    print(f"    VERDICT: {_verdict(r)}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")
    print("=" * 70)

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))
        print(f"  Raw result written to {args.out}")


if __name__ == "__main__":
    main()

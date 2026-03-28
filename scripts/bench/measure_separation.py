#!/usr/bin/env python3
"""Measure attractor separation under random vs learned projection.

Computes pairwise Pearson distance distribution for all stored attractors
and compares the distributions to assess whether a learned projection
improves basin separation.

Usage:
    python scripts/bench/measure_separation.py
    python scripts/bench/measure_separation.py --max-pairs 1000 --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _load_attractors(data_dir: Path) -> list[np.ndarray]:
    """Load all CONVERGED attractor arrays from storage."""
    attractors: list[np.ndarray] = []
    chunks_dir = data_dir / "chunks"
    if not chunks_dir.exists():
        return attractors

    for chunk_dir in sorted(chunks_dir.iterdir()):
        index_path = chunk_dir / "index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        for hex_key, meta in index.items():
            if meta.get("state") != "CONVERGED":
                continue
            att_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if att_path.exists():
                attractors.append(np.load(att_path))
    return attractors


def _sample_pairs(
    attractors: list[np.ndarray], max_pairs: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Sample up to max_pairs unique (i, j) pairs."""
    n = len(attractors)
    total_possible = n * (n - 1) // 2
    if total_possible <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        pairs.add((int(i), int(j)))
    return list(pairs)


def _pairwise_distances(
    attractors: list[np.ndarray], pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Compute Pearson correlation for each pair."""
    distances = np.zeros(len(pairs))
    for k, (i, j) in enumerate(pairs):
        r, _ = pearsonr(attractors[i].flatten(), attractors[j].flatten())
        distances[k] = float(r)
        if (k + 1) % 100 == 0:
            print(f"  Computed {k + 1}/{len(pairs)} pairs...", flush=True)
    return distances


def _print_stats(label: str, distances: np.ndarray) -> None:
    print(f"\n  {label}:")
    print(f"    N pairs:    {len(distances)}")
    print(f"    Mean r:     {distances.mean():.4f}")
    print(f"    Median r:   {np.median(distances):.4f}")
    print(f"    Std r:      {distances.std():.4f}")
    print(f"    Min r:      {distances.min():.4f}")
    print(f"    Max r:      {distances.max():.4f}")
    pcts = np.percentile(distances, [5, 25, 75, 95])
    print(f"    P5/P25/P75/P95: {pcts[0]:.3f} / {pcts[1]:.3f} / {pcts[2]:.3f} / {pcts[3]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Measure attractor separation")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--plot", action="store_true", help="Save histogram plot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("ATTRACTOR SEPARATION ANALYSIS")
    print("=" * 70)

    print("\nLoading stored attractors...")
    attractors = _load_attractors(data_dir)
    print(f"  Found {len(attractors)} CONVERGED attractors")

    if len(attractors) < 3:
        print("ERROR: need at least 3 attractors for meaningful analysis.")
        raise SystemExit(1)

    pairs = _sample_pairs(attractors, args.max_pairs, rng)
    print(f"\nComputing pairwise Pearson for {len(pairs)} pairs...")

    distances = _pairwise_distances(attractors, pairs)
    _print_stats("Current attractor distances", distances)

    # Separation quality: fraction of pairs with |r| < 0.1 (well-separated)
    well_separated = (np.abs(distances) < 0.1).mean()
    tightly_clustered = (np.abs(distances) > 0.4).mean()
    print(f"\n  Well-separated (|r| < 0.1): {well_separated:.1%}")
    print(f"  Tightly clustered (|r| > 0.4): {tightly_clustered:.1%}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(distances, bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Pairwise Pearson r")
            ax.set_ylabel("Count")
            ax.set_title(f"Attractor Separation ({len(attractors)} memories, {len(pairs)} pairs)")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)
            plot_path = data_dir / "separation_histogram.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"\n  Plot saved → {plot_path}")
            plt.close(fig)
        except ImportError:
            print("\n  matplotlib not available, skipping plot.")

    print(f"\n{'─' * 70}")


if __name__ == "__main__":
    main()

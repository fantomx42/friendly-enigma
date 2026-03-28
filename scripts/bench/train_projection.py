#!/usr/bin/env python3
"""Train a learned projection matrix from sentence embeddings to attractor seeds.

Replaces the fixed Gaussian random matrix in embedding.py with one that maps
embeddings to frame seeds that evolve to attractors close to already-stored ones.

Why this helps:
  - Random projection: a new query embedding lands at an arbitrary point in frame
    space — no connection to known attractor basins.
  - Learned projection: a query embedding lands near the frame seed that produced
    similar stored attractors → CA dynamics converge to a similar attractor →
    higher Pearson correlation → better recall.

Training objective:
  Ridge regression from sentence embeddings → pre-tanh frame seeds recovered
  from stored attractors.  Fidelity is measured as Pearson(evolved_seed, stored).

Requires: pip install -e ".[train]"   # scikit-learn

Usage:
    python scripts/train_projection.py
    python scripts/train_projection.py --data-dir /path/to/data --alpha 0.5
    python scripts/train_projection.py --dry-run   # evaluate only, don't save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.embedding import (
    EMBED_DIM,
    FRAME_CELLS,
    FRAME_SIZE,
    PROJECTION_SEED,
    _LEARNED_PROJECTION_PATH,
    embed_text_batch,
)

DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _load_corpus(data_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """Load all CONVERGED, non-polar stored memories.

    Returns (texts, attractors) where each attractor is a (64, 64) float32 array.
    """
    texts: list[str] = []
    attractors: list[np.ndarray] = []

    chunks_dir = data_dir / "chunks"
    if not chunks_dir.exists():
        return texts, attractors

    for chunk_dir in sorted(chunks_dir.iterdir()):
        index_path = chunk_dir / "index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        for hex_key, meta in index.items():
            # Skip non-converged, special memory types, and experiential entries
            # (experiential uses different CA dynamics and attractor structure)
            if meta.get("state") != "CONVERGED":
                continue
            mtype = meta.get("metadata", {}).get("memory_type")
            if mtype in ("polar", "avoidance"):
                continue
            if meta.get("metadata", {}).get("grid") == "experiential":
                continue
            att_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if not att_path.exists():
                continue
            texts.append(meta["text"])
            attractors.append(np.load(att_path))

    return texts, attractors


def _random_projection() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(PROJECTION_SEED))
    return rng.standard_normal((EMBED_DIM, FRAME_CELLS)).astype(np.float32) / np.sqrt(
        FRAME_CELLS
    )


def _fidelity(frames: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """Pearson correlation between each evolved frame and its stored attractor.

    Parameters
    ----------
    frames : (N, 64, 64) float32
    stored : (N, 64, 64) float32

    Returns
    -------
    (N,) float array of Pearson correlations
    """
    scores = []
    for i, (frame, att) in enumerate(zip(frames, stored)):
        result = evolve_and_interpret(frame)
        evolved = result["attractor"]
        try:
            r, _ = pearsonr(evolved.flatten(), att.flatten())
            scores.append(float(r))
        except ValueError:
            scores.append(0.0)
        if (i + 1) % 10 == 0:
            print(f"  Evolved {i + 1}/{len(frames)} frames...", flush=True)
    return np.array(scores)


def train(
    data_dir: Path,
    alpha: float = 1.0,
    test_frac: float = 0.2,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> None:
    print("=" * 70)
    print("PROJECTION TRAINING")
    print("=" * 70)

    # ── Step 1: Load corpus ───────────────────────────────────────────────────
    print("\nLoading stored memories...")
    texts, attractors = _load_corpus(data_dir)
    N = len(texts)
    if N < 5:
        print(f"ERROR: only {N} CONVERGED memories found — need at least 5 to train.")
        raise SystemExit(1)
    print(f"  Found {N} memories")

    # ── Step 2: Embeddings ────────────────────────────────────────────────────
    print("\nComputing sentence embeddings...")
    E = embed_text_batch(texts)  # (N, 384)
    print(f"  E shape: {E.shape}")

    # ── Step 3: Recover pre-tanh seeds ───────────────────────────────────────
    A = np.stack([a.flatten() for a in attractors])  # (N, 4096)
    # Invert: frame_flat = tanh(x * 3.0)  →  x = arctanh(frame_flat) / 3.0
    A_seed = np.arctanh(np.clip(A, -0.9999, 0.9999)) / 3.0  # (N, 4096)
    print(f"  A_seed shape: {A_seed.shape}, std={A_seed.std():.3f}")

    # ── Step 4: Train/test split ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    split = max(1, int(N * (1 - test_frac)))
    train_idx, test_idx = idx[:split], idx[split:]
    E_train, E_test = E[train_idx], E[test_idx]
    A_seed_train, A_seed_test = A_seed[train_idx], A_seed[test_idx]
    A_test = A[test_idx]
    attractors_test = [attractors[i] for i in test_idx]
    print(f"\n  Train: {len(train_idx)}  Test: {len(test_idx)}")

    # ── Step 5: Ridge regression ──────────────────────────────────────────────
    print(f"\nFitting Ridge regression (alpha={alpha})...")
    assert A_seed_train.shape == (len(train_idx), FRAME_CELLS), (
        f"Expected ({len(train_idx)}, {FRAME_CELLS}), got {A_seed_train.shape}"
    )
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(E_train, A_seed_train)
    W_learned = reg.coef_.T.astype(np.float32)  # (384, 4096)
    print(f"  W_learned shape: {W_learned.shape}, norm={np.linalg.norm(W_learned):.3f}")

    # Train residual (proxy for fit quality, no CA needed)
    A_seed_pred_train = E_train @ W_learned
    train_rmse = float(np.sqrt(np.mean((A_seed_pred_train - A_seed_train) ** 2)))
    print(f"  Train RMSE (seed space): {train_rmse:.4f}")

    # ── Step 6: Evaluate fidelity on test set ─────────────────────────────────
    if len(test_idx) == 0:
        print("\nSkipping fidelity evaluation (no test examples).")
        scores_learned = scores_random = np.array([])
    else:
        W_random = _random_projection()

        def _frames_from(W: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
            seeds = embeddings @ W  # (M, 4096)
            frames_flat = np.tanh(seeds * 3.0)
            return frames_flat.reshape(-1, FRAME_SIZE, FRAME_SIZE).astype(np.float32)

        print(
            f"\nEvaluating learned projection fidelity on {len(test_idx)} test examples..."
        )
        frames_learned = _frames_from(W_learned, E_test)
        scores_learned = _fidelity(frames_learned, attractors_test)

        print(
            f"\nEvaluating random projection fidelity on {len(test_idx)} test examples..."
        )
        frames_random = _frames_from(W_random, E_test)
        scores_random = _fidelity(frames_random, attractors_test)

        mean_l = float(scores_learned.mean())
        mean_r = float(scores_random.mean())
        delta = mean_l - mean_r
        print(f"\n{'─' * 70}")
        print("FIDELITY RESULTS")
        print(f"{'─' * 70}")
        print(f"  Learned projection:  mean Pearson = {mean_l:.4f}")
        print(f"  Random projection:   mean Pearson = {mean_r:.4f}")
        print(f"  Improvement:        {delta:+.4f}")

        if delta > 0.02:
            print("\n  ✓  Learned projection improves attractor fidelity.")
        elif delta > 0:
            print("\n  ⚡  Marginal improvement — may not be worth switching.")
        else:
            print(
                "\n  ✗  Learned projection does not improve fidelity. "
                "The bottleneck is elsewhere (CA dynamics, corpus size, or "
                "attractor landscape geometry)."
            )

        # Per-example delta histogram (ASCII)
        deltas = scores_learned - scores_random
        bins = np.array([-0.3, -0.1, -0.05, 0, 0.05, 0.1, 0.3])
        counts, _ = np.histogram(deltas, bins=bins)
        print("\n  Per-example Δ distribution:")
        labels = ["< -0.10", "-0.10–-0.05", "-0.05–0", "0–0.05", "0.05–0.10", "> 0.10"]
        for label, count in zip(labels, counts):
            bar = "█" * count
            print(f"    {label:>12}: {bar} ({count})")

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    if not dry_run:
        save_path = output_path or _LEARNED_PROJECTION_PATH
        np.save(save_path, W_learned)
        print(f"\n  Saved learned projection → {save_path}")
        print(
            "  embedding.py will auto-load this on next import "
            "(clear _projection_matrix cache if already running)."
        )
    else:
        print("\n  --dry-run: projection NOT saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Train learned projection for Wheeler Memory"
    )
    parser.add_argument("--data-dir", default=None, help="Wheeler data directory")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization (default: 1.0)"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.2, help="Fraction held out for evaluation"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate only, do not save"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for .npy (default: ~/.wheeler_memory/learned_projection.npy)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    output_path = Path(args.output) if args.output else None

    train(
        data_dir=data_dir,
        alpha=args.alpha,
        test_frac=args.test_frac,
        dry_run=args.dry_run,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

"""Shared similarity functions for comparing CA attractor grids.

Provides three similarity modes:
  - **pearson**: Pearson correlation on flattened 4096-vectors (same metric
    used throughout the core pipeline, extracted here for reuse).
  - **spatial**: Cosine similarity on 51-dim topology feature vectors
    (cluster counts, sizes, centroids, boundary length).
  - **hybrid**: Weighted blend of pearson and spatial components.

All functions accept (64, 64) grids or (4096,) flat vectors and handle
zero-variance edge cases.  They are pure functions with no side effects
or disk I/O, suitable for benchmarking and analysis.
"""

from __future__ import annotations

import numpy as np

from .constants import SPATIAL_PEARSON_WEIGHT
from .dynamics import extract_spatial_features


def _ensure_flat(arr: np.ndarray) -> np.ndarray:
    """Flatten to 1-D if needed."""
    return arr.ravel() if arr.ndim > 1 else arr


def _ensure_grid(arr: np.ndarray) -> np.ndarray:
    """Reshape to (64, 64) if needed."""
    return arr.reshape(64, 64) if arr.ndim == 1 else arr


def pearson_similarity(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Pearson correlation between two attractor grids.

    Parameters
    ----------
    grid_a, grid_b : ndarray
        Shape (64, 64) or (4096,).  Values in [-1, +1].

    Returns
    -------
    float
        Pearson r in [-1, +1].  Returns 0.0 if either array has zero variance.
    """
    a = _ensure_flat(grid_a).astype(np.float64)
    b = _ensure_flat(grid_b).astype(np.float64)

    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std()
    b_std = b.std()

    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0

    return float(np.dot(a - a_mean, b - b_mean) / (len(a) * a_std * b_std))


def spatial_cosine_similarity(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Cosine similarity between 51-dim spatial feature vectors.

    Extracts cluster topology features (counts, sizes, centroids, boundary
    length) from each grid via :func:`extract_spatial_features`, then computes
    cosine similarity.

    Parameters
    ----------
    grid_a, grid_b : ndarray
        Shape (64, 64) or (4096,).

    Returns
    -------
    float
        Cosine similarity in [-1, +1].  Returns 0.0 if either vector is zero.
    """
    fa = extract_spatial_features(_ensure_grid(grid_a))
    fb = extract_spatial_features(_ensure_grid(grid_b))

    na = np.linalg.norm(fa)
    nb = np.linalg.norm(fb)

    if na < 1e-10 or nb < 1e-10:
        return 0.0

    return float(np.dot(fa, fb) / (na * nb))


def hybrid_similarity(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    pearson_weight: float | None = None,
) -> float:
    """Blended Pearson + spatial topology similarity.

    Parameters
    ----------
    grid_a, grid_b : ndarray
        Shape (64, 64) or (4096,).
    pearson_weight : float, optional
        Weight for Pearson component in [0, 1].  Defaults to
        ``SPATIAL_PEARSON_WEIGHT`` from constants.py.

    Returns
    -------
    float
        ``pearson_weight * pearson_r + (1 - pearson_weight) * spatial_cosine``
    """
    if pearson_weight is None:
        pearson_weight = SPATIAL_PEARSON_WEIGHT

    p = pearson_similarity(grid_a, grid_b)
    s = spatial_cosine_similarity(grid_a, grid_b)

    return pearson_weight * p + (1.0 - pearson_weight) * s

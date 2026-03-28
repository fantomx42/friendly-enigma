"""Tests for wheeler_memory.similarity — shared hybrid similarity functions."""

import numpy as np
import pytest

from wheeler_memory.similarity import (
    hybrid_similarity,
    pearson_similarity,
    spatial_cosine_similarity,
)


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(42))


@pytest.fixture
def grid_a(rng):
    """Random 64x64 grid in [-1, +1]."""
    return rng.uniform(-1, 1, (64, 64)).astype(np.float32)


@pytest.fixture
def grid_b(rng):
    """Different random 64x64 grid."""
    # Use a different seed region by drawing twice
    _ = rng.uniform(-1, 1, (64, 64))
    return rng.uniform(-1, 1, (64, 64)).astype(np.float32)


# ── Pearson ──────────────────────────────────────────────────────────


def test_pearson_identical(grid_a):
    assert pearson_similarity(grid_a, grid_a) == pytest.approx(1.0, abs=1e-6)


def test_pearson_negated(grid_a):
    assert pearson_similarity(grid_a, -grid_a) == pytest.approx(-1.0, abs=1e-6)


def test_pearson_zero_variance():
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    grid = np.random.default_rng(0).uniform(-1, 1, (64, 64)).astype(np.float32)
    assert pearson_similarity(flat, grid) == 0.0


def test_pearson_accepts_flat(grid_a):
    flat = grid_a.ravel()
    assert pearson_similarity(flat, flat) == pytest.approx(1.0, abs=1e-6)


def test_pearson_mixed_shapes(grid_a):
    """One 2-D, one 1-D — should still work."""
    flat = grid_a.ravel()
    assert pearson_similarity(grid_a, flat) == pytest.approx(1.0, abs=1e-6)


# ── Spatial cosine ───────────────────────────────────────────────────


def test_spatial_identical(grid_a):
    assert spatial_cosine_similarity(grid_a, grid_a) == pytest.approx(1.0, abs=1e-6)


def test_spatial_different(grid_a, grid_b):
    sim = spatial_cosine_similarity(grid_a, grid_b)
    assert -1.0 <= sim <= 1.0


def test_spatial_zero_grid():
    zero = np.zeros((64, 64), dtype=np.float32)
    grid = np.random.default_rng(0).uniform(-1, 1, (64, 64)).astype(np.float32)
    assert spatial_cosine_similarity(zero, grid) == 0.0


def test_spatial_accepts_flat(grid_a):
    flat = grid_a.ravel()
    assert spatial_cosine_similarity(flat, flat) == pytest.approx(1.0, abs=1e-6)


# ── Hybrid ───────────────────────────────────────────────────────────


def test_hybrid_weight_one_equals_pearson(grid_a, grid_b):
    h = hybrid_similarity(grid_a, grid_b, pearson_weight=1.0)
    p = pearson_similarity(grid_a, grid_b)
    assert h == pytest.approx(p, abs=1e-6)


def test_hybrid_weight_zero_equals_spatial(grid_a, grid_b):
    h = hybrid_similarity(grid_a, grid_b, pearson_weight=0.0)
    s = spatial_cosine_similarity(grid_a, grid_b)
    assert h == pytest.approx(s, abs=1e-6)


def test_hybrid_default_weight(grid_a, grid_b):
    """Default weight should produce a value between pearson and spatial."""
    h = hybrid_similarity(grid_a, grid_b)
    p = pearson_similarity(grid_a, grid_b)
    s = spatial_cosine_similarity(grid_a, grid_b)
    lo, hi = min(p, s), max(p, s)
    assert lo - 1e-6 <= h <= hi + 1e-6


def test_hybrid_identical(grid_a):
    assert hybrid_similarity(grid_a, grid_a) == pytest.approx(1.0, abs=1e-6)

"""Tests for wheeler_memory.theories.lichtenberg — topology visualization."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from wheeler_memory.theories.lichtenberg import _pca_2d, plot_topology


# ── _pca_2d ───────────────────────────────────────────────────────────


class TestPCA2D:
    """_pca_2d() SVD-based dimensionality reduction."""

    def test_output_shape(self):
        """N vectors → (N, 2)."""
        vecs = np.random.default_rng(42).standard_normal((10, 100)).astype(np.float32)
        result = _pca_2d(vecs)
        assert result.shape == (10, 2)

    def test_single_vector(self):
        """Single vector → (1, 2) zeros."""
        vecs = np.random.default_rng(42).standard_normal((1, 50)).astype(np.float32)
        result = _pca_2d(vecs)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, 0.0)

    def test_two_vectors(self):
        """Two vectors → (2, 2) with opposite signs."""
        vecs = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
        result = _pca_2d(vecs)
        assert result.shape == (2, 2)
        # Opposite points should have opposite projections
        assert result[0, 0] != pytest.approx(result[1, 0], abs=1e-5)

    def test_preserves_relative_distances(self):
        """PCA preserves that cluster A is closer within than across clusters."""
        rng = np.random.default_rng(42)
        cluster_a = rng.standard_normal((5, 100)) + 10
        cluster_b = rng.standard_normal((5, 100)) - 10
        vecs = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        result = _pca_2d(vecs)
        # Mean of cluster A should be far from mean of cluster B in 2D
        mean_a = result[:5].mean(axis=0)
        mean_b = result[5:].mean(axis=0)
        dist_between = np.linalg.norm(mean_a - mean_b)
        assert dist_between > 1.0


# ── plot_topology ─────────────────────────────────────────────────────


class TestPlotTopology:
    """plot_topology() figure generation."""

    def test_empty_attractors(self):
        """No attractors → returns figure with 'No attractors' message."""
        fig = plot_topology({}, {}, {})
        assert isinstance(fig, Figure)

    def test_returns_figure(self):
        """With attractors → returns a matplotlib Figure."""
        rng = np.random.default_rng(42)
        attractors = {
            "a": rng.uniform(-1, 1, (64, 64)).astype(np.float32),
            "b": rng.uniform(-1, 1, (64, 64)).astype(np.float32),
            "c": rng.uniform(-1, 1, (64, 64)).astype(np.float32),
        }
        widths = {"a": 0.3, "b": 0.5, "c": 0.2}
        counts = {"a": 5, "b": 10, "c": 1}
        fig = plot_topology(attractors, widths, counts)
        assert isinstance(fig, Figure)

    def test_with_query(self):
        """Query attractor highlighted in the plot."""
        rng = np.random.default_rng(42)
        attractors = {
            "a": rng.uniform(-1, 1, (64, 64)).astype(np.float32),
            "b": rng.uniform(-1, 1, (64, 64)).astype(np.float32),
        }
        query = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        fig = plot_topology(attractors, {"a": 0.3, "b": 0.5}, {"a": 1, "b": 2}, query)
        assert isinstance(fig, Figure)

    def test_with_candidates(self):
        """Candidate nodes appear as dotted circles."""
        rng = np.random.default_rng(42)
        attractors = {"a": rng.uniform(-1, 1, (64, 64)).astype(np.float32)}
        candidate = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        fig = plot_topology(
            attractors, {"a": 0.5}, {"a": 3},
            candidates=[candidate],
        )
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_path):
        """output_path → PNG file created."""
        rng = np.random.default_rng(42)
        attractors = {"a": rng.uniform(-1, 1, (64, 64)).astype(np.float32)}
        path = tmp_path / "topology.png"
        plot_topology(attractors, {"a": 0.3}, {"a": 1}, output_path=path)
        assert path.exists()
        assert path.stat().st_size > 0

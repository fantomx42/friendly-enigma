"""Unit tests for trajectory signature system in Wheeler Memory.

Tests the compact fingerprinting of CA evolution trajectories, including
downsampling, correlation computation, signature extraction, similarity matching,
and batch operations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.trajectory import (
    TrajectorySignature,
    batch_similarity,
    compute_signature,
    compute_signature_batch,
    load_signature,
    save_signature,
    signature_similarity,
    _downsample,
    _pearson,
)


class TestDownsample:
    """Tests for _downsample: variable-length curve → fixed-length (10,) array."""

    def test_downsample_empty_list(self):
        """Empty list → (10,) zeros."""
        result = _downsample([])
        assert result.shape == (10,)
        assert np.allclose(result, 0.0)

    def test_downsample_single_value(self):
        """Single value [x] → (10,) all x."""
        result = _downsample([5.0])
        assert result.shape == (10,)
        assert np.allclose(result, 5.0)

    def test_downsample_exact_length(self):
        """Input length == 10 → interpolation is identity."""
        values = np.linspace(0, 1, 10).tolist()
        result = _downsample(values)
        assert result.shape == (10,)
        np.testing.assert_allclose(result, values, atol=1e-5)

    def test_downsample_longer_than_target(self):
        """Input length > 10 → downsampled via linear interp."""
        values = np.linspace(0, 1, 20).tolist()
        result = _downsample(values)
        assert result.shape == (10,)
        # First and last should be preserved
        assert np.isclose(result[0], 0.0, atol=1e-5)
        assert np.isclose(result[-1], 1.0, atol=1e-5)

    def test_downsample_shorter_than_target(self):
        """Input length < 10 → upsampled via linear interp."""
        values = [0.0, 1.0]
        result = _downsample(values)
        assert result.shape == (10,)
        assert np.isclose(result[0], 0.0, atol=1e-5)
        assert np.isclose(result[-1], 1.0, atol=1e-5)
        # Intermediate values should monotonically increase
        assert np.all(np.diff(result) >= -1e-5)

    def test_downsample_dtype_float32(self):
        """Output dtype is float32."""
        result = _downsample([1.0, 2.0])
        assert result.dtype == np.float32

    def test_downsample_negative_values(self):
        """Handles negative values correctly."""
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        result = _downsample(values)
        assert result.shape == (10,)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestPearson:
    """Tests for _pearson: Pearson correlation coefficient."""

    def test_pearson_identical_arrays(self):
        """Identical arrays → correlation ~1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a.copy()
        corr = _pearson(a, b)
        assert np.isclose(corr, 1.0, atol=1e-5)

    def test_pearson_perfect_negative(self):
        """Negative linear relationship → correlation ~-1.0, clamped to 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = -a
        corr = _pearson(a, b)
        # Since negative correlations are clamped in similarity, verify it computes correctly
        assert corr < 0.0  # Raw pearson is negative

    def test_pearson_uncorrelated_arrays(self):
        """Uncorrelated arrays → low absolute correlation."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Perfect negative correlation
        corr = _pearson(a, b)
        # Perfect negative correlation returns negative value
        assert corr < -0.99

    def test_pearson_constant_array_returns_zero(self):
        """Constant array has zero std → returns 0.0."""
        a = np.array([5.0, 5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        corr = _pearson(a, b)
        assert corr == 0.0

    def test_pearson_both_constant_returns_zero(self):
        """Both constant → returns 0.0."""
        a = np.array([3.0, 3.0, 3.0])
        b = np.array([7.0, 7.0, 7.0])
        corr = _pearson(a, b)
        assert corr == 0.0

    def test_pearson_scaled_array(self):
        """Scaled copy of array → correlation 1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a * 2.0 + 5.0  # Different scale and offset
        corr = _pearson(a, b)
        assert np.isclose(corr, 1.0, atol=1e-5)

    def test_pearson_return_type(self):
        """Returns a float."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        corr = _pearson(a, b)
        assert isinstance(corr, (float, np.floating))


class TestComputeSignature:
    """Tests for compute_signature: extract trajectory fingerprint."""

    def test_compute_signature_returns_trajectory_signature(self):
        """Output is a TrajectorySignature instance."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert isinstance(sig, TrajectorySignature)

    def test_compute_signature_energy_curve_shape(self):
        """energy_curve is shape (10,)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.energy_curve.shape == (10,)

    def test_compute_signature_role_curve_shape(self):
        """role_curve is shape (10,)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.role_curve.shape == (10,)

    def test_compute_signature_final_role_distribution_shape(self):
        """final_role_distribution is shape (3,)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.final_role_distribution.shape == (3,)

    def test_compute_signature_role_distribution_sums_to_one(self):
        """final_role_distribution [max, min, slope] sums to ~1.0."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        total = sig.final_role_distribution.sum()
        assert np.isclose(total, 1.0, atol=1e-5)

    def test_compute_signature_state_valid(self):
        """state is one of CONVERGED, OSCILLATING, CHAOTIC."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.state in ("CONVERGED", "OSCILLATING", "CHAOTIC")

    def test_compute_signature_convergence_ticks_positive(self):
        """convergence_ticks > 0."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.convergence_ticks > 0

    def test_compute_signature_convergence_ticks_respects_max_iters(self):
        """convergence_ticks <= max_iters."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        max_iters = 50
        sig = compute_signature(frame, max_iters=max_iters)
        assert sig.convergence_ticks <= max_iters

    def test_compute_signature_seed_attractor_corr_in_range(self):
        """seed_attractor_corr is a float in [-1, 1]."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert isinstance(sig.seed_attractor_corr, (float, np.floating))
        assert -1.0 <= sig.seed_attractor_corr <= 1.0

    def test_compute_signature_constant_frame_converges_quickly(self):
        """Constant frame (all zeros) converges in very few ticks."""
        frame = np.zeros((64, 64), dtype=np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.state == "CONVERGED"
        # Uniform frame is a fixed point; should converge in 1 tick
        assert sig.convergence_ticks == 1

    def test_compute_signature_curves_are_float32(self):
        """energy_curve and role_curve are float32."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert sig.energy_curve.dtype == np.float32
        assert sig.role_curve.dtype == np.float32
        assert sig.final_role_distribution.dtype == np.float32

    def test_compute_signature_energy_curve_non_negative(self):
        """energy_curve values are non-negative (mean abs delta)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert np.all(sig.energy_curve >= 0.0)

    def test_compute_signature_role_curve_in_range(self):
        """role_curve values in [0, 1] (fraction of slope cells)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert np.all(sig.role_curve >= 0.0)
        assert np.all(sig.role_curve <= 1.0)

    def test_compute_signature_role_distribution_fractions(self):
        """Each component of final_role_distribution in [0, 1]."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        assert np.all(sig.final_role_distribution >= 0.0)
        assert np.all(sig.final_role_distribution <= 1.0)

    def test_compute_signature_deterministic(self):
        """Same seed frame → same signature."""
        frame = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        sig1 = compute_signature(frame.copy(), max_iters=50)
        sig2 = compute_signature(frame.copy(), max_iters=50)
        assert sig1.convergence_ticks == sig2.convergence_ticks
        assert sig1.state == sig2.state
        np.testing.assert_array_equal(sig1.energy_curve, sig2.energy_curve)
        np.testing.assert_array_equal(sig1.role_curve, sig2.role_curve)
        np.testing.assert_array_equal(
            sig1.final_role_distribution, sig2.final_role_distribution
        )


class TestSignatureSimilarity:
    """Tests for signature_similarity: compare two signatures."""

    def test_signature_similarity_same_signature(self):
        """signature_similarity(s, s) == 1.0."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)
        sim = signature_similarity(sig, sig)
        assert np.isclose(sim, 1.0, atol=1e-5)

    def test_signature_similarity_range(self):
        """signature_similarity always in [0, 1]."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig1 = compute_signature(frame1, max_iters=50)
        sig2 = compute_signature(frame2, max_iters=50)
        sim = signature_similarity(sig1, sig2)
        assert 0.0 <= sim <= 1.0

    def test_signature_similarity_very_different_low(self):
        """Very different signatures → low similarity."""
        # Create two maximally different signatures
        sig1 = TrajectorySignature(
            convergence_ticks=10,
            energy_curve=np.zeros(10, dtype=np.float32),
            role_curve=np.zeros(10, dtype=np.float32),
            final_role_distribution=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=1.0,
        )
        sig2 = TrajectorySignature(
            convergence_ticks=100,
            energy_curve=np.ones(10, dtype=np.float32),
            role_curve=np.ones(10, dtype=np.float32),
            final_role_distribution=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            state="CHAOTIC",
            seed_attractor_corr=-1.0,
        )
        sim = signature_similarity(sig1, sig2)
        assert sim < 0.5  # Should be quite low

    def test_signature_similarity_symmetric(self):
        """signature_similarity(a, b) == signature_similarity(b, a)."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig1 = compute_signature(frame1, max_iters=50)
        sig2 = compute_signature(frame2, max_iters=50)
        sim_12 = signature_similarity(sig1, sig2)
        sim_21 = signature_similarity(sig2, sig1)
        assert np.isclose(sim_12, sim_21, atol=1e-5)

    def test_signature_similarity_return_type(self):
        """Returns a float."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig1 = compute_signature(frame1, max_iters=50)
        sig2 = compute_signature(frame2, max_iters=50)
        sim = signature_similarity(sig1, sig2)
        assert isinstance(sim, (float, np.floating))


class TestSaveLoadSignature:
    """Tests for save_signature / load_signature round-tripping."""

    def test_save_load_signature_roundtrip_convergence_ticks(self, tmp_path):
        """convergence_ticks preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        assert sig_loaded.convergence_ticks == sig_orig.convergence_ticks

    def test_save_load_signature_roundtrip_state(self, tmp_path):
        """state preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        assert sig_loaded.state == sig_orig.state

    def test_save_load_signature_roundtrip_energy_curve(self, tmp_path):
        """energy_curve preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        np.testing.assert_array_almost_equal(
            sig_loaded.energy_curve, sig_orig.energy_curve, decimal=5
        )

    def test_save_load_signature_roundtrip_role_curve(self, tmp_path):
        """role_curve preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        np.testing.assert_array_almost_equal(
            sig_loaded.role_curve, sig_orig.role_curve, decimal=5
        )

    def test_save_load_signature_roundtrip_role_distribution(self, tmp_path):
        """final_role_distribution preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        np.testing.assert_array_almost_equal(
            sig_loaded.final_role_distribution,
            sig_orig.final_role_distribution,
            decimal=5,
        )

    def test_save_load_signature_roundtrip_seed_attractor_corr(self, tmp_path):
        """seed_attractor_corr preserved after save/load."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        assert np.isclose(
            sig_loaded.seed_attractor_corr, sig_orig.seed_attractor_corr, atol=1e-5
        )

    def test_save_load_signature_file_exists(self, tmp_path):
        """save_signature creates a file."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig = compute_signature(frame, max_iters=50)

        sig_path = tmp_path / "test_sig.npz"
        save_signature(sig, sig_path)

        assert sig_path.exists()

    def test_save_load_signature_with_string_path(self, tmp_path):
        """save/load work with string paths (not just Path objects)."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        sig_orig = compute_signature(frame, max_iters=50)

        sig_path = str(tmp_path / "test_sig.npz")
        save_signature(sig_orig, sig_path)
        sig_loaded = load_signature(sig_path)

        assert sig_loaded.convergence_ticks == sig_orig.convergence_ticks


class TestBatchSimilarity:
    """Tests for batch_similarity: vectorized similarity computation."""

    def test_batch_similarity_empty_arrays(self):
        """Empty batch arrays → empty result."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        query = compute_signature(frame, max_iters=50)

        empty_energy = np.empty((0, 10), dtype=np.float32)
        empty_roles = np.empty((0, 10), dtype=np.float32)
        empty_dist = np.empty((0, 3), dtype=np.float32)
        empty_ticks = np.empty((0,), dtype=np.int32)
        empty_corrs = np.empty((0,), dtype=np.float32)

        result = batch_similarity(
            query, empty_energy, empty_roles, empty_dist, empty_ticks, empty_corrs
        )
        assert result.shape == (0,)

    def test_batch_similarity_single_entry(self):
        """Single entry matches signature_similarity result."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)

        query = compute_signature(frame1, max_iters=50)
        target = compute_signature(frame2, max_iters=50)

        # Pack target into batch arrays
        energy_curves = target.energy_curve.reshape(1, 10)
        role_curves = target.role_curve.reshape(1, 10)
        role_dists = target.final_role_distribution.reshape(1, 3)
        ticks = np.array([target.convergence_ticks], dtype=np.int32)
        corrs = np.array([target.seed_attractor_corr], dtype=np.float32)

        batch_result = batch_similarity(
            query, energy_curves, role_curves, role_dists, ticks, corrs
        )
        scalar_result = signature_similarity(query, target)

        assert np.isclose(batch_result[0], scalar_result, atol=1e-4)

    def test_batch_similarity_result_shape(self):
        """Result shape is (N,) where N is batch size."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        query = compute_signature(frame, max_iters=50)

        N = 5
        energy_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_dists = rng.uniform(0, 1, (N, 3)).astype(np.float32)
        # Normalize role distributions to sum to 1
        role_dists = role_dists / role_dists.sum(axis=1, keepdims=True)
        ticks = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        corrs = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        result = batch_similarity(
            query, energy_curves, role_curves, role_dists, ticks, corrs
        )
        assert result.shape == (N,)

    def test_batch_similarity_result_range(self):
        """All similarities in [0, 1]."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        query = compute_signature(frame, max_iters=50)

        N = 10
        energy_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_dists = rng.uniform(0, 1, (N, 3)).astype(np.float32)
        role_dists = role_dists / role_dists.sum(axis=1, keepdims=True)
        ticks = np.arange(1, N + 1, dtype=np.int32)
        corrs = rng.uniform(-1, 1, N).astype(np.float32)

        result = batch_similarity(
            query, energy_curves, role_curves, role_dists, ticks, corrs
        )
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_batch_similarity_dtype(self):
        """Result dtype is float32."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        query = compute_signature(frame, max_iters=50)

        N = 3
        energy_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_curves = rng.uniform(0, 1, (N, 10)).astype(np.float32)
        role_dists = rng.uniform(0, 1, (N, 3)).astype(np.float32)
        role_dists = role_dists / role_dists.sum(axis=1, keepdims=True)
        ticks = np.array([10, 20, 30], dtype=np.int32)
        corrs = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = batch_similarity(
            query, energy_curves, role_curves, role_dists, ticks, corrs
        )
        assert result.dtype == np.float32


class TestComputeSignatureBatch:
    """Tests for compute_signature_batch: batch signature extraction."""

    def test_compute_signature_batch_empty_list(self):
        """Empty frame list → empty signature list."""
        result = compute_signature_batch([], max_iters=50)
        assert result == []

    def test_compute_signature_batch_single_frame(self):
        """Single frame → list with one signature."""
        rng = np.random.default_rng(42)
        frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        result = compute_signature_batch([frame], max_iters=50)
        assert len(result) == 1
        assert isinstance(result[0], TrajectorySignature)

    def test_compute_signature_batch_correct_length(self):
        """N frames → list of N signatures."""
        rng = np.random.default_rng(42)
        frames = [rng.uniform(-1, 1, (64, 64)).astype(np.float32) for _ in range(5)]
        result = compute_signature_batch(frames, max_iters=50)
        assert len(result) == 5

    def test_compute_signature_batch_all_valid_signatures(self):
        """All results are valid TrajectorySignature instances."""
        rng = np.random.default_rng(42)
        frames = [rng.uniform(-1, 1, (64, 64)).astype(np.float32) for _ in range(3)]
        result = compute_signature_batch(frames, max_iters=50)
        for sig in result:
            assert isinstance(sig, TrajectorySignature)
            assert sig.energy_curve.shape == (10,)
            assert sig.role_curve.shape == (10,)
            assert sig.final_role_distribution.shape == (3,)

    def test_compute_signature_batch_respects_max_iters(self):
        """All convergence_ticks <= max_iters."""
        rng = np.random.default_rng(42)
        frames = [rng.uniform(-1, 1, (64, 64)).astype(np.float32) for _ in range(3)]
        max_iters = 50
        result = compute_signature_batch(frames, max_iters=max_iters)
        for sig in result:
            assert sig.convergence_ticks <= max_iters

    def test_compute_signature_batch_order_preserved(self):
        """Frame order is preserved in output."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)

        frames = [frame1, frame2]
        sigs = compute_signature_batch(frames, max_iters=50)

        # Re-compute individually to verify order
        sig1_alone = compute_signature(frame1, max_iters=50)
        sig2_alone = compute_signature(frame2, max_iters=50)

        assert np.array_equal(sigs[0].energy_curve, sig1_alone.energy_curve)
        assert np.array_equal(sigs[1].energy_curve, sig2_alone.energy_curve)


class TestIntegration:
    """Integration tests combining multiple trajectory operations."""

    def test_save_load_and_similarity(self, tmp_path):
        """Save signatures, load them, compare similarity."""
        rng = np.random.default_rng(42)
        frame1 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        frame2 = rng.uniform(-1, 1, (64, 64)).astype(np.float32)

        sig1_orig = compute_signature(frame1, max_iters=50)
        sig2_orig = compute_signature(frame2, max_iters=50)

        # Compute similarity before save
        sim_orig = signature_similarity(sig1_orig, sig2_orig)

        # Save and load
        path1 = tmp_path / "sig1.npz"
        path2 = tmp_path / "sig2.npz"
        save_signature(sig1_orig, path1)
        save_signature(sig2_orig, path2)
        sig1_loaded = load_signature(path1)
        sig2_loaded = load_signature(path2)

        # Compute similarity after load
        sim_loaded = signature_similarity(sig1_loaded, sig2_loaded)

        # Should be identical
        assert np.isclose(sim_orig, sim_loaded, atol=1e-4)

    def test_batch_vs_scalar_similarity(self):
        """batch_similarity results match individual signature_similarity calls."""
        rng = np.random.default_rng(42)
        query_frame = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
        query = compute_signature(query_frame, max_iters=50)

        N = 5
        target_frames = [
            rng.uniform(-1, 1, (64, 64)).astype(np.float32) for _ in range(N)
        ]
        targets = [compute_signature(f, max_iters=50) for f in target_frames]

        # Compute batch similarities
        energy_curves = np.stack([t.energy_curve for t in targets])
        role_curves = np.stack([t.role_curve for t in targets])
        role_dists = np.stack([t.final_role_distribution for t in targets])
        ticks = np.array([t.convergence_ticks for t in targets], dtype=np.int32)
        corrs = np.array([t.seed_attractor_corr for t in targets], dtype=np.float32)

        batch_sims = batch_similarity(
            query, energy_curves, role_curves, role_dists, ticks, corrs
        )

        # Compute scalar similarities
        scalar_sims = [signature_similarity(query, t) for t in targets]

        # Should match closely
        for i in range(N):
            assert np.isclose(batch_sims[i], scalar_sims[i], atol=1e-4)

    def test_compute_batch_and_save_all(self, tmp_path):
        """Compute batch signatures and save each one."""
        rng = np.random.default_rng(42)
        frames = [rng.uniform(-1, 1, (64, 64)).astype(np.float32) for _ in range(3)]
        sigs = compute_signature_batch(frames, max_iters=50)

        paths = [tmp_path / f"sig_{i}.npz" for i in range(len(sigs))]
        for sig, path in zip(sigs, paths):
            save_signature(sig, path)

        # Verify all files exist
        for path in paths:
            assert path.exists()

        # Load and verify
        loaded_sigs = [load_signature(path) for path in paths]
        for orig, loaded in zip(sigs, loaded_sigs):
            assert orig.convergence_ticks == loaded.convergence_ticks
            np.testing.assert_array_almost_equal(orig.energy_curve, loaded.energy_curve)

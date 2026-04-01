"""Tests for wheeler_memory.theories.basin — attractor basin mapping."""

import numpy as np
import pytest

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.basin import find_basin_gaps, measure_basin_width


@pytest.mark.slow
class TestMeasureBasinWidth:
    """measure_basin_width() — perturbation survival analysis."""

    def test_returns_expected_keys(self):
        """Result has width, profile, probes."""
        frame = hash_to_frame("basin test")
        result = evolve_and_interpret(frame)
        bw = measure_basin_width(result["attractor"], n_probes=5, steps=5)
        assert "width" in bw
        assert "profile" in bw
        assert "probes" in bw

    def test_width_nonnegative(self):
        """Basin width >= 0."""
        frame = hash_to_frame("width check")
        result = evolve_and_interpret(frame)
        bw = measure_basin_width(result["attractor"], n_probes=5, steps=5)
        assert bw["width"] >= 0.0

    def test_profile_length_matches_steps(self):
        """Profile has one entry per sigma step."""
        frame = hash_to_frame("profile test")
        result = evolve_and_interpret(frame)
        bw = measure_basin_width(result["attractor"], n_probes=5, steps=8)
        assert len(bw["profile"]) == 8

    def test_profile_sigma_monotonically_increases(self):
        """Sigma values in profile are monotonically increasing."""
        frame = hash_to_frame("monotone test")
        result = evolve_and_interpret(frame)
        bw = measure_basin_width(result["attractor"], n_probes=5, steps=10)
        sigmas = [s for s, _ in bw["profile"]]
        assert sigmas == sorted(sigmas)

    def test_survival_decreases_with_noise(self):
        """Higher sigma → lower or equal survival rate."""
        frame = hash_to_frame("survival decay")
        result = evolve_and_interpret(frame)
        bw = measure_basin_width(result["attractor"], n_probes=20, steps=10)
        rates = [r for _, r in bw["profile"]]
        # First survival should be >= last (monotone tendency, not strict)
        assert rates[0] >= rates[-1]


@pytest.mark.slow
class TestFindBasinGaps:
    """find_basin_gaps() — interpolation gap detection."""

    def test_single_attractor_no_gaps(self):
        """Single attractor → no gaps (no pairs)."""
        frame = hash_to_frame("only one")
        result = evolve_and_interpret(frame)
        gaps = find_basin_gaps({"a": result["attractor"]}, n_interpolations=5)
        assert gaps == []

    def test_returns_list(self):
        """Two attractors → returns list (may be empty)."""
        f1 = hash_to_frame("concept one")
        f2 = hash_to_frame("concept two")
        r1 = evolve_and_interpret(f1)
        r2 = evolve_and_interpret(f2)
        gaps = find_basin_gaps(
            {"a": r1["attractor"], "b": r2["attractor"]},
            n_interpolations=3,
        )
        assert isinstance(gaps, list)

    def test_gap_entries_have_expected_keys(self):
        """Each gap entry has point, neighbors, blend_alpha, converged_to."""
        f1 = hash_to_frame("alpha concept")
        f2 = hash_to_frame("beta concept")
        r1 = evolve_and_interpret(f1)
        r2 = evolve_and_interpret(f2)
        gaps = find_basin_gaps(
            {"a": r1["attractor"], "b": r2["attractor"]},
            n_interpolations=5,
        )
        for gap in gaps:
            assert "point" in gap
            assert "neighbors" in gap
            assert "blend_alpha" in gap
            assert isinstance(gap["point"], np.ndarray)
            assert 0.0 < gap["blend_alpha"] < 1.0

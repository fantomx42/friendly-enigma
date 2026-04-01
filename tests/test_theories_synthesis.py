"""Tests for wheeler_memory.theories.synthesis — Apple Test protocol."""

import numpy as np
import pytest

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.synthesis import apple_test, synthesize_from_gap


@pytest.mark.slow
class TestSynthesizeFromGap:
    """synthesize_from_gap() — evolve a gap point to candidate."""

    def test_returns_expected_keys(self):
        """Result has candidate, state, confidence, neighbor_correlations."""
        gap = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        att = evolve_and_interpret(hash_to_frame("neighbor"))["attractor"]
        result = synthesize_from_gap(gap, [att])
        assert "candidate" in result
        assert "state" in result
        assert "confidence" in result
        assert "neighbor_correlations" in result

    def test_candidate_is_64x64(self):
        """Candidate attractor has correct shape."""
        gap = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        result = synthesize_from_gap(gap, [])
        assert result["candidate"].shape == (64, 64)

    def test_state_is_valid(self):
        """State is one of the recognized CA outcomes."""
        gap = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        result = synthesize_from_gap(gap, [])
        assert result["state"] in ("CONVERGED", "OSCILLATING", "CHAOTIC")

    def test_confidence_range(self):
        """Confidence is in [0, 1]."""
        gap = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        att = evolve_and_interpret(hash_to_frame("test"))["attractor"]
        result = synthesize_from_gap(gap, [att])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_no_neighbors_full_confidence(self):
        """No neighbors → confidence = 1.0."""
        gap = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        result = synthesize_from_gap(gap, [])
        assert result["confidence"] == pytest.approx(1.0)


@pytest.mark.slow
class TestAppleTest:
    """apple_test() — full Apple Test protocol."""

    def test_returns_expected_keys(self, tmp_path):
        """Result has verdict, holdout_attractor, candidate, etc."""
        result = apple_test(
            domain_items=["cat", "dog", "fish"],
            holdout="fish",
            data_dir=tmp_path,
        )
        assert "verdict" in result
        assert "holdout_attractor" in result
        assert "candidate" in result
        assert "correlation" in result
        assert "trajectory" in result
        assert "log" in result
        assert "n_gaps" in result

    def test_verdict_is_valid(self, tmp_path):
        """Verdict is one of convergence/dissolution/hallucination."""
        result = apple_test(
            domain_items=["red", "blue", "green", "yellow"],
            holdout="green",
            data_dir=tmp_path,
        )
        assert result["verdict"] in ("convergence", "dissolution", "hallucination")

    def test_holdout_attractor_shape(self, tmp_path):
        """Holdout attractor is 64×64."""
        result = apple_test(
            domain_items=["one", "two", "three"],
            holdout="two",
            data_dir=tmp_path,
        )
        assert result["holdout_attractor"].shape == (64, 64)

    def test_log_is_nonempty(self, tmp_path):
        """Log contains descriptive messages."""
        result = apple_test(
            domain_items=["x", "y", "z"],
            holdout="z",
            data_dir=tmp_path,
        )
        assert len(result["log"]) > 0
        assert any("Storing" in line for line in result["log"])

    def test_cleanup_temp_dir(self):
        """Default (no data_dir) uses temp dir that gets cleaned up."""
        result = apple_test(
            domain_items=["a", "b", "c"],
            holdout="c",
        )
        assert "verdict" in result

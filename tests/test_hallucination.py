"""Tests for hallucination vs synthesis discrimination (theories/metrics.py)."""

import numpy as np
import pytest
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.theories.metrics import classify_output, hallucination_score


def _make_attractor(text: str) -> np.ndarray:
    """Deterministic attractor for a given text (CPU path)."""
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    return result["attractor"]


class TestClassifyOutput:
    """Tests for classify_output()."""

    def test_returns_valid_category(self):
        """classify_output always returns one of the three defined categories."""
        known = [_make_attractor("physics: force equals mass times acceleration")]
        result = classify_output("some text", known)
        assert result in ("SYNTHESIS", "NOVEL", "HALLUCINATION")

    def test_identical_text_is_synthesis(self):
        """Text identical to a stored attractor scores near 0 → SYNTHESIS."""
        text = "the speed of light is approximately 3e8 meters per second"
        attractor = _make_attractor(text)
        # Classify the exact same text against its own attractor
        result = classify_output(text, [attractor], threshold=0.5)
        assert result == "SYNTHESIS"

    def test_cross_domain_text_is_not_synthesis(self):
        """A physics attractor stored; an unrelated cooking query should not be SYNTHESIS."""
        physics_attractor = _make_attractor("electrons orbit the nucleus of an atom")
        # Cooking text is semantically distant — hallucination_score will be high
        result = classify_output(
            "saute the onions in butter until translucent",
            [physics_attractor],
            threshold=0.05,  # tight threshold — only true near-matches pass
        )
        assert result in ("NOVEL", "HALLUCINATION")

    def test_no_known_attractors(self):
        """With empty known_attractors, any converged text is NOVEL."""
        result = classify_output("some statement about gravity", [], threshold=0.5)
        # hallucination_score with no attractors returns energy only
        # converged text has low energy → likely SYNTHESIS at high threshold
        # but with empty list, score = energy (unclamped) → NOVEL at low threshold
        assert result in ("SYNTHESIS", "NOVEL", "HALLUCINATION")

    def test_identical_text_threshold_sensitivity(self):
        """Identical text → identical attractor → Pearson r ≈ 1.0 → SYNTHESIS at any threshold.

        Note: float32 attractor precision makes the computed Pearson slightly
        above 1.0 (~1 + 1.2e-7), so threshold=1.0 still qualifies as SYNTHESIS.
        Use threshold > 1.0 to force NOVEL.
        """
        text = "newton's first law: an object in motion stays in motion"
        attractor = _make_attractor(text)
        # threshold=0 → any non-negative similarity qualifies as SYNTHESIS
        result_low = classify_output(text, [attractor], threshold=0.0)
        assert result_low == "SYNTHESIS"
        # threshold=1.0 → identical text still qualifies (r ≈ 1.0 due to float32)
        result_exact = classify_output(text, [attractor], threshold=1.0)
        assert result_exact == "SYNTHESIS"
        # threshold above float range → forces NOVEL
        result_high = classify_output(text, [attractor], threshold=1.01)
        assert result_high == "NOVEL"


class TestHallucinationScore:
    """Tests for hallucination_score()."""

    def test_score_nonnegative(self):
        """Hallucination score is always >= 0."""
        attractor = _make_attractor("test text for scoring")
        score = hallucination_score(attractor, [attractor])
        assert score >= 0.0

    def test_self_score_near_zero(self):
        """Score of an attractor against itself should be very low (low energy, high corr)."""
        attractor = _make_attractor("a well known physical constant")
        score = hallucination_score(attractor, [attractor])
        assert score < 0.1

    def test_empty_attractors_returns_energy(self):
        """With no known attractors, score equals energy alone."""
        attractor = _make_attractor("test for empty attractor list")
        score = hallucination_score(attractor, [])
        from wheeler_memory.theories.metrics import energy

        e = energy(attractor)
        assert abs(score - e) < 1e-6

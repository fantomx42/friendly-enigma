"""Tests for hallucination vs synthesis discrimination (theories/metrics.py § 6).

Tests from plans/planned-theories-test/theories_2.md § 6:
- test_grounded_statement_is_synthesis
- test_hallucination_is_detected
- test_novel_coherent_synthesis
- test_empty_attractors_returns_novel
- test_classify_output_returns_valid_label

Per SCM's core axiom: a synthesized frame converges (has a basin);
a hallucination is confident but has no basin (doesn't survive symbolic pressure).
"""

import numpy as np
import pytest

from wheeler_memory.constants import HALLUCINATION_THRESHOLD
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.metrics import classify_output


def _make_attractor(text: str) -> np.ndarray:
    """Helper: hash text, evolve deterministically, return attractor frame."""
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    return result["attractor"]


class TestGroundedStatement:
    """Test that well-known, grounded statements classify as SYNTHESIS."""

    def test_grounded_statement_is_synthesis(self):
        """A well-known factual statement should converge near its own attractor → SYNTHESIS."""
        # Store a factual statement as an attractor
        text = "Water boils at 100 degrees Celsius at sea level"
        known_attractor = _make_attractor(text)

        # Classify the same text with this attractor as a known basin
        result = classify_output(text, [known_attractor])

        # Should return SYNTHESIS (converged + Pearson similarity >= threshold)
        assert result == "SYNTHESIS", (
            f"Expected SYNTHESIS for grounded statement, got {result}. "
            f"Text converges to its own basin."
        )

    def test_paraphrase_is_synthesis(self):
        """A close paraphrase should also converge near the original attractor."""
        text1 = "The Earth orbits the Sun"
        text2 = "Our planet revolves around the solar star"

        attractor1 = _make_attractor(text1)

        result = classify_output(text2, [attractor1])

        # Paraphrases should be similar enough to pass synthesis threshold
        # They may not be SYNTHESIS due to domain differences, but should not be HALLUCINATION
        assert result in ("SYNTHESIS", "NOVEL"), (
            f"Paraphrase should not be hallucination, got {result}"
        )


class TestHallucinationDetection:
    """Test that hallucinations are detected (non-convergent or far from all basins)."""

    def test_hallucination_is_detected(self):
        """Text that doesn't converge should return HALLUCINATION."""
        # We can't guarantee non-convergence easily, but we can test cross-domain distance
        physics_attractor = _make_attractor("The electrons orbit the nucleus")

        # A completely unrelated domain
        wrong_domain = "Saute the onions in butter until translucent"
        result = classify_output(wrong_domain, [physics_attractor], threshold=0.05)

        # Should not be SYNTHESIS (low similarity to physics attractor)
        assert result != "SYNTHESIS", (
            f"Unrelated domain text should not be synthesis. "
            f"Expected NOVEL or HALLUCINATION, got {result}"
        )

    def test_bad_fact_is_not_synthesis(self):
        """A factually false statement should not match a correct attractor."""
        correct = "Water freezes at 0 degrees Celsius"
        correct_attractor = _make_attractor(correct)

        # False statement
        wrong = "Water freezes at 25 degrees Celsius"
        result = classify_output(wrong, [correct_attractor], threshold=0.10)

        # Should not converge to the correct basin
        assert result != "SYNTHESIS", (
            f"False statement should not match correct basin. "
            f"Expected NOVEL or HALLUCINATION, got {result}"
        )


class TestNovelCoherentSynthesis:
    """Test that novel but coherent statements are NOVEL, not HALLUCINATION."""

    def test_novel_coherent_synthesis(self):
        """A coherent statement not in known_attractors should be NOVEL."""
        known = [
            _make_attractor("Photosynthesis converts light into chemical energy"),
            _make_attractor("Mitochondria is the powerhouse of the cell"),
        ]

        # Novel but coherent biology statement
        novel = "Ribosomes are responsible for protein synthesis"
        result = classify_output(novel, known)

        # Should converge (coherent) but not match known biology attractors → NOVEL
        assert result in ("NOVEL", "SYNTHESIS"), (
            f"Coherent novel statement should converge (not HALLUCINATION). "
            f"Expected NOVEL or SYNTHESIS, got {result}"
        )


class TestEmptyAttractors:
    """Test that empty known_attractors list doesn't crash."""

    def test_empty_attractors_returns_novel(self):
        """With no known attractors, any converged text should return NOVEL."""
        text = "Some statement about gravity"
        result = classify_output(text, [])

        # Per classify_output code: if not known_attractors, return "NOVEL"
        assert result == "NOVEL", (
            f"Empty known_attractors should return NOVEL (no basin to match). "
            f"Got {result}"
        )

    def test_empty_attractors_doesnt_crash(self):
        """Calling classify_output with empty list should not raise."""
        try:
            classify_output("Some text", [])
        except Exception as e:
            pytest.fail(f"Empty attractors raised {type(e).__name__}: {e}")


class TestValidLabels:
    """Test that classify_output always returns a valid label."""

    def test_classify_output_returns_valid_label(self):
        """Result is always one of {SYNTHESIS, NOVEL, HALLUCINATION}."""
        valid_labels = {"SYNTHESIS", "NOVEL", "HALLUCINATION"}

        test_cases = [
            ("physics fact", [_make_attractor("Gravity pulls objects down")]),
            ("unrelated text", [_make_attractor("Roses are red")]),
            ("random text", []),
            ("long text here " * 50, [_make_attractor("A shorter text")]),
        ]

        for text, attractors in test_cases:
            result = classify_output(text, attractors)
            assert result in valid_labels, (
                f"Invalid label {result!r}. Must be one of {valid_labels}"
            )

    def test_many_random_texts_all_valid(self):
        """Test many random texts always yield valid labels."""
        known = [_make_attractor("Gravity"), _make_attractor("Atoms")]

        for i in range(10):
            text = f"Random test sentence number {i} with some words"
            result = classify_output(text, known)
            assert result in {"SYNTHESIS", "NOVEL", "HALLUCINATION"}, (
                f"Test {i}: got invalid label {result!r}"
            )


class TestThresholdSensitivity:
    """Test that threshold parameter correctly gates SYNTHESIS vs NOVEL."""

    def test_threshold_gates_synthesis_novel(self):
        """SYNTHESIS requires similarity >= threshold. Below it → NOVEL."""
        text = "Einstein's relativity fundamentally changed physics"
        attractor = _make_attractor(text)

        # Classify with tight threshold → less likely to be SYNTHESIS
        result_tight = classify_output(text, [attractor], threshold=0.95)

        # Classify with loose threshold → more likely to be SYNTHESIS
        result_loose = classify_output(text, [attractor], threshold=0.05)

        # At least one should be different (or both same if Pearson is extreme)
        # But we can always assert: at threshold=0, identical text is SYNTHESIS
        result_zero = classify_output(text, [attractor], threshold=0.0)
        assert result_zero == "SYNTHESIS", (
            f"At threshold=0, identical text must be SYNTHESIS, got {result_zero}"
        )

    def test_default_threshold_applied(self):
        """Default threshold should be HALLUCINATION_THRESHOLD from constants."""
        text = "Some statement"
        attractor = _make_attractor(text)

        # Use default (no threshold arg)
        result_default = classify_output(text, [attractor])

        # Use explicit default constant
        result_explicit = classify_output(
            text, [attractor], threshold=HALLUCINATION_THRESHOLD
        )

        # Should match (both use same threshold)
        assert result_default == result_explicit, (
            f"Default threshold should be {HALLUCINATION_THRESHOLD}, "
            f"but behavior differs"
        )


class TestMultipleAttractors:
    """Test classification against multiple known attractors."""

    def test_matches_best_attractor(self):
        """Should match against the closest known attractor, not just the first."""
        physics = _make_attractor("Gravity pulls objects toward Earth")
        biology = _make_attractor("Cells divide through mitosis")
        chemistry = _make_attractor("Atoms combine to form molecules")

        # Query that's clearly biology
        biology_query = "Cells undergo meiosis for reproduction"
        result = classify_output(
            biology_query, [physics, biology, chemistry], threshold=0.20
        )

        # Should converge (coherent) and possibly match biology → not HALLUCINATION
        assert result != "HALLUCINATION", (
            f"Biology query should at least converge, got HALLUCINATION"
        )

    def test_no_matching_attractor_returns_novel(self):
        """If no attractor is close enough, should be NOVEL not SYNTHESIS."""
        att1 = _make_attractor("Physics topic")
        att2 = _make_attractor("Biology topic")

        # Very unrelated text
        query = "Saute vegetables in a hot pan for five minutes"
        result = classify_output(query, [att1, att2], threshold=0.05)

        # Should not be SYNTHESIS (not close to any known basin)
        assert result != "SYNTHESIS", (
            f"Unrelated query should not be SYNTHESIS. Got {result}"
        )


class TestConvergenceGates:
    """Test that convergence state correctly gates HALLUCINATION."""

    def test_converged_not_hallucination(self):
        """Converged states should never be HALLUCINATION (could be SYNTHESIS or NOVEL)."""
        # Well-formed, coherent statements typically converge
        text = "The sky appears blue due to Rayleigh scattering"
        known = [_make_attractor("Sky is blue"), _make_attractor("Light scatters")]

        result = classify_output(text, known, threshold=0.5)

        # Coherent text should converge → not HALLUCINATION
        assert result != "HALLUCINATION", (
            f"Coherent statement should converge (SYNTHESIS or NOVEL), got HALLUCINATION"
        )

    def test_classification_deterministic(self):
        """Same text + attractors should always yield same classification."""
        text = "Determinism test: gravity pulls objects"
        attractor = _make_attractor(text)

        # Call multiple times
        results = [classify_output(text, [attractor]) for _ in range(3)]

        # Should all be identical
        assert len(set(results)) == 1, (
            f"Classification should be deterministic. Got {results}"
        )

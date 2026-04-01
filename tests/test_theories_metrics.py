"""Tests for wheeler_memory.theories.metrics — formal CA metrics."""

import numpy as np
import pytest

from wheeler_memory.dynamics import apply_ca_dynamics, evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.theories.metrics import (
    classify_output,
    context_weight,
    energy,
    hallucination_score,
)


# ── energy ────────────────────────────────────────────────────────────


class TestEnergy:
    """energy() — mean absolute delta after one CA step."""

    def test_attractor_has_low_energy(self):
        """Converged attractor → near-zero energy."""
        frame = hash_to_frame("hello world")
        result = evolve_and_interpret(frame)
        e = energy(result["attractor"])
        assert e < 0.1, f"Attractor energy should be low, got {e}"

    def test_random_frame_has_high_energy(self):
        """Random noise → higher energy than attractor."""
        frame = hash_to_frame("test text")
        result = evolve_and_interpret(frame)
        e_attractor = energy(result["attractor"])
        e_random = energy(np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32))
        assert e_random > e_attractor

    def test_energy_nonnegative(self):
        """Energy is always >= 0."""
        frame = np.zeros((64, 64), dtype=np.float32)
        assert energy(frame) >= 0.0


# ── context_weight ────────────────────────────────────────────────────


class TestContextWeight:
    """context_weight() — hit_factor * basin_width."""

    def test_zero_hits_zero_weight(self):
        """No hits → zero weight."""
        assert context_weight(0, 0.5) == pytest.approx(0.0)

    def test_saturated_hits(self):
        """hit_count >= saturation → weight = basin_width."""
        assert context_weight(10, 0.8, hit_saturation=10) == pytest.approx(0.8)

    def test_partial_hits(self):
        """5 of 10 → factor=0.5 → 0.5*0.4=0.2."""
        assert context_weight(5, 0.4, hit_saturation=10) == pytest.approx(0.2)

    def test_zero_saturation(self):
        """hit_saturation=0 → factor=1.0."""
        assert context_weight(5, 0.6, hit_saturation=0) == pytest.approx(0.6)

    def test_zero_basin_width(self):
        """Zero basin width → zero weight."""
        assert context_weight(10, 0.0) == pytest.approx(0.0)


# ── hallucination_score ───────────────────────────────────────────────


class TestHallucinationScore:
    """hallucination_score() — energy * (1 - max_correlation)."""

    def test_attractor_near_known(self):
        """Attractor compared to itself → near-zero hallucination."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame)
        att = result["attractor"]
        score = hallucination_score(att, [att])
        assert score < 0.01

    def test_no_known_attractors(self):
        """Empty known list → score equals energy."""
        frame = np.random.default_rng(42).uniform(-1, 1, (64, 64)).astype(np.float32)
        score = hallucination_score(frame, [])
        e = energy(frame)
        assert score == pytest.approx(e)

    def test_distant_from_known(self):
        """Random frame far from known attractor → higher score."""
        frame = hash_to_frame("known text")
        result = evolve_and_interpret(frame)
        att = result["attractor"]
        random_frame = np.random.default_rng(99).uniform(-1, 1, (64, 64)).astype(np.float32)
        score_near = hallucination_score(att, [att])
        score_far = hallucination_score(random_frame, [att])
        assert score_far > score_near

    def test_score_nonnegative(self):
        """Hallucination score is always >= 0."""
        frame = np.zeros((64, 64), dtype=np.float32)
        assert hallucination_score(frame, []) >= 0.0


# ── classify_output ───────────────────────────────────────────────────


class TestClassifyOutput:
    """classify_output() — SYNTHESIS, NOVEL, or HALLUCINATION."""

    def test_returns_valid_classification(self):
        """Returns one of the three valid labels."""
        result = classify_output("hello world", [])
        assert result in ("SYNTHESIS", "NOVEL", "HALLUCINATION")

    def test_novel_with_empty_attractors(self):
        """No known attractors → NOVEL (if converged)."""
        result = classify_output("hello world", [])
        assert result in ("NOVEL", "HALLUCINATION")

    def test_synthesis_with_known_attractor(self):
        """Same text as known attractor → SYNTHESIS."""
        frame = hash_to_frame("test concept")
        r = evolve_and_interpret(frame)
        att = r["attractor"]
        result = classify_output("test concept", [att])
        assert result == "SYNTHESIS"

    def test_novel_with_distant_attractor(self):
        """Text far from known attractor → NOVEL."""
        frame = hash_to_frame("quantum electrodynamics")
        r = evolve_and_interpret(frame)
        att = r["attractor"]
        # Completely unrelated text
        result = classify_output("banana bread recipe", [att])
        assert result in ("NOVEL", "HALLUCINATION")

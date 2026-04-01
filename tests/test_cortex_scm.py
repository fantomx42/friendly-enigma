"""Tests for wheeler_memory.cortex_scm — Structural Coherence Measure scoring."""

import numpy as np
import pytest

from wheeler_memory.cortex_scm import (
    SCMResult,
    compute_net_warrant,
    compute_scm,
    score_coevolution_convergence,
    score_coevolution_energy,
    score_coevolution_spread,
    score_energy,
    score_explanation_readiness,
    score_integration,
    score_polarity,
    score_salience,
    score_temperature,
)


# ── score_temperature ─────────────────────────────────────────────────


class TestScoreTemperature:
    """Novelty layer: 1.0 - max(similarities)."""

    def test_empty_returns_one(self):
        """No similarities → maximum novelty."""
        assert score_temperature(np.array([])) == 1.0

    def test_high_similarity_low_temperature(self):
        """High max sim → low temperature."""
        assert score_temperature(np.array([0.9])) == pytest.approx(0.1)

    def test_zero_similarity_full_novelty(self):
        """Zero max sim → full novelty."""
        assert score_temperature(np.array([0.0, 0.0])) == pytest.approx(1.0)

    def test_clipped_to_unit(self):
        """Negative similarity → clipped to 1.0."""
        result = score_temperature(np.array([-0.5]))
        assert 0.0 <= result <= 1.0


# ── score_salience ────────────────────────────────────────────────────


class TestScoreSalience:
    """Retrieval strength: mean(similarities)."""

    def test_empty_returns_zero(self):
        """No similarities → zero salience."""
        assert score_salience(np.array([])) == 0.0

    def test_uniform(self):
        """All 0.5 → salience 0.5."""
        assert score_salience(np.array([0.5, 0.5])) == pytest.approx(0.5)

    def test_mixed(self):
        """Mean of [0.2, 0.8] → 0.5."""
        assert score_salience(np.array([0.2, 0.8])) == pytest.approx(0.5)


# ── score_energy ──────────────────────────────────────────────────────


class TestScoreEnergy:
    """Settlement stability: 1 - max(|delta|)."""

    def test_identical_is_stable(self):
        """No change → energy 1.0."""
        s = np.array([0.5, 0.3])
        assert score_energy(s, s) == pytest.approx(1.0)

    def test_large_change_low_energy(self):
        """Max delta 1.0 → energy 0.0."""
        a = np.array([0.0])
        b = np.array([1.0])
        assert score_energy(a, b) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        """Empty arrays → 0.0."""
        assert score_energy(np.array([]), np.array([])) == 0.0


# ── score_integration ─────────────────────────────────────────────────


class TestScoreIntegration:
    """Within-cluster edge fraction."""

    def test_single_cluster_is_one(self):
        """All in same cluster → 1.0."""
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([0, 0])
        assert score_integration(adj, labels) == pytest.approx(1.0)

    def test_two_clusters_no_cross(self):
        """Block-diagonal adjacency with two clusters → 1.0."""
        adj = np.array([
            [0.0, 0.8, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.7, 0.0],
        ])
        labels = np.array([0, 0, 1, 1])
        assert score_integration(adj, labels) == pytest.approx(1.0)

    def test_all_cross_cluster_is_zero(self):
        """All edges between different clusters → 0.0."""
        adj = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ])
        labels = np.array([0, 0, 1, 1])
        assert score_integration(adj, labels) == pytest.approx(0.0)

    def test_empty_returns_one(self):
        """Empty adjacency → 1.0."""
        assert score_integration(np.array([]), np.array([])) == 1.0

    def test_single_node_returns_one(self):
        """Single node → 1.0."""
        assert score_integration(np.array([[0.0]]), np.array([0])) == 1.0


# ── score_polarity ────────────────────────────────────────────────────


class TestScorePolarity:
    """Contradiction penalty: 1 - |min_negative_weight|."""

    def test_all_positive_is_one(self):
        """No negative edges → 1.0."""
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert score_polarity(adj) == 1.0

    def test_negative_edge_reduces_score(self):
        """Negative edge of -0.3 → polarity 0.7."""
        adj = np.array([[0.0, -0.3], [-0.3, 0.0]])
        assert score_polarity(adj) == pytest.approx(0.7)

    def test_empty_returns_one(self):
        """Empty → 1.0."""
        assert score_polarity(np.array([])) == 1.0


# ── compute_net_warrant ───────────────────────────────────────────────


class TestComputeNetWarrant:
    """Product of five sub-scores."""

    def test_all_ones(self):
        """All 1.0 → warrant 1.0."""
        assert compute_net_warrant(1.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_one_zero_kills_warrant(self):
        """Any zero → warrant 0.0."""
        assert compute_net_warrant(1.0, 0.0, 1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_partial_scores(self):
        """0.5^5 = 0.03125."""
        assert compute_net_warrant(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.03125)


# ── score_explanation_readiness ───────────────────────────────────────


class TestScoreExplanationReadiness:
    """Answer-context alignment."""

    def test_empty_settled_returns_zero(self):
        """No settlement → 0.0."""
        assert score_explanation_readiness(np.array([]), 0.9) == 0.0

    def test_positive_similarity(self):
        """Positive similarity → that value."""
        assert score_explanation_readiness(np.array([1.0]), 0.7) == pytest.approx(0.7)

    def test_negative_similarity_clipped(self):
        """Negative similarity → 0.0."""
        assert score_explanation_readiness(np.array([1.0]), -0.3) == 0.0


# ── coevolution scores ────────────────────────────────────────────────


class TestCoevolutionScores:
    """Coevolution convergence, spread, and energy layers."""

    def test_convergence_fast(self):
        """Instant convergence (tick=0) → 1.0."""
        ticks = np.array([0, 500, 700, 900])
        assert score_coevolution_convergence(ticks) == pytest.approx(1.0)

    def test_convergence_slow(self):
        """Tick=1000 → 0.0."""
        ticks = np.array([1000, 1000, 1000, 1000])
        assert score_coevolution_convergence(ticks) == pytest.approx(0.0)

    def test_convergence_empty(self):
        """Empty ticks → 0.0."""
        assert score_coevolution_convergence(np.array([])) == 0.0

    def test_spread_identical_is_zero(self):
        """All same ticks → no discrimination."""
        assert score_coevolution_spread(np.array([100, 100, 100, 100])) == pytest.approx(0.0)

    def test_spread_varied(self):
        """Wide range → high spread."""
        result = score_coevolution_spread(np.array([100, 200, 300, 1000]))
        assert result > 0.5

    def test_spread_single_element(self):
        """Single element → 0.0."""
        assert score_coevolution_spread(np.array([42])) == 0.0

    def test_energy_empty(self):
        """Empty → 0.0."""
        assert score_coevolution_energy(np.array([])) == 0.0

    def test_energy_high(self):
        """High energy drop → high score."""
        assert score_coevolution_energy(np.array([0.1, 0.3, 0.9, 0.2])) == pytest.approx(0.9)


# ── compute_scm ───────────────────────────────────────────────────────


class TestComputeSCM:
    """Full SCM computation and classification."""

    def _make_inputs(self, sim_val=0.8, stable=True, choice_sim=0.9):
        """Helper to build consistent SCM inputs."""
        sims = np.array([sim_val, sim_val * 0.9, sim_val * 0.8])
        adj = np.array([[0.0, 0.5, 0.3], [0.5, 0.0, 0.4], [0.3, 0.4, 0.0]])
        labels = np.array([0, 0, 0])
        settled = np.array([0.5, 0.4, 0.6])
        prev = settled if stable else settled + 0.9
        return sims, adj, labels, settled, prev, choice_sim

    def test_returns_scm_result(self):
        """Returns SCMResult dataclass."""
        result = compute_scm(*self._make_inputs())
        assert isinstance(result, SCMResult)

    def test_high_coherence_is_synthesis(self):
        """Relaxed thresholds + good inputs → SYNTHESIS."""
        # Default thresholds are hard to reach because temp*salience maxes at ~0.25,
        # so we lower the threshold to test classification logic.
        result = compute_scm(
            *self._make_inputs(sim_val=0.5, stable=True, choice_sim=0.9),
            high_threshold=0.05,
        )
        assert result.mode == "SYNTHESIS"
        assert result.scm >= 0.05

    def test_low_coherence_is_hallucination(self):
        """Low similarity + unstable → HALLUCINATION."""
        result = compute_scm(
            *self._make_inputs(sim_val=0.05, stable=False, choice_sim=0.01)
        )
        assert result.mode == "HALLUCINATION"
        assert result.scm < 0.3

    def test_all_scores_in_unit_range(self):
        """All SCM sub-scores are in [0, 1]."""
        result = compute_scm(*self._make_inputs())
        for field_name in [
            "temperature", "salience", "energy", "integration", "polarity",
            "net_warrant", "explanation_readiness", "scm",
            "coevolution_convergence", "coevolution_spread", "coevolution_energy",
        ]:
            val = getattr(result, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of range"

    def test_custom_thresholds(self):
        """Custom high/low thresholds change classification."""
        result = compute_scm(
            *self._make_inputs(sim_val=0.5, choice_sim=0.5),
            high_threshold=0.99,
            low_threshold=0.98,
        )
        # With extreme thresholds, most inputs become HALLUCINATION
        assert result.mode == "HALLUCINATION"

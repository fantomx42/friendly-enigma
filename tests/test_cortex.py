"""Tests for cortex L1 graph reasoning, L2 settlement, and SCM."""

import numpy as np
import pytest
from wheeler_memory.cortex import (
    compute_adjacency,
    find_clusters,
    find_bridges,
    build_graph,
    settle,
    cortex_reason,
    GraphReasoning,
)
from wheeler_memory.cortex_scm import (
    score_temperature,
    score_salience,
    score_energy,
    score_integration,
    score_polarity,
    compute_net_warrant,
    score_explanation_readiness,
    score_coevolution_convergence,
    score_coevolution_spread,
    score_coevolution_energy,
    compute_scm,
    SCMResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def identical_pair():
    """Two identical attractors — should have correlation ~1.0."""
    a = np.random.default_rng(0).uniform(-1, 1, (4096,)).astype(np.float32)
    return np.stack([a, a.copy()])


@pytest.fixture
def orthogonal_pair():
    """Two unrelated random attractors — correlation ~0.0."""
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (4096,)).astype(np.float32)
    b = rng.uniform(-1, 1, (4096,)).astype(np.float32)
    return np.stack([a, b])


# ---------------------------------------------------------------------------
# L1: compute_adjacency
# ---------------------------------------------------------------------------


class TestComputeAdjacency:
    def test_empty(self):
        adj = compute_adjacency(np.zeros((0, 4096), dtype=np.float32))
        assert adj.shape == (0, 0)

    def test_single(self):
        adj = compute_adjacency(np.ones((1, 4096), dtype=np.float32))
        assert adj.shape == (1, 1)
        assert adj[0, 0] == 0.0

    def test_identical_high_correlation(self, identical_pair):
        adj = compute_adjacency(identical_pair)
        assert adj.shape == (2, 2)
        assert adj[0, 0] == 0.0  # diagonal zero
        assert adj[0, 1] > 0.99  # near-perfect correlation

    def test_random_low_correlation(self, orthogonal_pair):
        adj = compute_adjacency(orthogonal_pair)
        assert abs(adj[0, 1]) < 0.1  # random vectors ~uncorrelated

    def test_symmetric(self, rng):
        atts = rng.uniform(-1, 1, (5, 4096)).astype(np.float32)
        adj = compute_adjacency(atts)
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)

    def test_diagonal_zero(self, rng):
        atts = rng.uniform(-1, 1, (5, 4096)).astype(np.float32)
        adj = compute_adjacency(atts)
        np.testing.assert_array_equal(np.diag(adj), 0.0)


# ---------------------------------------------------------------------------
# L1: find_clusters
# ---------------------------------------------------------------------------


class TestFindClusters:
    def test_empty(self):
        labels = find_clusters(np.zeros((0, 0), dtype=np.float32), 0.5)
        assert len(labels) == 0

    def test_single_node(self):
        labels = find_clusters(np.zeros((1, 1), dtype=np.float32), 0.5)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_two_connected(self):
        adj = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float32)
        labels = find_clusters(adj, 0.5)
        assert labels[0] == labels[1]

    def test_two_disconnected(self):
        adj = np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float32)
        labels = find_clusters(adj, 0.5)
        assert labels[0] != labels[1]

    def test_three_with_two_clusters(self):
        # 0-1 connected, 2 isolated
        adj = np.array(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.0, 0.1],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        labels = find_clusters(adj, 0.5)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]


# ---------------------------------------------------------------------------
# L1: find_bridges
# ---------------------------------------------------------------------------


class TestFindBridges:
    def test_no_bridges_single_cluster(self):
        adj = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=np.float32)
        clusters = np.array([0, 0], dtype=np.int32)
        bridges = find_bridges(adj, clusters, 0.5)
        assert not bridges.any()

    def test_bridge_between_clusters(self):
        # Node 1 connects cluster {0} to cluster {2}
        adj = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.8, 0.0, 0.7],
                [0.1, 0.7, 0.0],
            ],
            dtype=np.float32,
        )
        clusters = np.array([0, 0, 1], dtype=np.int32)
        bridges = find_bridges(adj, clusters, 0.5)
        assert bridges[1]  # node 1 is a bridge


# ---------------------------------------------------------------------------
# L1: build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_returns_graph_reasoning(self, rng):
        atts = rng.uniform(-1, 1, (5, 4096)).astype(np.float32)
        sims = np.array([0.8, 0.6, 0.5, 0.3, 0.1])
        graph = build_graph(atts, sims)
        assert isinstance(graph, GraphReasoning)
        assert graph.adjacency.shape == (5, 5)
        assert len(graph.cluster_labels) == 5
        assert len(graph.bridges) == 5

    def test_empty_graph(self):
        graph = build_graph(
            np.zeros((0, 4096), dtype=np.float32),
            np.array([], dtype=np.float32),
        )
        assert graph.adjacency.shape == (0, 0)
        assert len(graph.contradictions) == 0


# ---------------------------------------------------------------------------
# L2: settle
# ---------------------------------------------------------------------------


class TestSettle:
    def test_empty(self):
        result = settle(np.array([]), np.zeros((0, 0)))
        assert result["converged"]
        assert result["steps"] == 0

    def test_single_node(self):
        result = settle(np.array([0.5]), np.zeros((1, 1)))
        assert result["converged"]
        assert abs(result["settled"][0] - 0.5) < 1e-6

    def test_converges(self):
        opinions = np.array([1.0, 0.0])
        adj = np.array([[0.0, 0.9], [0.9, 0.0]])
        result = settle(opinions, adj)
        assert result["converged"]
        # Should converge toward each other
        assert abs(result["settled"][0] - result["settled"][1]) < 0.01

    def test_inertia_preserves_original(self):
        opinions = np.array([1.0, 0.0])
        adj = np.array([[0.0, 0.9], [0.9, 0.0]])
        # High inertia → slow convergence, values stay closer to original
        result_high = settle(opinions.copy(), adj, inertia=0.99, max_steps=10)
        result_low = settle(opinions.copy(), adj, inertia=0.1, max_steps=10)
        # High inertia should preserve more of the original difference
        diff_high = abs(result_high["settled"][0] - result_high["settled"][1])
        diff_low = abs(result_low["settled"][0] - result_low["settled"][1])
        assert diff_high > diff_low


# ---------------------------------------------------------------------------
# L1+L2: cortex_reason
# ---------------------------------------------------------------------------


class TestCortexReason:
    def test_smoke(self, rng):
        atts = rng.uniform(-1, 1, (5, 4096)).astype(np.float32)
        sims = np.array([0.8, 0.6, 0.5, 0.3, 0.1])
        result = cortex_reason(atts, sims)
        assert "graph" in result
        assert "settlement" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty(self):
        result = cortex_reason(
            np.zeros((0, 4096), dtype=np.float32),
            np.array([], dtype=np.float32),
        )
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# SCM: individual layers
# ---------------------------------------------------------------------------


class TestSCMLayers:
    def test_temperature_empty(self):
        assert score_temperature(np.array([])) == 1.0

    def test_temperature_high_sim(self):
        assert score_temperature(np.array([0.9])) == pytest.approx(0.1)

    def test_salience_empty(self):
        assert score_salience(np.array([])) == 0.0

    def test_salience_mean(self):
        assert score_salience(np.array([0.8, 0.4])) == pytest.approx(0.6)

    def test_energy_converged(self):
        s = np.array([0.5, 0.5])
        assert score_energy(s, s) == 1.0

    def test_energy_diverged(self):
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        assert score_energy(s1, s2) == 0.0

    def test_integration_single_cluster(self):
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([0, 0])
        assert score_integration(adj, labels) == 1.0

    def test_polarity_no_negatives(self):
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert score_polarity(adj) == 1.0

    def test_polarity_with_contradiction(self):
        adj = np.array([[0.0, -0.8], [-0.8, 0.0]])
        assert score_polarity(adj) == pytest.approx(0.2)

    def test_net_warrant_product(self):
        assert compute_net_warrant(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.03125)

    def test_explanation_readiness_clamp(self):
        assert score_explanation_readiness(np.array([1.0]), -0.5) == 0.0
        assert score_explanation_readiness(np.array([1.0]), 0.8) == pytest.approx(0.8)

    def test_coevolution_convergence_empty(self):
        assert score_coevolution_convergence(np.array([])) == 0.0

    def test_coevolution_convergence_fast(self):
        # 10 ticks out of 1000 max → high convergence score
        assert score_coevolution_convergence(
            np.array([10, 100, 200, 500])
        ) == pytest.approx(0.99)

    def test_coevolution_convergence_slow(self):
        # All hit max iterations
        assert score_coevolution_convergence(
            np.array([1000, 1000, 1000, 1000])
        ) == pytest.approx(0.0)

    def test_coevolution_spread_empty(self):
        assert score_coevolution_spread(np.array([])) == 0.0

    def test_coevolution_spread_identical(self):
        assert score_coevolution_spread(np.array([100, 100, 100, 100])) == 0.0

    def test_coevolution_spread_high(self):
        # One fast, rest slow → high spread
        result = score_coevolution_spread(np.array([10, 200, 200, 200]))
        assert result > 0.9

    def test_coevolution_energy_empty(self):
        assert score_coevolution_energy(np.array([])) == 0.0

    def test_coevolution_energy_high(self):
        assert score_coevolution_energy(
            np.array([0.9, 0.3, 0.1, 0.5])
        ) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# SCM: compute_scm
# ---------------------------------------------------------------------------


class TestComputeSCM:
    def test_returns_scm_result(self):
        sims = np.array([0.8, 0.6])
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([0, 0])
        settled = np.array([0.7, 0.7])
        prev = np.array([0.8, 0.6])
        result = compute_scm(sims, adj, labels, settled, prev, 0.5)
        assert isinstance(result, SCMResult)
        assert result.mode in ("SYNTHESIS", "NOVEL", "HALLUCINATION")
        assert 0.0 <= result.scm <= 1.0

    def test_mode_dispatch(self):
        sims = np.array([0.95])
        adj = np.array([[0.0]])
        labels = np.array([0])
        settled = np.array([0.95])
        prev = np.array([0.95])
        # High similarity, converged → high SCM → SYNTHESIS
        result = compute_scm(sims, adj, labels, settled, prev, 0.9, high_threshold=0.01)
        assert result.mode == "SYNTHESIS"

    def test_coevolution_passthrough(self):
        sims = np.array([0.8, 0.6])
        adj = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([0, 0])
        settled = np.array([0.7, 0.7])
        prev = np.array([0.8, 0.6])
        result = compute_scm(
            sims,
            adj,
            labels,
            settled,
            prev,
            0.5,
            coevo_convergence=0.8,
            coevo_spread=0.5,
            coevo_energy=0.9,
        )
        assert result.coevolution_convergence == pytest.approx(0.8)
        assert result.coevolution_spread == pytest.approx(0.5)
        assert result.coevolution_energy == pytest.approx(0.9)

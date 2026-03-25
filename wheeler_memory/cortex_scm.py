"""Structural Coherence Measure (SCM) for Wheeler Memory synthesis evaluation.

Computes seven scoring layers from similarity, graph, and settlement data to
produce a unified coherence score and classification (SYNTHESIS/NOVEL/HALLUCINATION).

Core axiom: meaning is what survives structural pressure. High SCM = stable,
well-integrated answer grounded in retrieved context.
"""

from dataclasses import dataclass

import numpy as np


def score_temperature(similarities: np.ndarray) -> float:
    """Novelty layer: how far from known territory (1.0 - max similarity).

    Args:
        similarities: Retrieved context similarities, shape (K,). Empty → 1.0.

    Returns:
        float in [0, 1]. 1.0 = entirely novel, 0.0 = max similarity to retrieved.
    """
    if len(similarities) == 0:
        return 1.0
    max_sim = float(np.max(similarities))
    temp = 1.0 - max_sim
    return float(np.clip(temp, 0.0, 1.0))


def score_salience(similarities: np.ndarray) -> float:
    """Retrieval strength layer: mean similarity weight.

    Args:
        similarities: Retrieved context similarities, shape (K,). Empty → 0.0.

    Returns:
        float in [0, 1]. High = strong retrieval signal.
    """
    if len(similarities) == 0:
        return 0.0
    sal = float(np.mean(similarities))
    return float(np.clip(sal, 0.0, 1.0))


def score_energy(settled: np.ndarray, prev_settled: np.ndarray) -> float:
    """Settlement stability layer: inverted max delta (1.0 - volatility).

    Args:
        settled: Current settlement, shape (K,).
        prev_settled: Previous settlement, shape (K,).

    Returns:
        float in [0, 1]. 1.0 = stable (no change), 0.0 = max volatility.
    """
    if len(settled) == 0 or len(prev_settled) == 0:
        return 0.0
    delta = np.abs(settled - prev_settled)
    max_delta = float(np.max(delta))
    energy = 1.0 - max_delta
    return float(np.clip(energy, 0.0, 1.0))


def score_integration(
    adjacency: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """Graph coherence layer: within-cluster edge weight fraction.

    Args:
        adjacency: K x K similarity matrix (off-diagonal weights).
        cluster_labels: Cluster assignment per node, shape (K,).

    Returns:
        float in [0, 1]. 1.0 = fully clustered, 0.0 = no within-cluster edges.
        Single cluster (K<=1) → 1.0. Empty → 1.0.
    """
    if len(adjacency) == 0 or len(cluster_labels) == 0:
        return 1.0
    if len(cluster_labels) <= 1:
        return 1.0

    # Mask off-diagonal
    mask = np.eye(len(adjacency), dtype=bool)
    masked = adjacency.copy()
    masked[mask] = 0.0

    total_weight = float(np.sum(np.abs(masked)))
    if total_weight < 1e-10:
        return 1.0

    # Within-cluster edges
    within = 0.0
    for i in range(len(adjacency)):
        for j in range(len(adjacency)):
            if i != j and cluster_labels[i] == cluster_labels[j]:
                within += masked[i, j]

    within_weight = float(within)
    integ = within_weight / total_weight if total_weight > 0 else 1.0
    return float(np.clip(integ, 0.0, 1.0))


def score_polarity(adjacency: np.ndarray) -> float:
    """Contradiction penalty layer: inverted negative weights (1.0 - harm).

    Args:
        adjacency: K x K similarity matrix (off-diagonal edges only).

    Returns:
        float in [0, 1]. 1.0 = no contradictions, 0.0 = max negative weight.
    """
    if len(adjacency) == 0:
        return 1.0

    # Mask off-diagonal
    mask = np.eye(len(adjacency), dtype=bool)
    masked = adjacency.copy()
    masked[mask] = 0.0

    min_weight = float(np.min(masked))
    if min_weight >= 0.0:
        return 1.0

    # Penalty: how far below zero
    penalty = max(0.0, -min_weight)
    polar = 1.0 - penalty
    return float(np.clip(polar, 0.0, 1.0))


def compute_net_warrant(
    temperature: float,
    salience: float,
    energy: float,
    integration: float,
    polarity: float,
) -> float:
    """Unified coherence warrant: product of all sub-scores.

    Args:
        temperature, salience, energy, integration, polarity: Each in [0, 1].

    Returns:
        float in [0, 1]. Product of inputs.
    """
    warrant = temperature * salience * energy * integration * polarity
    return float(np.clip(warrant, 0.0, 1.0))


def score_explanation_readiness(
    settled: np.ndarray,
    choice_similarity: float,
) -> float:
    """Answer-context alignment layer: how well chosen answer matches retrieval.

    Args:
        settled: Settlement vector (used for presence check), shape (K,).
        choice_similarity: Similarity of chosen answer to best-retrieved context.

    Returns:
        float in [0, 1]. High = chosen answer aligns with retrieval signal.
    """
    if len(settled) == 0:
        return 0.0
    explain = max(0.0, choice_similarity)
    return float(np.clip(explain, 0.0, 1.0))


def score_coevolution_convergence(
    choice_ticks: np.ndarray, max_iters: int = 1000
) -> float:
    """Co-evolution convergence layer: how fast the best Q+A combination settled.

    Args:
        choice_ticks: (4,) convergence ticks for each choice co-evolution.
        max_iters: Maximum possible iterations.

    Returns:
        float in [0, 1]. 1.0 = instant convergence, 0.0 = hit max iterations.
    """
    if len(choice_ticks) == 0:
        return 0.0
    best = float(np.min(choice_ticks))
    return float(np.clip(1.0 - best / max_iters, 0.0, 1.0))


def score_coevolution_spread(choice_ticks: np.ndarray) -> float:
    """Co-evolution discrimination layer: how much dynamics distinguish between choices.

    High spread = CA dynamics clearly prefer one Q+A combination.
    Low spread = all combinations equally (in)coherent.

    Args:
        choice_ticks: (4,) convergence ticks for each choice co-evolution.

    Returns:
        float in [0, 1]. 1.0 = maximum discrimination, 0.0 = all identical.
    """
    if len(choice_ticks) < 2:
        return 0.0
    ticks = choice_ticks.astype(np.float64)
    spread = float(np.max(ticks) - np.min(ticks))
    denom = float(np.max(ticks))
    if denom < 1.0:
        return 0.0
    return float(np.clip(spread / denom, 0.0, 1.0))


def score_coevolution_energy(choice_energies: np.ndarray) -> float:
    """Co-evolution energy layer: steepest energy drop among the 4 choices.

    Measures how much the best Q+A combination's CA dynamics reduced energy
    (mean absolute delta) from start to end of evolution.

    Args:
        choice_energies: (4,) total energy drop for each choice (initial - final delta).

    Returns:
        float in [0, 1]. 1.0 = maximum energy dissipation, 0.0 = no energy change.
    """
    if len(choice_energies) == 0:
        return 0.0
    best = float(np.max(choice_energies))
    return float(np.clip(best, 0.0, 1.0))


@dataclass
class SCMResult:
    """Structural Coherence Measure result with all layers and final score."""

    temperature: float
    salience: float
    energy: float
    integration: float
    polarity: float
    net_warrant: float
    explanation_readiness: float
    coevolution_convergence: float
    coevolution_spread: float
    coevolution_energy: float
    scm: float
    mode: str


def compute_scm(
    similarities: np.ndarray,
    adjacency: np.ndarray,
    cluster_labels: np.ndarray,
    settled: np.ndarray,
    prev_settled: np.ndarray,
    choice_similarity: float,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
    coevo_convergence: float = 0.0,
    coevo_spread: float = 0.0,
    coevo_energy: float = 0.0,
) -> SCMResult:
    """Compute structural coherence measure and classification.

    Args:
        similarities: Retrieved context similarities, shape (K,).
        adjacency: K x K graph weight matrix (can include contradictions < 0).
        cluster_labels: Cluster assignment per context, shape (K,).
        settled: Current settlement (evidence weights), shape (K,).
        prev_settled: Previous settlement, shape (K,).
        choice_similarity: Chosen answer similarity to best retrieval.
        high_threshold: SCM threshold for SYNTHESIS (default 0.7).
        low_threshold: SCM threshold for NOVEL (default 0.3).
        coevo_convergence: Co-evolution convergence score (default 0.0).
        coevo_spread: Co-evolution spread score (default 0.0).
        coevo_energy: Co-evolution energy score (default 0.0).

    Returns:
        SCMResult with all ten layer scores, net warrant, and classification.
    """
    # Compute each layer
    temp = score_temperature(similarities)
    sal = score_salience(similarities)
    energ = score_energy(settled, prev_settled)
    integ = score_integration(adjacency, cluster_labels)
    polar = score_polarity(adjacency)

    # Unified warrant
    warrant = compute_net_warrant(temp, sal, energ, integ, polar)

    # Answer readiness
    readiness = score_explanation_readiness(settled, choice_similarity)

    # Final SCM
    scm_score = warrant * readiness
    scm_score = float(np.clip(scm_score, 0.0, 1.0))

    # Classification
    if scm_score >= high_threshold:
        mode = "SYNTHESIS"
    elif scm_score >= low_threshold:
        mode = "NOVEL"
    else:
        mode = "HALLUCINATION"

    return SCMResult(
        temperature=float(np.clip(temp, 0.0, 1.0)),
        salience=float(np.clip(sal, 0.0, 1.0)),
        energy=float(np.clip(energ, 0.0, 1.0)),
        integration=float(np.clip(integ, 0.0, 1.0)),
        polarity=float(np.clip(polar, 0.0, 1.0)),
        net_warrant=float(np.clip(warrant, 0.0, 1.0)),
        explanation_readiness=float(np.clip(readiness, 0.0, 1.0)),
        coevolution_convergence=float(np.clip(coevo_convergence, 0.0, 1.0)),
        coevolution_spread=float(np.clip(coevo_spread, 0.0, 1.0)),
        coevolution_energy=float(np.clip(coevo_energy, 0.0, 1.0)),
        scm=scm_score,
        mode=mode,
    )

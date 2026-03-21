"""L1 Graph Reasoning and L2 Settlement Cellular Automata for Wheeler Memory.

Implements attractor correlation graphs and opinion settlement via diffusion
on the correlation adjacency matrix. Used for multi-hop reasoning and
confidence aggregation across retrieved memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class GraphReasoning:
    """Attractor correlation graph with cluster topology and contradictions.

    Attributes:
        adjacency: (K, K) Pearson correlation matrix, diagonal = 0.
        cluster_labels: (K,) int cluster IDs from connected component analysis.
        bridges: (K,) bool array — True if node connects 2+ clusters.
        contradictions: List of (i, j) tuples with strong negative correlation.
        similarities: (K,) original retrieval similarities.
    """

    adjacency: np.ndarray
    cluster_labels: np.ndarray
    bridges: np.ndarray
    contradictions: list[tuple[int, int]]
    similarities: np.ndarray


def compute_adjacency(attractors: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix from attractor embeddings.

    Vectorized computation with no loops. Handles edge cases:
    - K=0: returns (0, 0) array
    - K=1: returns [[0.0]]
    - All-zero attractors: returns zeros with 0 on diagonal

    Args:
        attractors: (K, D) array of attractor embeddings.

    Returns:
        (K, K) correlation matrix with 0 on diagonal.
    """
    K = attractors.shape[0]
    if K == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if K == 1:
        return np.zeros((1, 1), dtype=np.float32)

    # Center each row
    centered = attractors - attractors.mean(axis=1, keepdims=True)

    # Normalize by row norm
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)  # avoid division by zero
    normalized = centered / norms

    # Correlation: normalized @ normalized.T
    adj = normalized @ normalized.T

    # Zero out diagonal (no self-edges)
    np.fill_diagonal(adj, 0.0)

    return adj.astype(np.float32)


def find_clusters(adj: np.ndarray, threshold: float) -> np.ndarray:
    """Find connected components via BFS on thresholded adjacency matrix.

    Pure numpy/Python implementation. A node is reachable from another
    if adj[i, j] > threshold.

    Args:
        adj: (K, K) adjacency matrix.
        threshold: Correlation threshold for connectivity.

    Returns:
        (K,) int array where each element is its cluster ID (0-indexed).
    """
    K = adj.shape[0]
    if K == 0:
        return np.array([], dtype=np.int32)
    if K == 1:
        return np.array([0], dtype=np.int32)

    labels = np.full(K, -1, dtype=np.int32)
    cluster_id = 0

    for start in range(K):
        if labels[start] != -1:
            continue

        # BFS from start node
        queue = [start]
        labels[start] = cluster_id

        while queue:
            u = queue.pop(0)
            # Find all neighbors v where adj[u, v] > threshold
            neighbors = np.where(adj[u] > threshold)[0]
            for v in neighbors:
                if labels[v] == -1:
                    labels[v] = cluster_id
                    queue.append(v)

        cluster_id += 1

    return labels


def find_bridges(adj: np.ndarray, clusters: np.ndarray, threshold: float) -> np.ndarray:
    """Identify bridge nodes that connect 2+ clusters.

    A node is a bridge if it has neighbors (adj > threshold) in 2+ clusters.

    Args:
        adj: (K, K) adjacency matrix.
        clusters: (K,) cluster IDs from find_clusters().
        threshold: Correlation threshold for connectivity.

    Returns:
        (K,) bool array, True if node is a bridge.
    """
    K = adj.shape[0]
    if K <= 1:
        return np.zeros(K, dtype=bool)

    bridges = np.zeros(K, dtype=bool)

    for i in range(K):
        # Find all neighbors of i
        neighbors = np.where(adj[i] > threshold)[0]
        if len(neighbors) == 0:
            continue

        # Count unique clusters among neighbors
        neighbor_clusters = set(clusters[neighbors].tolist())
        if len(neighbor_clusters) >= 2:
            bridges[i] = True

    return bridges


def build_graph(
    attractors: np.ndarray,
    similarities: np.ndarray,
    cluster_threshold: float = 0.5,
    contradiction_threshold: float = 0.3,
) -> GraphReasoning:
    """Build full correlation graph with clusters and contradictions.

    Orchestrates compute_adjacency, find_clusters, find_bridges, and
    identifies contradiction pairs.

    Args:
        attractors: (K, D) attractor embeddings.
        similarities: (K,) retrieval similarities.
        cluster_threshold: Correlation threshold for clustering.
        contradiction_threshold: Negative correlation threshold.

    Returns:
        GraphReasoning object with full topology.
    """
    K = attractors.shape[0]

    adj = compute_adjacency(attractors)
    clusters = find_clusters(adj, cluster_threshold)
    bridges = find_bridges(adj, clusters, cluster_threshold)

    # Find contradictions: pairs with strong negative correlation
    contradictions: list[tuple[int, int]] = []
    if K > 1:
        neg_adj = adj < -contradiction_threshold
        # Get upper triangle to avoid duplicates
        for i in range(K):
            for j in range(i + 1, K):
                if neg_adj[i, j]:
                    contradictions.append((int(i), int(j)))

    return GraphReasoning(
        adjacency=adj,
        cluster_labels=clusters,
        bridges=bridges,
        contradictions=contradictions,
        similarities=similarities,
    )


def settle(
    opinions: np.ndarray,
    adjacency: np.ndarray,
    max_steps: int = 100,
    threshold: float = 1e-4,
    inertia: float = 0.8,
) -> dict[str, int | bool | np.ndarray]:
    """Settlement CA: diffuse opinions via correlation-weighted averaging.

    Iteratively averages opinions over correlation-weighted neighbors,
    with inertia to preserve original values. Uses absolute values of
    correlation weights to allow both positive and negative edges to
    carry information.

    Convergence: when max absolute change < threshold.

    Args:
        opinions: (K,) float initial values (retrieval similarities).
        adjacency: (K, K) correlation weights.
        max_steps: Maximum iteration count.
        threshold: Convergence threshold for max change.
        inertia: Weight of original value (0.0 to 1.0).

    Returns:
        Dict with keys:
        - "settled": (K,) settled opinion values
        - "steps": number of iterations until convergence or max_steps
        - "converged": True if threshold reached before max_steps
    """
    K = opinions.shape[0]
    if K == 0:
        return {"settled": opinions, "steps": 0, "converged": True}
    if K == 1:
        return {"settled": opinions, "steps": 0, "converged": True}

    current = opinions.astype(np.float32).copy()
    weights = np.abs(adjacency)  # Use absolute values for diffusion

    for step in range(max_steps):
        old = current.copy()

        # Degree (sum of outgoing edge weights)
        degree = weights.sum(axis=1) + 1e-10

        # Neighbor mean = weighted sum / degree
        neighbor_mean = (weights @ current) / degree

        # Update with inertia: keep original value proportion, move toward neighbors
        current = inertia * current + (1.0 - inertia) * neighbor_mean

        # Check convergence
        max_change = np.max(np.abs(current - old))
        if max_change < threshold:
            return {
                "settled": current,
                "steps": step + 1,
                "converged": True,
            }

    return {
        "settled": current,
        "steps": max_steps,
        "converged": False,
    }


def cortex_reason(
    attractors: np.ndarray,
    similarities: np.ndarray,
    **kwargs: float | int,
) -> dict[str, GraphReasoning | dict | float]:
    """Full pipeline: build graph and settle opinions via diffusion.

    Integrates graph reasoning (L1) with settlement dynamics (L2).

    Args:
        attractors: (K, D) attractor embeddings.
        similarities: (K,) retrieval similarities.
        **kwargs: Passed to build_graph and settle:
            - cluster_threshold, contradiction_threshold → build_graph
            - max_steps, threshold, inertia → settle

    Returns:
        Dict with keys:
        - "graph": GraphReasoning object
        - "settlement": Dict from settle() with settled opinions
        - "confidence": float, mean of settled opinions (all K)
    """
    K = attractors.shape[0]

    # Extract parameters
    cluster_threshold = kwargs.get("cluster_threshold", 0.5)
    contradiction_threshold = kwargs.get("contradiction_threshold", 0.3)
    max_steps = kwargs.get("max_steps", 100)
    settle_threshold = kwargs.get("threshold", 1e-4)
    inertia = kwargs.get("inertia", 0.8)

    # Build graph
    graph = build_graph(
        attractors,
        similarities,
        cluster_threshold=cluster_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    # Settle opinions
    settlement = settle(
        similarities,
        graph.adjacency,
        max_steps=max_steps,
        threshold=settle_threshold,
        inertia=inertia,
    )

    # Compute confidence as mean of settled opinions
    settled = settlement["settled"]
    confidence = float(settled.mean()) if K > 0 else 0.0

    return {
        "graph": graph,
        "settlement": settlement,
        "confidence": confidence,
    }

"""Computable Formal Metrics for Wheeler Memory attractor topology.

Implements energy, basin width, context weight, hallucination score,
and topology consistency — all computed from CA dynamics alone.
"""

import numpy as np
from scipy.stats import pearsonr

from ..constants import HALLUCINATION_THRESHOLD
from ..dynamics import apply_ca_dynamics, evolve_and_interpret
from ..hashing import hash_to_frame
from .basin import measure_basin_width


def energy(state: np.ndarray) -> float:
    """Energy = mean absolute delta after one CA step.

    Low energy = near attractor. High energy = far from any basin.
    """
    next_state = apply_ca_dynamics(state)
    return float(np.abs(next_state - state).mean())


def basin_width(attractor: np.ndarray, **kwargs) -> float:
    """Delegates to basin.measure_basin_width(), returns just the width scalar."""
    result = measure_basin_width(attractor, **kwargs)
    return result["width"]


def context_weight(
    hit_count: int,
    b_width: float,
    hit_saturation: int = 10,
) -> float:
    """Combined weight: min(1, hit_count/saturation) * b_width, normalized."""
    hit_factor = min(1.0, hit_count / hit_saturation) if hit_saturation > 0 else 1.0
    return hit_factor * b_width


def hallucination_score(
    frame: np.ndarray,
    known_attractors: list[np.ndarray],
) -> float:
    """How much a frame is drifting without basin pull.

    Score = energy(frame) * (1 - max_correlation_with_any_attractor).
    High score = high energy AND far from all known basins = hallucination.
    """
    e = energy(frame)
    if not known_attractors:
        return e

    frame_flat = frame.flatten()
    max_corr = max(
        pearsonr(frame_flat, att.flatten())[0] for att in known_attractors
    )
    # Clamp correlation to [0, 1] for score computation
    max_corr = max(0.0, min(1.0, max_corr))
    return e * (1.0 - max_corr)


def classify_output(
    text: str,
    known_attractors: list[np.ndarray],
    threshold: float = HALLUCINATION_THRESHOLD,
) -> str:
    """Classify text as SYNTHESIS, NOVEL, or HALLUCINATION.

    SYNTHESIS     — CA converged AND Pearson similarity >= threshold (grounded, near a known basin)
    NOVEL         — CA converged AND Pearson similarity < threshold (stable but unknown territory)
    HALLUCINATION — CA did not converge (no basin = confident but groundless)

    This makes SCM's core axiom executable: meaning is what survives symbolic pressure.
    A hallucination has no basin; it doesn't survive CA evolution.

    Note: uses Pearson similarity for SYNTHESIS/NOVEL (not hallucination_score, which
    is energy-weighted and collapses to ~0 for any fully-converged attractor).
    """
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    if result["state"] != "CONVERGED":
        return "HALLUCINATION"
    if not known_attractors:
        return "NOVEL"
    flat = result["attractor"].flatten()
    flat_mean, flat_std = flat.mean(), flat.std()
    if flat_std < 1e-10:
        return "NOVEL"
    max_sim = max(
        float(
            np.dot(flat - flat_mean, att.flatten() - att.flatten().mean())
            / (len(flat) * flat_std * (att.flatten().std() + 1e-10))
        )
        for att in known_attractors
    )
    return "SYNTHESIS" if max_sim >= threshold else "NOVEL"


def topology_consistency(
    attractors: list[np.ndarray],
    n_samples: int = 100,
) -> float:
    """Sample random points in [-1,1]^(64x64), evolve each, check convergence.

    Consistency = fraction that land on a known attractor (correlation > 0.9).
    1.0 = basins tile space perfectly. <1.0 = gaps exist.
    """
    if not attractors:
        return 0.0

    flat_attractors = [att.flatten() for att in attractors]
    size = attractors[0].shape
    landed = 0

    for _ in range(n_samples):
        random_frame = np.random.uniform(-1, 1, size=size).astype(np.float32)
        result = evolve_and_interpret(random_frame)
        result_flat = result["attractor"].flatten()

        for att_flat in flat_attractors:
            corr, _ = pearsonr(result_flat, att_flat)
            if corr > 0.9:
                landed += 1
                break

    return landed / n_samples

"""Trajectory signature extraction and similarity for Wheeler Memory.

Every CA evolution traces a path from seed frame to attractor. The trajectory
— how fast it converges, what roles emerge, how energy decays — carries
structural information that endpoint matching alone misses.

Two texts about the same concept may encode to different seed frames (different
n-grams) but evolve with similar dynamics: similar convergence speed, similar
energy decay, similar role distributions. This module captures that structural
similarity as a compact ~40-float fingerprint per memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    CONVERGENCE_PERCENTILE,
    SALIENCE_MAX_ITERS_MED,
    SALIENCE_THRESHOLD_MED,
    STABILITY_WINDOW,
)
from .dynamics import apply_ca_dynamics
from .oscillation import get_cell_roles


# ---------------------------------------------------------------------------
# Trajectory Signature dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySignature:
    """Compact fingerprint of a CA evolution trajectory (~40 floats)."""

    convergence_ticks: int  # ticks until settled
    energy_curve: np.ndarray  # (10,) downsampled mean-abs-delta per tick
    role_curve: np.ndarray  # (10,) downsampled fraction of slope cells per tick
    final_role_distribution: np.ndarray  # (3,) fraction of [max, min, slope] cells
    state: str  # CONVERGED | OSCILLATING | CHAOTIC
    seed_attractor_corr: float  # Pearson(seed, attractor) — how much it changed


# ---------------------------------------------------------------------------
# Signature computation
# ---------------------------------------------------------------------------


def _downsample(curve: list[float], target_len: int = 10) -> np.ndarray:
    """Downsample a variable-length curve to fixed length via linear interp."""
    n = len(curve)
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)
    if n == 1:
        return np.full(target_len, curve[0], dtype=np.float32)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, curve).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flat arrays."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    a_std = a_c.std()
    b_std = b_c.std()
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0
    return float(np.dot(a_c, b_c) / (len(a) * a_std * b_std))


def compute_signature(
    seed_frame: np.ndarray,
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> TrajectorySignature:
    """Re-evolve a seed frame on CPU and extract trajectory signature.

    Always uses CPU path (not GPU) to get full per-tick history.

    Args:
        seed_frame: (64, 64) initial frame.
        max_iters: Maximum CA iterations.
        stability_threshold: p99 abs delta below which we declare convergence.

    Returns:
        TrajectorySignature with downsampled curves and metadata.
    """
    frame = seed_frame.copy()
    total_cells = frame.size

    energy_raw: list[float] = []
    role_raw: list[float] = []
    state = "CHAOTIC"
    ticks = 0
    stable_count = 0

    for i in range(max_iters):
        frame_old = frame
        frame = apply_ca_dynamics(frame)
        abs_deltas = np.abs(frame - frame_old)
        delta = float(np.percentile(abs_deltas, CONVERGENCE_PERCENTILE))
        energy_raw.append(delta)

        # Role stats: fraction of slope cells (neither max nor min)
        roles = get_cell_roles(frame)
        n_slope = int(np.sum(roles == 0))
        role_raw.append(n_slope / total_cells)

        ticks = i + 1

        if delta < stability_threshold:
            stable_count += 1
            if stable_count >= STABILITY_WINDOW:
                state = "CONVERGED"
                break
        else:
            stable_count = 0

        # Simple oscillation check: if we're past 50 ticks and delta
        # is bouncing in a narrow band, call it oscillating
        if i > 50 and len(energy_raw) >= 20:
            recent = energy_raw[-20:]
            if max(recent) - min(recent) < stability_threshold * 10:
                state = "OSCILLATING"
                break

    # Final role distribution: [max_frac, min_frac, slope_frac]
    final_roles = get_cell_roles(frame)
    n_max = int(np.sum(final_roles == 1))
    n_min = int(np.sum(final_roles == -1))
    n_slope = int(np.sum(final_roles == 0))
    role_dist = np.array(
        [n_max / total_cells, n_min / total_cells, n_slope / total_cells],
        dtype=np.float32,
    )

    # Pearson between seed and final attractor (how much it changed)
    seed_flat = seed_frame.flatten().astype(np.float32)
    att_flat = frame.flatten().astype(np.float32)
    corr = _pearson(seed_flat, att_flat)

    return TrajectorySignature(
        convergence_ticks=ticks,
        energy_curve=_downsample(energy_raw),
        role_curve=_downsample(role_raw),
        final_role_distribution=role_dist,
        state=state,
        seed_attractor_corr=corr,
    )


def compute_signature_batch(
    seed_frames: list[np.ndarray],
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> list[TrajectorySignature]:
    """Compute trajectory signatures for a batch of seed frames."""
    return [
        compute_signature(
            f, max_iters=max_iters, stability_threshold=stability_threshold
        )
        for f in seed_frames
    ]


# ---------------------------------------------------------------------------
# Signature similarity
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def signature_similarity(a: TrajectorySignature, b: TrajectorySignature) -> float:
    """Compute similarity between two trajectory signatures.

    Weighted combination of five components:
      - Pearson on energy_curve       (0.3)
      - Pearson on role_curve          (0.3)
      - Convergence speed similarity   (0.2)
      - Cosine on final role dist      (0.1)
      - Seed-attractor corr closeness  (0.1)

    Returns:
        float in [0, 1]. Higher = more similar trajectories.
    """
    # Energy curve similarity
    e_sim = _pearson(a.energy_curve, b.energy_curve)
    e_sim = max(0.0, e_sim)  # clamp negative correlations to 0

    # Role curve similarity
    r_sim = _pearson(a.role_curve, b.role_curve)
    r_sim = max(0.0, r_sim)

    # Convergence speed similarity
    max_ticks = max(a.convergence_ticks, b.convergence_ticks, 1)
    tick_sim = 1.0 - abs(a.convergence_ticks - b.convergence_ticks) / max_ticks

    # Final role distribution similarity (cosine)
    dist_sim = _cosine_sim(a.final_role_distribution, b.final_role_distribution)
    dist_sim = max(0.0, dist_sim)

    # Seed-attractor correlation closeness
    corr_sim = 1.0 - abs(a.seed_attractor_corr - b.seed_attractor_corr)
    corr_sim = max(0.0, corr_sim)

    return 0.3 * e_sim + 0.3 * r_sim + 0.2 * tick_sim + 0.1 * dist_sim + 0.1 * corr_sim


# ---------------------------------------------------------------------------
# Signature I/O
# ---------------------------------------------------------------------------


def save_signature(sig: TrajectorySignature, filepath: str | Path) -> None:
    """Save a trajectory signature to an .npz file."""
    np.savez_compressed(
        filepath,
        energy_curve=sig.energy_curve,
        role_curve=sig.role_curve,
        role_dist=sig.final_role_distribution,
        convergence_ticks=np.array(sig.convergence_ticks),
        state=np.array(str(sig.state)),
        seed_att_corr=np.array(sig.seed_attractor_corr),
    )


def load_signature(filepath: str | Path) -> TrajectorySignature:
    """Load a trajectory signature from an .npz file."""
    data = np.load(filepath, allow_pickle=False)
    return TrajectorySignature(
        convergence_ticks=int(data["convergence_ticks"]),
        energy_curve=data["energy_curve"].astype(np.float32),
        role_curve=data["role_curve"].astype(np.float32),
        final_role_distribution=data["role_dist"].astype(np.float32),
        state=str(data["state"]),
        seed_attractor_corr=float(data["seed_att_corr"]),
    )


# ---------------------------------------------------------------------------
# Vectorized batch similarity (for TrajectoryCache)
# ---------------------------------------------------------------------------


def batch_similarity(
    query: TrajectorySignature,
    energy_curves: np.ndarray,
    role_curves: np.ndarray,
    role_dists: np.ndarray,
    ticks: np.ndarray,
    seed_att_corrs: np.ndarray,
) -> np.ndarray:
    """Compute similarity of a query signature against N stored signatures.

    All array params have first dimension N (number of stored signatures).

    Args:
        query: Query trajectory signature.
        energy_curves: (N, 10) energy curves.
        role_curves: (N, 10) role curves.
        role_dists: (N, 3) final role distributions.
        ticks: (N,) convergence ticks.
        seed_att_corrs: (N,) seed-attractor correlations.

    Returns:
        (N,) array of similarities in [0, 1].
    """
    N = energy_curves.shape[0]
    if N == 0:
        return np.array([], dtype=np.float32)

    # Vectorized Pearson on energy curves: (N, 10) vs (10,)
    qe = query.energy_curve.astype(np.float64)
    qe_c = qe - qe.mean()
    qe_std = qe_c.std()

    ec = energy_curves.astype(np.float64)
    ec_c = ec - ec.mean(axis=1, keepdims=True)
    ec_std = ec_c.std(axis=1)

    if qe_std > 1e-10:
        e_dots = ec_c @ qe_c
        valid = ec_std > 1e-10
        e_sims = np.where(valid, e_dots / (10 * ec_std * qe_std), 0.0)
    else:
        e_sims = np.zeros(N, dtype=np.float64)
    e_sims = np.clip(e_sims, 0.0, 1.0)

    # Vectorized Pearson on role curves
    qr = query.role_curve.astype(np.float64)
    qr_c = qr - qr.mean()
    qr_std = qr_c.std()

    rc = role_curves.astype(np.float64)
    rc_c = rc - rc.mean(axis=1, keepdims=True)
    rc_std = rc_c.std(axis=1)

    if qr_std > 1e-10:
        r_dots = rc_c @ qr_c
        valid = rc_std > 1e-10
        r_sims = np.where(valid, r_dots / (10 * rc_std * qr_std), 0.0)
    else:
        r_sims = np.zeros(N, dtype=np.float64)
    r_sims = np.clip(r_sims, 0.0, 1.0)

    # Convergence speed similarity
    max_ticks = np.maximum(ticks, query.convergence_ticks).astype(np.float64)
    max_ticks = np.maximum(max_ticks, 1.0)
    tick_sims = (
        1.0 - np.abs(ticks.astype(np.float64) - query.convergence_ticks) / max_ticks
    )

    # Cosine on role distributions
    qd = query.final_role_distribution.astype(np.float64)
    qd_norm = np.linalg.norm(qd)
    rd = role_dists.astype(np.float64)
    rd_norms = np.linalg.norm(rd, axis=1)

    if qd_norm > 1e-10:
        d_dots = rd @ qd
        valid = rd_norms > 1e-10
        dist_sims = np.where(valid, d_dots / (rd_norms * qd_norm), 0.0)
    else:
        dist_sims = np.zeros(N, dtype=np.float64)
    dist_sims = np.clip(dist_sims, 0.0, 1.0)

    # Seed-attractor correlation closeness
    corr_sims = 1.0 - np.abs(
        seed_att_corrs.astype(np.float64) - query.seed_attractor_corr
    )
    corr_sims = np.clip(corr_sims, 0.0, 1.0)

    # Weighted combination
    combined = (
        0.3 * e_sims
        + 0.3 * r_sims
        + 0.2 * tick_sims
        + 0.1 * dist_sims
        + 0.1 * corr_sims
    )
    return combined.astype(np.float32)

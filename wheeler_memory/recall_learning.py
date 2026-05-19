"""Substrate-update side effects for recall events.

Public entry point: ``apply_recall_learning``. Owns three independently
kwarg-gated channels:

  - T-clock advancement (always): EMA-update of per-basin ``t_stability``
    and increment of ``t_recall_count``. Observation comes from a 1-step
    CA probe on the stored attractor, fed through ``_basin_stability_weighted``
    (Who-axis-aware when an SCM grid is provided, uniform otherwise).
  - Basin drift (``drift_basin=True``, default): blend the stored
    attractor toward ``apply_ca_dynamics(query_frame)`` at plasticity
    ``(1 - old_T) * BASIN_DRIFT_BASE_RATE`` and rewrite the .npy file.
    Preserves the legacy behavior used by ``recognize`` / ``recognize_top_k``.
  - SCM update (``scm`` + SCM kwargs present): not yet implemented in
    this commit — that's the next step. Accepted but ignored.

Promoted from ``recall_api._apply_recall_learning``; the legacy callsites
in ``recognize`` and ``recognize_top_k`` now delegate here. With all
defaults (``drift_basin=True``, no SCM kwargs), this function is
byte-equivalent to the legacy implementation.
"""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .constants import BASIN_DRIFT_BASE_RATE, T_EMA_RATE
from .dynamics import apply_ca_dynamics
from .t_metadata import ensure_t_fields, update_t_stability
from .temperature import bump_access, ensure_access_fields

if TYPE_CHECKING:
    from .scm_grid import SCMGrid


def _basin_stability_weighted(
    stored: np.ndarray,
    one_step: np.ndarray,
    scm_grid: np.ndarray | None = None,
) -> float:
    """Per-basin stability reading in [0, 1] from a 1-step CA probe.

    Currently uniform: 1 - p99(|one_step - stored|) / 2. The SCM-aware
    variant (restricting the p99 to cells where the SCM is permissive)
    is introduced in a later commit. For now, ``scm_grid`` is accepted
    but ignored — passing it produces the same result as omitting it.
    """
    delta = np.abs(one_step.flatten() - stored.flatten())
    p99 = float(np.percentile(delta, 99))
    return float(np.clip(1.0 - p99 / 2.0, 0.0, 1.0))


def apply_recall_learning(
    chunk_dir: Path,
    hex_key: str,
    query_frame: np.ndarray,
    *,
    drift_basin: bool = True,
    scm: "SCMGrid | None" = None,
    scm_top_corpus: np.ndarray | None = None,
    scm_top_exp: np.ndarray | None = None,
    scm_top_kappa: float | None = None,
) -> None:
    """Apply per-recall substrate updates for a single recognized basin.

    Side effect: acquires the chunk's index.json.lock, writes updated
    t_stability + t_recall_count + (optionally) the stored attractor.
    Persistence is intrinsic to the contract — substrate state is on disk.

    Parameters
    ----------
    drift_basin : if True (default), rewrites the stored attractor blended
        toward ``apply_ca_dynamics(query_frame)``. recognize/recognize_top_k
        pass True (legacy behavior). recall_with_interference will pass
        False once it is wired up — interference path is T-clock-only.
    scm, scm_top_corpus, scm_top_exp, scm_top_kappa : optional SCM context
        for the SCM update channel. Accepted but not yet acted on in this
        commit; the next commit moves the inline SCM update from
        ``recall_with_interference`` into here.

    With ``drift_basin=True`` and no SCM kwargs, this function is
    byte-equivalent to the legacy ``recall_api._apply_recall_learning``.
    """
    from .recall_api import _save_index_with_lock, _load_index

    lock_path = chunk_dir / "index.json.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        index = _load_index(chunk_dir)
        if hex_key not in index:
            return
        entry = index[hex_key]
        ensure_t_fields(entry)
        ensure_access_fields(entry, entry["timestamp"])

        attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
        stored = np.load(attractor_path)

        one_step = apply_ca_dynamics(stored)
        observed_stability = _basin_stability_weighted(stored, one_step)

        old_t = float(entry["metadata"]["t_stability"])
        new_t = update_t_stability(old_t, observed_stability, T_EMA_RATE)
        entry["metadata"]["t_stability"] = new_t
        entry["metadata"]["t_recall_count"] = (
            int(entry["metadata"].get("t_recall_count", 0)) + 1
        )

        if drift_basin:
            observed = apply_ca_dynamics(query_frame.astype(np.float32))
            plasticity = (1.0 - old_t) * BASIN_DRIFT_BASE_RATE
            new_attractor = (
                stored.astype(np.float32)
                + plasticity * (observed - stored.astype(np.float32))
            ).astype(stored.dtype)

            np.save(attractor_path, new_attractor)
            flat = new_attractor.flatten()
            entry["metadata"]["att_mean"] = float(flat.mean())
            entry["metadata"]["att_std"] = float(flat.std())

        bump_access(entry)
        _save_index_with_lock(chunk_dir, index)

"""Read-only analytical decomposition of one CA tick.

Mirrors the update rule in ``apply_ca_dynamics()`` (``dynamics.py:24-57``) and
exposes the per-cell contributions of each axis in a 5W1H frame. Pure
observation: this module never writes to the substrate.

Canon's 5W1H axes vs the per-tick code surface:

* ``what``  — cell current state ``frame[i,j]``. PRESENT.
* ``where`` — neighborhood input (von Neumann 4-neighbors with wrapping).
  PRESENT.
* ``who``   — neighbor identity-weights. DERIVABLE via ``np.argmax`` over the
  same neighbor stack the rule reduces with ``np.max`` — the substrate
  discards identity in the reduction, the diagnostic recovers it.
* ``how``   — combination-operator output (``delta`` and ``next_frame``).
  PRESENT.
* ``when``  — drift contribution from T (Temporal Stability). NOT CONSULTED
  by the per-tick rule. T lives in ``recall_api.py`` at the basin level
  (``plasticity = (1 - T) * BASIN_DRIFT_BASE_RATE``), not in
  ``apply_ca_dynamics``. Reported as ``None``.
* ``why``   — SCM gradient pressure. NOT CONSULTED by the per-tick rule.
  SCM gates the answer equation in ``compute_interference`` at recall time
  and feeds back via ``SCMGrid.update_from_recall``. Reported as ``None``.

The ``when``/``why`` gaps are canon-vs-code documentation, not TODOs:
synthesizing them by reading T or SCM here would misrepresent what the
per-tick substrate actually consults.
"""

from __future__ import annotations

import numpy as np

from .constants import MAX_PUSH_STRENGTH, SLOPE_FLOW_STRENGTH

NEIGHBOR_NAMES: tuple[str, str, str, str] = ("up", "down", "left", "right")


def decompose_tick(frame: np.ndarray) -> dict:
    """Decompose one CA tick into 5W1H components without changing the rule.

    Parameters
    ----------
    frame : (H, W) float array
        The CA frame at tick t.

    Returns
    -------
    dict with fields:

    ``what`` : (H, W) — ``frame[i,j]`` (cell current state).
    ``where`` : (4, H, W) — neighbor stack in ``NEIGHBOR_NAMES`` order.
    ``where_names`` : tuple — neighbor name order for the ``where`` stack.
    ``who`` : (H, W) int8 — ``argmax`` over the neighbor stack, indexing into
        ``where_names``. Recovers the neighbor identity the rule discards
        when it takes ``np.max`` of the stack.
    ``when`` : None — per-tick rule does not consult T (see module docstring).
    ``why``  : None — per-tick rule does not consult SCM (see module docstring).
    ``how_delta`` : (H, W) — same ``delta`` that ``apply_ca_dynamics`` computes.
    ``next_frame`` : (H, W) — ``clip(frame + how_delta, -1, 1)``; must equal
        ``apply_ca_dynamics(frame)`` byte-for-byte.
    ``is_max``, ``is_min``, ``is_flat`` : (H, W) bool — local-extremum masks.
    """
    n_up = np.roll(frame, 1, axis=0)
    n_down = np.roll(frame, -1, axis=0)
    n_left = np.roll(frame, 1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    neighbors = np.stack([n_up, n_down, n_left, n_right])
    max_neighbor = neighbors.max(axis=0)

    is_max = (
        (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    )
    is_min = (
        (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)
    )
    is_flat = is_max & is_min

    delta = np.zeros_like(frame)
    delta = np.where(is_max & ~is_flat, (1 - frame) * MAX_PUSH_STRENGTH, delta)
    delta = np.where(is_min & ~is_flat, (-1 - frame) * MAX_PUSH_STRENGTH, delta)
    delta = np.where(
        ~is_max & ~is_min, (max_neighbor - frame) * SLOPE_FLOW_STRENGTH, delta
    )

    who = np.argmax(neighbors, axis=0).astype(np.int8)

    return {
        "what": frame.copy(),
        "where": neighbors,
        "where_names": NEIGHBOR_NAMES,
        "who": who,
        "when": None,
        "why": None,
        "how_delta": delta,
        "next_frame": np.clip(frame + delta, -1, 1),
        "is_max": is_max,
        "is_min": is_min,
        "is_flat": is_flat,
    }

"""Per-basin Temporal Stability (T) metadata: lazy backfill and EMA updates.

T is a per-attractor float in [0, 1] that gates basin drift on recall.
High T → rigid (drifts slowly). Low T → plastic (drifts quickly).
Stored in each chunk's index.json under entry["metadata"]["t_stability"].
"""

from __future__ import annotations

from .constants import T_EMA_RATE, T_INIT_DEFAULT


def ensure_t_fields(entry: dict) -> dict:
    """Backfill t_stability and t_recall_count on legacy entries.

    Mirrors temperature.ensure_access_fields. Mutates and returns *entry*
    so callers can chain.
    """
    meta = entry.setdefault("metadata", {})
    if "t_stability" not in meta:
        meta["t_stability"] = T_INIT_DEFAULT
    if "t_recall_count" not in meta:
        meta["t_recall_count"] = 0
    return entry


def update_t_stability(
    current_t: float,
    observed_stability: float,
    rate: float = T_EMA_RATE,
) -> float:
    """EMA update: T_new = (1 - rate) * T_old + rate * observed_stability.

    Result is clamped to [0, 1]. observed_stability is expected to be a
    score_energy reading from cortex_scm (already in [0, 1]).
    """
    new = (1.0 - rate) * current_t + rate * observed_stability
    return float(max(0.0, min(1.0, new)))

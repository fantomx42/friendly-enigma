"""Temperature dynamics for Wheeler Memory.

Pure computation module — no I/O. Memories have a temperature reflecting
how recently and frequently they're accessed:

    temp = base_from_hits * decay_from_time

    base_from_hits = min(1.0, 0.3 + 0.7 * (hit_count / HIT_SATURATION))
    decay_from_time = 2 ^ (-days_since_last_access / HALF_LIFE_DAYS)

Tiers: hot >= 0.6, warm >= 0.3, cold < 0.3
"""

from datetime import datetime, timezone

HALF_LIFE_DAYS = 7.0
HIT_SATURATION = 10
TIER_HOT = 0.6
TIER_WARM = 0.3


def compute_temperature(
    hit_count: int,
    last_accessed: str | datetime,
    now: datetime | None = None,
) -> float:
    """Compute temperature from access count and recency.

    Args:
        hit_count: Number of times this memory has been recalled.
        last_accessed: ISO-8601 timestamp or datetime of last access.
        now: Current time (defaults to utcnow).

    Returns:
        Temperature in [0, 1].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if isinstance(last_accessed, str):
        last_accessed = datetime.fromisoformat(last_accessed)

    days_since = max(0.0, (now - last_accessed).total_seconds() / 86400.0)

    base_from_hits = min(1.0, 0.3 + 0.7 * (hit_count / HIT_SATURATION))
    decay_from_time = 2.0 ** (-days_since / HALF_LIFE_DAYS)

    # Round to 4 decimals to avoid float noise at tier boundaries
    # (handles seconds-level gaps between store and immediate recall)
    return round(base_from_hits * decay_from_time, 4)


def temperature_tier(temp: float) -> str:
    """Classify temperature into hot / warm / cold."""
    if temp >= TIER_HOT:
        return "hot"
    if temp >= TIER_WARM:
        return "warm"
    return "cold"


def ensure_access_fields(entry: dict, creation_timestamp: str) -> dict:
    """Backfill hit_count and last_accessed on legacy entries.

    Mutates and returns *entry* so callers can chain.
    """
    meta = entry.setdefault("metadata", {})
    if "hit_count" not in meta:
        meta["hit_count"] = 0
    if "last_accessed" not in meta:
        meta["last_accessed"] = creation_timestamp
    return entry


def bump_access(entry: dict) -> dict:
    """Increment hit_count and update last_accessed to now.

    Mutates and returns *entry*.
    """
    meta = entry.setdefault("metadata", {})
    meta["hit_count"] = meta.get("hit_count", 0) + 1
    meta["last_accessed"] = datetime.now(timezone.utc).isoformat()
    return entry

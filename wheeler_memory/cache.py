"""Mtime-checking in-memory cache for JSON files (index, associations).

Avoids redundant disk reads and JSON parsing when the same chunk is
accessed multiple times within a single recall pass.
"""

import copy
import json
from pathlib import Path

_cache: dict[str, tuple[float, dict]] = {}  # str(path) → (mtime, data)


def cached_load_json(path: Path, default: dict | None = None) -> dict:
    """Load a JSON file, returning a cached copy if mtime hasn't changed."""
    key = str(path)
    if not path.exists():
        _cache.pop(key, None)
        return default if default is not None else {}

    mtime = path.stat().st_mtime
    entry = _cache.get(key)
    if entry is not None and entry[0] == mtime:
        return copy.deepcopy(entry[1])

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        data = default if default is not None else {}

    _cache[key] = (mtime, data)
    return data


def invalidate(path: Path) -> None:
    """Remove a path from the cache (call after writes)."""
    _cache.pop(str(path), None)


def clear() -> None:
    """Clear all cached entries."""
    _cache.clear()

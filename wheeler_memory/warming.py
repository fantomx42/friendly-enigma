"""Associative warming -- spreading activation between related memories.

When a memory fires (is recalled), associated memories receive a temporary
temperature boost that decays with a half-life of 1 day.  Associations
are formed at store time (attractor correlation) and by co-recall patterns.

Association graph and warmth state are persisted in the per-data-dir SQLite
database (wheeler.db) via the wheeler_memory.db backend.  This module keeps
operating on the in-memory ``{"edges": ..., "warmth": ...}`` dict shape; the
load/save helpers translate to and from the relational tables.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .constants import ASSOCIATION_THRESHOLD, CROSS_CHUNK_DECAY
from .temperature import (
    MAX_WARMTH,
    WARMTH_FLOOR,
    WARMTH_HOP1,
    WARMTH_HOP2,
    compute_warmth,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_associations(chunk_dir: Path) -> dict:
    """Load a chunk's association graph + warmth as a dict (from SQLite)."""
    from . import db

    return db.load_associations(chunk_dir)


def _save_associations(chunk_dir: Path, assoc: dict) -> None:
    """Reconcile a chunk's edges + warmth in SQLite to match *assoc*."""
    from . import db

    db.save_associations(chunk_dir, assoc)


def load_associations(chunk_dir: Path) -> dict:
    """Public accessor for the full associations dict."""
    return _load_associations(chunk_dir)


def save_associations(chunk_dir: Path, assoc: dict) -> None:
    """Public accessor to persist the full associations dict."""
    _save_associations(chunk_dir, assoc)


def load_warmth(chunk_dir: Path) -> dict:
    """Load the warmth map for a chunk.

    Returns dict of {hex_key: {"boost": float, "applied_at": str}}.
    Garbage-collects expired entries (below WARMTH_FLOOR).
    """
    from . import db

    warmth = db.load_warmth_rows(chunk_dir)
    now = datetime.now(timezone.utc)
    cleaned = {}
    expired = []
    for hk, entry in warmth.items():
        decayed = compute_warmth(entry["boost"], entry["applied_at"], now=now)
        if decayed >= WARMTH_FLOOR:
            cleaned[hk] = entry
        else:
            expired.append(hk)
    if expired:
        db.delete_warmth(chunk_dir, expired)
    return cleaned


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def _add_edge(assoc: dict, key_a: str, key_b: str, weight: float, source: str) -> None:
    """Add a bidirectional edge to the association graph."""
    now_iso = datetime.now(timezone.utc).isoformat()
    edges = assoc["edges"]
    edge_data = {
        "weight": round(weight, 4),
        "created": now_iso,
        "source": source,
    }
    edges.setdefault(key_a, {})[key_b] = edge_data
    edges.setdefault(key_b, {})[key_a] = {**edge_data}


def _has_edge(assoc: dict, key_a: str, key_b: str) -> bool:
    """Check if an edge exists between two keys."""
    return key_b in assoc.get("edges", {}).get(key_a, {})


def get_neighbors(chunk_dir: Path, hex_key: str) -> dict:
    """Return neighbors of a memory in the association graph.

    Returns dict of {neighbor_hex_key: {"weight", "source", "created"}}.
    """
    assoc = _load_associations(chunk_dir)
    return assoc.get("edges", {}).get(hex_key, {})


# ---------------------------------------------------------------------------
# Association building
# ---------------------------------------------------------------------------


def build_store_associations(
    chunk_dir: Path,
    new_hex_key: str,
    threshold: float = ASSOCIATION_THRESHOLD,
) -> int:
    """Compare new attractor against all in chunk, create edges above threshold.

    Returns the number of new edges created.
    """
    new_att_path = chunk_dir / "attractors" / f"{new_hex_key}.npy"
    if not new_att_path.exists():
        return 0

    new_att = np.load(new_att_path).flatten()
    assoc = _load_associations(chunk_dir)
    count = 0

    for att_path in (chunk_dir / "attractors").glob("*.npy"):
        other_key = att_path.stem
        if other_key == new_hex_key:
            continue
        if _has_edge(assoc, new_hex_key, other_key):
            continue
        other_att = np.load(att_path).flatten()
        corr, _ = pearsonr(new_att, other_att)
        if corr >= threshold:
            _add_edge(
                assoc,
                new_hex_key,
                other_key,
                weight=float(corr),
                source="store_correlation",
            )
            count += 1

    if count > 0:
        _save_associations(chunk_dir, assoc)
    return count


def build_co_recall_associations(chunk_dir: Path, hex_keys: list[str]) -> int:
    """Create edges between co-recalled memories within the same chunk.

    Returns the number of new edges created.
    """
    if len(hex_keys) < 2:
        return 0

    assoc = _load_associations(chunk_dir)
    count = 0

    for i, a in enumerate(hex_keys):
        for b in hex_keys[i + 1 :]:
            if _has_edge(assoc, a, b):
                continue
            att_a_path = chunk_dir / "attractors" / f"{a}.npy"
            att_b_path = chunk_dir / "attractors" / f"{b}.npy"
            if not att_a_path.exists() or not att_b_path.exists():
                continue
            att_a = np.load(att_a_path).flatten()
            att_b = np.load(att_b_path).flatten()
            corr, _ = pearsonr(att_a, att_b)
            _add_edge(assoc, a, b, weight=float(corr), source="co_recall")
            count += 1

    if count > 0:
        _save_associations(chunk_dir, assoc)
    return count


# ---------------------------------------------------------------------------
# Warmth propagation
# ---------------------------------------------------------------------------


def remove_memory_from_associations(chunk_dir: Path, hex_key: str) -> int:
    """Remove all edges and warmth for a memory. Returns edges removed."""
    assoc = _load_associations(chunk_dir)
    edges = assoc.get("edges", {})
    count = 0

    # Remove edges pointing FROM this key
    if hex_key in edges:
        count += len(edges[hex_key])
        del edges[hex_key]

    # Remove edges pointing TO this key from other nodes
    for other_key in list(edges.keys()):
        if hex_key in edges[other_key]:
            del edges[other_key][hex_key]
            count += 1
            # Clean up empty neighbor dicts
            if not edges[other_key]:
                del edges[other_key]

    # Remove warmth entry
    warmth = assoc.get("warmth", {})
    warmth.pop(hex_key, None)

    _save_associations(chunk_dir, assoc)
    return count


def propagate_warmth(chunk_dir: Path, fired_keys: list[str]) -> dict[str, float]:
    """Spread warmth from fired memories to their neighbors (max 2 hops).

    Returns dict of {hex_key: total_new_boost} for all warmed memories.
    """
    assoc = _load_associations(chunk_dir)
    edges = assoc.get("edges", {})
    warmth = assoc.setdefault("warmth", {})
    now_iso = datetime.now(timezone.utc).isoformat()

    from .polarity import EDGE_SOURCE_POLARITY

    _polar_sources = (EDGE_SOURCE_POLARITY, "avoidance_link")
    fired_set = set(fired_keys)
    warmed: dict[str, float] = {}

    for fired in fired_keys:
        visited = {fired}

        # Hop 1 — polarity_link edges do not spread warmth
        for n1, edge1 in edges.get(fired, {}).items():
            if edge1.get("source") in _polar_sources:
                continue
            if n1 in fired_set:
                continue
            if n1 not in visited:
                visited.add(n1)
                warmed[n1] = warmed.get(n1, 0.0) + WARMTH_HOP1

            # Hop 2 — polarity_link edges do not spread warmth
            for n2, edge2 in edges.get(n1, {}).items():
                if edge2.get("source") in _polar_sources:
                    continue
                if n2 in visited or n2 in fired_set:
                    continue
                visited.add(n2)
                warmed[n2] = warmed.get(n2, 0.0) + WARMTH_HOP2

    # Apply warmth, respecting MAX_WARMTH cap.  Targeted upsert of only the
    # changed warmth rows — does not rewrite the edge graph.
    updates: dict[str, dict] = {}
    for hk, boost in warmed.items():
        existing = warmth.get(hk, {})
        existing_boost = existing.get("boost", 0.0)
        if existing_boost > 0 and "applied_at" in existing:
            existing_boost = compute_warmth(existing_boost, existing["applied_at"])
        new_boost = min(existing_boost + boost, MAX_WARMTH)
        updates[hk] = {"boost": round(new_boost, 4), "applied_at": now_iso}

    if updates:
        from . import db

        db.set_warmth(chunk_dir, updates)

    return warmed


# ---------------------------------------------------------------------------
# Cross-chunk warmth propagation
# ---------------------------------------------------------------------------

_CROSS_CHUNK_INDEX = "cross_chunk_edges.json"


def _load_cross_chunk_edges(data_dir: Path) -> dict:
    """Load the cross-chunk edge index from SQLite."""
    from . import db

    return db.load_cross_chunk_edges(data_dir)


def _save_cross_chunk_edges(data_dir: Path, edges: dict) -> None:
    """Reconcile cross-chunk edges in SQLite to match *edges*."""
    from . import db

    db.save_cross_chunk_edges(data_dir, edges)


def build_cross_chunk_co_recall(
    data_dir: Path,
    results: list[dict],
) -> int:
    """Create cross-chunk edges between co-recalled memories in different chunks.

    Called after recall when top-k results span multiple chunks.
    Returns number of new edges created.
    """
    if len(results) < 2:
        return 0

    # Group by chunk
    by_chunk: dict[str, list[str]] = {}
    for r in results:
        by_chunk.setdefault(r["chunk"], []).append(r["hex_key"])

    if len(by_chunk) < 2:
        return 0  # all results from same chunk, nothing to bridge

    cross = _load_cross_chunk_edges(data_dir)
    edges = cross.setdefault("edges", {})
    now_iso = datetime.now(timezone.utc).isoformat()
    count = 0

    # Create edges between all cross-chunk pairs
    chunks = list(by_chunk.keys())
    for i, chunk_a in enumerate(chunks):
        for chunk_b in chunks[i + 1 :]:
            for key_a in by_chunk[chunk_a]:
                for key_b in by_chunk[chunk_b]:
                    edge_id = (
                        f"{key_a}:{key_b}" if key_a < key_b else f"{key_b}:{key_a}"
                    )
                    if edge_id not in edges:
                        edges[edge_id] = {
                            "chunks": sorted([chunk_a, chunk_b]),
                            "keys": sorted([key_a, key_b]),
                            "created": now_iso,
                            "co_recall_count": 1,
                        }
                        count += 1
                    else:
                        edges[edge_id]["co_recall_count"] = (
                            edges[edge_id].get("co_recall_count", 0) + 1
                        )

    if count > 0:
        _save_cross_chunk_edges(data_dir, cross)
    return count


def propagate_warmth_cross_chunk(
    data_dir: Path,
    fired_results: list[dict],
) -> dict[str, float]:
    """Spread warmth across chunk boundaries via cross-chunk edges.

    Returns dict of {hex_key: boost_applied} for warmed cross-chunk memories.
    """
    if not fired_results:
        return {}

    cross = _load_cross_chunk_edges(data_dir)
    edges = cross.get("edges", {})
    if not edges:
        return {}

    fired_keys = {r["hex_key"] for r in fired_results}
    warmed: dict[str, tuple[str, float]] = {}  # hex_key → (chunk, boost)
    now_iso = datetime.now(timezone.utc).isoformat()

    for edge_id, edge_data in edges.items():
        keys = edge_data["keys"]
        chunks = edge_data["chunks"]
        for idx, key in enumerate(keys):
            if key in fired_keys:
                other_key = keys[1 - idx]
                other_chunk = chunks[1 - idx] if chunks[0] != chunks[1] else chunks[0]
                if other_key not in fired_keys:
                    boost = WARMTH_HOP1 * CROSS_CHUNK_DECAY
                    if other_key in warmed:
                        _, existing = warmed[other_key]
                        boost = min(existing + boost, MAX_WARMTH)
                    warmed[other_key] = (other_chunk, boost)

    # Apply warmth to each target chunk's associations
    by_chunk: dict[str, dict[str, float]] = {}
    for hk, (chunk, boost) in warmed.items():
        by_chunk.setdefault(chunk, {})[hk] = boost

    from . import db

    for chunk_name, boosts in by_chunk.items():
        chunk_dir = data_dir / "chunks" / chunk_name
        existing_warmth = db.load_warmth_rows(chunk_dir)
        updates: dict[str, dict] = {}
        for hk, boost in boosts.items():
            existing = existing_warmth.get(hk, {})
            existing_boost = existing.get("boost", 0.0)
            if existing_boost > 0 and "applied_at" in existing:
                existing_boost = compute_warmth(existing_boost, existing["applied_at"])
            new_boost = min(existing_boost + boost, MAX_WARMTH)
            updates[hk] = {"boost": round(new_boost, 4), "applied_at": now_iso}
        db.set_warmth(chunk_dir, updates)

    return {hk: boost for hk, (_, boost) in warmed.items()}

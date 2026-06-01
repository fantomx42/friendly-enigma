"""SQLite persistence backend for Wheeler Memory.

This module is the single source of truth for the memory *catalog*, the
association *edge graph*, *warmth* state, the *cross-chunk* edge index, and a
``sqlite-vec`` *vector index* of corpus attractors used for recall.

It replaces the previous per-chunk JSON layout (``index.json`` +
``associations.json`` + root ``cross_chunk_edges.json``) with one SQLite
database per data directory at ``<data_dir>/wheeler.db``.

Design contract
---------------
The rest of the package reads/writes memory metadata and edges as plain Python
dicts.  To keep that contract intact, this module exposes ``load_index`` /
``save_index`` (returning the same ``{hex_key: entry}`` shape the old
``index.json`` produced) and ``load_associations`` / ``save_associations``
(returning ``{"edges": {...}, "warmth": {...}}``).  Hot paths — per-recall
``hit_count`` bumps, warmth spread, and the similarity scan — use *targeted*
SQL (``bump_access``, ``set_warmth``, ``vector_topk``) instead of rewriting a
whole-chunk blob, which is the performance win over the JSON layout.

Attractor grids (.npy), bricks (.npz), experiential grids, signatures, chunk
``metadata.json`` centroids, and the large ``*.npz`` corpora remain external
files referenced by SHA-256 hex key — they are NOT stored in the database.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 64x64 attractor flattened.  Must match dynamics grid size.
VEC_DIM = 4096

# Metadata keys promoted to their own columns (single source of truth); every
# other metadata key is round-tripped through the ``metadata_json`` column.
_PROMOTED_META = ("hit_count", "last_accessed", "att_mean", "att_std", "memory_type")

# Edge sources that carry a meaningful decay_count in the rebuilt dict.
_POLAR_SOURCES = ("polarity_link", "avoidance_link")

_DB_FILENAME = "wheeler.db"

_connections: dict[str, sqlite3.Connection] = {}
_write_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------


def _db_path(data_dir: Path) -> Path:
    return Path(data_dir) / _DB_FILENAME


def _resolve(chunk_dir: Path) -> tuple[Path, str]:
    """Derive (data_dir, chunk_name) from a ``<data_dir>/chunks/<chunk>`` path."""
    chunk_dir = Path(chunk_dir)
    return chunk_dir.parent.parent, chunk_dir.name


def get_connection(data_dir: Path) -> sqlite3.Connection:
    """Open (and cache) the SQLite connection for *data_dir*.

    Enables WAL, a busy timeout, loads the ``sqlite-vec`` extension, and
    ensures the schema exists.  Cached per resolved DB path so repeated calls
    are cheap.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = _db_path(data_dir)
    key = str(path.resolve())
    conn = _connections.get(key)
    if conn is not None:
        return conn

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")

    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _ensure_schema(conn)
    _connections[key] = conn
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            key               TEXT PRIMARY KEY,
            chunk             TEXT NOT NULL,
            text              TEXT,
            state             TEXT,
            convergence_ticks INTEGER,
            timestamp         TEXT,
            hit_count         INTEGER DEFAULT 0,
            last_accessed     TEXT,
            att_mean          REAL,
            att_std           REAL,
            memory_type       TEXT,
            grid              TEXT DEFAULT 'corpus',
            metadata_json     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_memories_chunk ON memories(chunk);
        CREATE INDEX IF NOT EXISTS idx_memories_recall
            ON memories(chunk, grid, memory_type);

        CREATE TABLE IF NOT EXISTS edges (
            src             TEXT NOT NULL,
            dst             TEXT NOT NULL,
            chunk_src       TEXT,
            chunk_dst       TEXT,
            weight          REAL,
            source          TEXT,
            created         TEXT,
            decay_count     INTEGER DEFAULT 0,
            co_recall_count INTEGER DEFAULT 0,
            PRIMARY KEY (src, dst)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
        CREATE INDEX IF NOT EXISTS idx_edges_chunkpair ON edges(chunk_src, chunk_dst);

        CREATE TABLE IF NOT EXISTS warmth (
            key        TEXT NOT NULL,
            chunk      TEXT NOT NULL,
            boost      REAL,
            applied_at TEXT,
            PRIMARY KEY (key, chunk)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            key TEXT PRIMARY KEY,
            chunk TEXT PARTITION KEY,
            embedding float[{VEC_DIM}] distance_metric=cosine
        );
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------


def _pack(vec: np.ndarray) -> bytes:
    """Pack a 1-D float array into the little-endian float32 blob vec0 wants."""
    arr = np.ascontiguousarray(vec, dtype=np.float32)
    return struct.pack(f"<{arr.size}f", *arr.tolist())


def _centered(attractor: np.ndarray) -> np.ndarray | None:
    """Return the mean-centered flattened attractor, or None if degenerate.

    Cosine similarity of two mean-centered vectors equals their Pearson
    correlation, so storing the centered grid lets ``sqlite-vec``'s cosine
    KNN reproduce the recall ranking exactly.  A constant grid (std == 0)
    centers to all-zeros, for which cosine is undefined — skip it (it could
    never be a meaningful match, mirroring the old NaN→0.0 behavior).
    """
    flat = np.asarray(attractor, dtype=np.float32).flatten()
    if flat.size != VEC_DIM:
        return None
    centered = flat - flat.mean()
    if not np.any(centered):
        return None
    return centered


def _entry_from_row(row: sqlite3.Row) -> dict:
    """Rebuild the ``index.json``-shaped entry dict from a memories row."""
    meta: dict = {}
    if row["metadata_json"]:
        meta = json.loads(row["metadata_json"])
    # Promoted columns are the source of truth — overlay them onto the blob.
    meta["hit_count"] = row["hit_count"] if row["hit_count"] is not None else 0
    meta["last_accessed"] = row["last_accessed"]
    if row["att_mean"] is not None:
        meta["att_mean"] = row["att_mean"]
    if row["att_std"] is not None:
        meta["att_std"] = row["att_std"]
    if row["memory_type"] is not None:
        meta["memory_type"] = row["memory_type"]

    entry: dict = {
        "text": row["text"],
        "state": row["state"],
        "convergence_ticks": row["convergence_ticks"],
        "timestamp": row["timestamp"],
        "metadata": meta,
        "chunk": row["chunk"],
    }
    grid = row["grid"]
    if grid and grid != "corpus":
        entry["grid"] = grid
    return entry


def _row_params_from_entry(key: str, chunk: str, entry: dict) -> tuple:
    """Flatten an entry dict into the column tuple for an upsert."""
    meta = dict(entry.get("metadata", {}))
    hit_count = int(meta.get("hit_count", 0) or 0)
    last_accessed = meta.get("last_accessed")
    att_mean = meta.get("att_mean")
    att_std = meta.get("att_std")
    memory_type = meta.get("memory_type")
    # Everything not promoted to a column goes into metadata_json.
    extra = {k: v for k, v in meta.items() if k not in _PROMOTED_META}
    metadata_json = json.dumps(extra) if extra else None
    grid = entry.get("grid", "corpus")
    return (
        key,
        chunk,
        entry.get("text"),
        entry.get("state"),
        entry.get("convergence_ticks"),
        entry.get("timestamp"),
        hit_count,
        last_accessed,
        att_mean,
        att_std,
        memory_type,
        grid,
        metadata_json,
    )


_UPSERT_MEMORY_SQL = """
    INSERT INTO memories
        (key, chunk, text, state, convergence_ticks, timestamp,
         hit_count, last_accessed, att_mean, att_std, memory_type, grid, metadata_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(key) DO UPDATE SET
        chunk=excluded.chunk, text=excluded.text, state=excluded.state,
        convergence_ticks=excluded.convergence_ticks, timestamp=excluded.timestamp,
        hit_count=excluded.hit_count, last_accessed=excluded.last_accessed,
        att_mean=excluded.att_mean, att_std=excluded.att_std,
        memory_type=excluded.memory_type, grid=excluded.grid,
        metadata_json=excluded.metadata_json
"""


# ---------------------------------------------------------------------------
# Catalog: dict-contract accessors
# ---------------------------------------------------------------------------


def load_index(chunk_dir: Path) -> dict:
    """Return ``{hex_key: entry}`` for one chunk (drop-in for the old reader)."""
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    rows = conn.execute("SELECT * FROM memories WHERE chunk = ?", (chunk,)).fetchall()
    return {row["key"]: _entry_from_row(row) for row in rows}


def get_entries(data_dir: Path, keys: list[str]) -> dict:
    """Targeted fetch of specific entries (used by the recall candidate path)."""
    if not keys:
        return {}
    conn = get_connection(data_dir)
    out: dict = {}
    # Chunk the IN-list to stay under SQLite's variable limit.
    for i in range(0, len(keys), 500):
        batch = keys[i : i + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT * FROM memories WHERE key IN ({placeholders})", batch
        ).fetchall()
        for row in rows:
            out[row["key"]] = _entry_from_row(row)
    return out


def upsert_memory(chunk_dir: Path, hex_key: str, entry: dict) -> None:
    """Insert or replace a single memory row (hot store path)."""
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.execute(_UPSERT_MEMORY_SQL, _row_params_from_entry(hex_key, chunk, entry))
        conn.commit()


def save_index(chunk_dir: Path, index: dict) -> None:
    """Reconcile a whole chunk's rows to match *index* (cold paths only).

    Upserts every entry present and deletes rows for keys no longer present,
    so callers that do ``load → mutate/pop → save`` behave as before.  Also
    drops vectors for removed keys.
    """
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        existing = {
            r["key"]
            for r in conn.execute(
                "SELECT key FROM memories WHERE chunk = ?", (chunk,)
            ).fetchall()
        }
        desired = set(index)
        for key in existing - desired:
            conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.execute("DELETE FROM vec_memories WHERE key = ?", (key,))
        conn.executemany(
            _UPSERT_MEMORY_SQL,
            [_row_params_from_entry(k, chunk, e) for k, e in index.items()],
        )
        conn.commit()


def delete_memory(chunk_dir: Path, hex_key: str) -> None:
    """Remove a memory row and its vector (caller deletes .npy/.npz/edges)."""
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.execute("DELETE FROM memories WHERE key = ?", (hex_key,))
        conn.execute("DELETE FROM vec_memories WHERE key = ?", (hex_key,))
        conn.commit()


def bump_access(chunk_dir: Path, hex_keys: list[str]) -> None:
    """Increment hit_count and refresh last_accessed for the given keys.

    Targeted UPDATE — no whole-chunk rewrite.  Matches temperature.bump_access
    semantics (hit_count += 1, last_accessed = now).
    """
    if not hex_keys:
        return
    data_dir, _ = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn.executemany(
            "UPDATE memories SET hit_count = COALESCE(hit_count, 0) + 1, "
            "last_accessed = ? WHERE key = ?",
            [(now_iso, hk) for hk in hex_keys],
        )
        conn.commit()


def existing_chunks(data_dir: Path) -> list[str]:
    """Distinct chunks that hold at least one memory (sorted)."""
    if not _db_path(Path(data_dir)).exists():
        return []
    conn = get_connection(data_dir)
    rows = conn.execute(
        "SELECT DISTINCT chunk FROM memories ORDER BY chunk"
    ).fetchall()
    return [r["chunk"] for r in rows]


def all_texts(data_dir: Path) -> list[tuple[str, str]]:
    """Return [(chunk, text), ...] for every stored memory (corpus harvest)."""
    if not _db_path(Path(data_dir)).exists():
        return []
    conn = get_connection(data_dir)
    rows = conn.execute(
        "SELECT chunk, text FROM memories WHERE text IS NOT NULL"
    ).fetchall()
    return [(r["chunk"], r["text"]) for r in rows]


# ---------------------------------------------------------------------------
# Vector index (recall)
# ---------------------------------------------------------------------------


def upsert_vector(chunk_dir: Path, hex_key: str, attractor: np.ndarray) -> None:
    """Store the mean-centered attractor for cosine(=Pearson) KNN recall.

    Only recall-eligible corpus attractors should be passed here; degenerate
    constant grids are skipped.
    """
    centered = _centered(attractor)
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.execute("DELETE FROM vec_memories WHERE key = ?", (hex_key,))
        if centered is not None:
            conn.execute(
                "INSERT INTO vec_memories(key, chunk, embedding) VALUES (?,?,?)",
                (hex_key, chunk, _pack(centered)),
            )
        conn.commit()


def delete_vector(chunk_dir: Path, hex_key: str) -> None:
    data_dir, _ = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.execute("DELETE FROM vec_memories WHERE key = ?", (hex_key,))
        conn.commit()


def vector_topk(
    data_dir: Path,
    query: np.ndarray,
    chunks: list[str],
    k: int,
) -> list[tuple[str, str, float]]:
    """Top-k corpus attractors by Pearson similarity to *query*.

    *query* is a 64x64 (or flat 4096) attractor; it is mean-centered here so
    cosine distance maps to ``similarity = 1 - distance = Pearson``.
    Returns [(hex_key, chunk, similarity), ...] sorted by descending
    similarity.  Restricted to the given chunks.
    """
    if not chunks or k <= 0:
        return []
    centered = _centered(query)
    if centered is None:
        return []
    if not _db_path(Path(data_dir)).exists():
        return []
    conn = get_connection(data_dir)
    placeholders = ",".join("?" * len(chunks))
    sql = (
        "SELECT key, chunk, distance FROM vec_memories "
        f"WHERE embedding MATCH ? AND k = ? AND chunk IN ({placeholders}) "
        "ORDER BY distance"
    )
    params: list = [_pack(centered), int(k), *chunks]
    rows = conn.execute(sql, params).fetchall()
    return [(r["key"], r["chunk"], 1.0 - float(r["distance"])) for r in rows]


# ---------------------------------------------------------------------------
# Associations: dict-contract accessors (edges + warmth)
# ---------------------------------------------------------------------------


def load_associations(chunk_dir: Path) -> dict:
    """Return ``{"edges": {src: {dst: edge}}, "warmth": {key: warmth}}``.

    Edges here are the *intra-chunk* graph (chunk_src == chunk_dst), matching
    the old per-chunk associations.json.
    """
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)

    edges: dict = {}
    rows = conn.execute(
        "SELECT src, dst, weight, source, created, decay_count "
        "FROM edges WHERE chunk_src = ? AND chunk_dst = ?",
        (chunk, chunk),
    ).fetchall()
    for r in rows:
        edge: dict = {
            "weight": r["weight"],
            "created": r["created"],
            "source": r["source"],
        }
        if r["source"] in _POLAR_SOURCES:
            edge["decay_count"] = r["decay_count"] or 0
        edges.setdefault(r["src"], {})[r["dst"]] = edge

    warmth: dict = {}
    for r in conn.execute(
        "SELECT key, boost, applied_at FROM warmth WHERE chunk = ?", (chunk,)
    ).fetchall():
        warmth[r["key"]] = {"boost": r["boost"], "applied_at": r["applied_at"]}

    return {"edges": edges, "warmth": warmth}


def save_associations(chunk_dir: Path, assoc: dict) -> None:
    """Reconcile a chunk's intra-chunk edges + warmth to match *assoc*."""
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    edges_dict = assoc.get("edges", {})
    warmth_dict = assoc.get("warmth", {})

    desired_edges: set[tuple[str, str]] = set()
    edge_rows = []
    for src, nbrs in edges_dict.items():
        for dst, ed in nbrs.items():
            desired_edges.add((src, dst))
            decay = ed.get("decay_count", ed.get("safe_recall_count", 0))
            edge_rows.append(
                (
                    src,
                    dst,
                    chunk,
                    chunk,
                    ed.get("weight"),
                    ed.get("source"),
                    ed.get("created"),
                    int(decay or 0),
                )
            )

    with _write_lock:
        existing = conn.execute(
            "SELECT src, dst FROM edges WHERE chunk_src = ? AND chunk_dst = ?",
            (chunk, chunk),
        ).fetchall()
        for r in existing:
            if (r["src"], r["dst"]) not in desired_edges:
                conn.execute(
                    "DELETE FROM edges WHERE src = ? AND dst = ?", (r["src"], r["dst"])
                )
        conn.executemany(
            """INSERT INTO edges
                 (src, dst, chunk_src, chunk_dst, weight, source, created, decay_count)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(src, dst) DO UPDATE SET
                 chunk_src=excluded.chunk_src, chunk_dst=excluded.chunk_dst,
                 weight=excluded.weight, source=excluded.source,
                 created=excluded.created, decay_count=excluded.decay_count""",
            edge_rows,
        )

        existing_w = {
            r["key"]
            for r in conn.execute(
                "SELECT key FROM warmth WHERE chunk = ?", (chunk,)
            ).fetchall()
        }
        for key in existing_w - set(warmth_dict):
            conn.execute(
                "DELETE FROM warmth WHERE key = ? AND chunk = ?", (key, chunk)
            )
        conn.executemany(
            """INSERT INTO warmth (key, chunk, boost, applied_at) VALUES (?,?,?,?)
               ON CONFLICT(key, chunk) DO UPDATE SET
                 boost=excluded.boost, applied_at=excluded.applied_at""",
            [
                (k, chunk, w.get("boost"), w.get("applied_at"))
                for k, w in warmth_dict.items()
            ],
        )
        conn.commit()


# --- targeted warmth ops (per-recall hot path) -----------------------------


def load_warmth_rows(chunk_dir: Path) -> dict:
    """Return ``{key: {boost, applied_at}}`` for a chunk (no GC)."""
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    rows = conn.execute(
        "SELECT key, boost, applied_at FROM warmth WHERE chunk = ?", (chunk,)
    ).fetchall()
    return {
        r["key"]: {"boost": r["boost"], "applied_at": r["applied_at"]} for r in rows
    }


def set_warmth(chunk_dir: Path, warmth_map: dict) -> None:
    """Upsert warmth rows for a chunk (targeted; no edge rewrite)."""
    if not warmth_map:
        return
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.executemany(
            """INSERT INTO warmth (key, chunk, boost, applied_at) VALUES (?,?,?,?)
               ON CONFLICT(key, chunk) DO UPDATE SET
                 boost=excluded.boost, applied_at=excluded.applied_at""",
            [
                (k, chunk, w.get("boost"), w.get("applied_at"))
                for k, w in warmth_map.items()
            ],
        )
        conn.commit()


def delete_warmth(chunk_dir: Path, keys: list[str]) -> None:
    """Delete warmth rows for the given keys in a chunk (GC of expired warmth)."""
    if not keys:
        return
    data_dir, chunk = _resolve(chunk_dir)
    conn = get_connection(data_dir)
    with _write_lock:
        conn.executemany(
            "DELETE FROM warmth WHERE key = ? AND chunk = ?",
            [(k, chunk) for k in keys],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Cross-chunk edge index
# ---------------------------------------------------------------------------


def load_cross_chunk_edges(data_dir: Path) -> dict:
    """Return ``{"edges": {edge_id: {chunks, keys, created, co_recall_count}}}``.

    Cross-chunk edges are rows where chunk_src != chunk_dst.  edge_id, keys,
    and chunks reproduce the old ``cross_chunk_edges.json`` shape (keys and
    chunks each independently sorted, as the original builder stored them).
    """
    if not _db_path(Path(data_dir)).exists():
        return {"edges": {}}
    conn = get_connection(data_dir)
    edges: dict = {}
    rows = conn.execute(
        "SELECT src, dst, chunk_src, chunk_dst, created, co_recall_count "
        "FROM edges WHERE chunk_src <> chunk_dst"
    ).fetchall()
    for r in rows:
        keys = sorted([r["src"], r["dst"]])
        edge_id = f"{keys[0]}:{keys[1]}"
        edges[edge_id] = {
            "chunks": [r["chunk_src"], r["chunk_dst"]],
            "keys": keys,
            "created": r["created"],
            "co_recall_count": r["co_recall_count"] or 0,
        }
    return {"edges": edges}


def save_cross_chunk_edges(data_dir: Path, cross: dict) -> None:
    """Reconcile cross-chunk edge rows (chunk_src != chunk_dst) to match *cross*."""
    conn = get_connection(data_dir)
    edges_dict = cross.get("edges", {})

    desired: set[tuple[str, str]] = set()
    rows = []
    for edge_data in edges_dict.values():
        keys = sorted(edge_data["keys"])
        chunks = sorted(edge_data["chunks"])
        src, dst = keys[0], keys[1]
        desired.add((src, dst))
        rows.append(
            (
                src,
                dst,
                chunks[0],
                chunks[1],
                edge_data.get("created"),
                int(edge_data.get("co_recall_count", 0) or 0),
            )
        )

    with _write_lock:
        existing = conn.execute(
            "SELECT src, dst FROM edges WHERE chunk_src <> chunk_dst"
        ).fetchall()
        for r in existing:
            if (r["src"], r["dst"]) not in desired:
                conn.execute(
                    "DELETE FROM edges WHERE src = ? AND dst = ?", (r["src"], r["dst"])
                )
        conn.executemany(
            """INSERT INTO edges
                 (src, dst, chunk_src, chunk_dst, source, created, co_recall_count)
               VALUES (?,?,?,?,'cross_co_recall',?,?)
               ON CONFLICT(src, dst) DO UPDATE SET
                 chunk_src=excluded.chunk_src, chunk_dst=excluded.chunk_dst,
                 created=excluded.created, co_recall_count=excluded.co_recall_count""",
            rows,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Test / maintenance helpers
# ---------------------------------------------------------------------------


def close_all() -> None:
    """Close and forget all cached connections (used by tests/teardown)."""
    with _write_lock:
        for conn in _connections.values():
            try:
                conn.close()
            except Exception:
                pass
        _connections.clear()

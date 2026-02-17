"""Attractor storage, indexing, and recall by Pearson correlation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .brick import MemoryBrick
from .chunking import (
    DEFAULT_CHUNK,
    get_chunk_dir,
    list_existing_chunks,
    select_chunk,
    select_recall_chunks,
    touch_chunk_metadata,
)
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex

DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _get_data_dir(data_dir: str | Path | None = None) -> Path:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(chunk_dir: Path, index: dict) -> None:
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def store_memory(
    text: str,
    result: dict,
    brick: MemoryBrick,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> str:
    """Save attractor, brick, and index entry for a memory.

    Returns the hex hash key used for storage.
    """
    d = _get_data_dir(data_dir)
    if chunk is None:
        chunk = select_chunk(text)

    chunk_dir = get_chunk_dir(d, chunk)
    hex_key = text_to_hex(text)

    np.save(chunk_dir / "attractors" / f"{hex_key}.npy", result["attractor"])
    brick.save(chunk_dir / "bricks" / f"{hex_key}.npz")

    index = _load_index(chunk_dir)
    index[hex_key] = {
        "text": text,
        "state": result["state"],
        "convergence_ticks": result["convergence_ticks"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": result.get("metadata", {}),
        "chunk": chunk,
    }
    _save_index(chunk_dir, index)
    touch_chunk_metadata(chunk_dir, stored=True)
    return hex_key


def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> list[dict]:
    """Recall stored memories by Pearson correlation with the query's attractor.

    Searches across matched chunks, merges results sorted by similarity.
    """
    d = _get_data_dir(data_dir)

    if chunk is not None:
        chunks_to_search = [chunk]
    else:
        chunks_to_search = select_recall_chunks(text)
        # Also include any existing chunks not already selected
        existing = list_existing_chunks(d)
        for c in existing:
            if c not in chunks_to_search:
                chunks_to_search.append(c)

    query_frame = hash_to_frame(text)
    query_result = evolve_and_interpret(query_frame)
    query_flat = query_result["attractor"].flatten()

    results = []
    for c in chunks_to_search:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        if not index:
            continue

        touch_chunk_metadata(chunk_dir)

        for hex_key, meta in index.items():
            attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if not attractor_path.exists():
                continue
            attractor = np.load(attractor_path)
            corr, _ = pearsonr(query_flat, attractor.flatten())
            results.append({
                "hex_key": hex_key,
                "text": meta["text"],
                "similarity": float(corr),
                "state": meta["state"],
                "convergence_ticks": meta["convergence_ticks"],
                "timestamp": meta["timestamp"],
                "chunk": c,
            })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


def list_memories(
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> list[dict]:
    """List all stored memories from the index."""
    d = _get_data_dir(data_dir)

    if chunk is not None:
        chunks_to_list = [chunk]
    else:
        chunks_to_list = list_existing_chunks(d)

    all_memories = []
    for c in chunks_to_list:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        for k, v in index.items():
            all_memories.append({"hex_key": k, "chunk": c, **v})

    return all_memories

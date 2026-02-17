"""Attractor storage, indexing, and recall by Pearson correlation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .brick import MemoryBrick
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex

DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _get_data_dir(data_dir: str | Path | None = None) -> Path:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "attractors").mkdir(exist_ok=True)
    (d / "bricks").mkdir(exist_ok=True)
    return d


def _load_index(data_dir: Path) -> dict:
    index_path = data_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(data_dir: Path, index: dict) -> None:
    index_path = data_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def store_memory(
    text: str,
    result: dict,
    brick: MemoryBrick,
    data_dir: str | Path | None = None,
) -> str:
    """Save attractor, brick, and index entry for a memory.

    Returns the hex hash key used for storage.
    """
    d = _get_data_dir(data_dir)
    hex_key = text_to_hex(text)

    np.save(d / "attractors" / f"{hex_key}.npy", result["attractor"])
    brick.save(d / "bricks" / f"{hex_key}.npz")

    index = _load_index(d)
    index[hex_key] = {
        "text": text,
        "state": result["state"],
        "convergence_ticks": result["convergence_ticks"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": result.get("metadata", {}),
    }
    _save_index(d, index)
    return hex_key


def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
) -> list[dict]:
    """Recall stored memories by Pearson correlation with the query's attractor.

    Hashes the query text to a frame, then compares against all stored
    attractors. Returns top_k matches sorted by similarity.
    """
    d = _get_data_dir(data_dir)
    index = _load_index(d)

    if not index:
        return []

    query_frame = hash_to_frame(text)
    query_result = evolve_and_interpret(query_frame)
    query_flat = query_result["attractor"].flatten()

    results = []
    for hex_key, meta in index.items():
        attractor_path = d / "attractors" / f"{hex_key}.npy"
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
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


def list_memories(data_dir: str | Path | None = None) -> list[dict]:
    """List all stored memories from the index."""
    d = _get_data_dir(data_dir)
    index = _load_index(d)
    return [{"hex_key": k, **v} for k, v in index.items()]

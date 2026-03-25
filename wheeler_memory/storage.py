"""Attractor storage, indexing, and recall by Pearson correlation."""

import fcntl
import heapq
import json
import sys
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
from .attention import compute_attention_budget, salience_from_temperature
from .dynamics import compute_attractor_features, evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .temperature import (
    SALIENCE_DEFAULT,
    bump_access,
    compute_temperature,
    effective_temperature,
    ensure_access_fields,
    temperature_tier,
)
from .warming import (
    _load_associations,
    _save_associations,
    build_co_recall_associations,
    build_cross_chunk_co_recall,
    build_store_associations,
    load_warmth,
    propagate_warmth,
    propagate_warmth_cross_chunk,
)


# Lazy import for embedding (optional dependency)
def _get_embed_to_frame():
    from .embedding import embed_to_frame

    return embed_to_frame


# Default encoder from constants
def _get_default_encoder():
    from .constants import DEFAULT_ENCODER

    return DEFAULT_ENCODER


DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"


def _get_data_dir(data_dir: str | Path | None = None) -> Path:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(chunk_dir: Path) -> dict:
    from .cache import cached_load_json

    return cached_load_json(chunk_dir / "index.json", default={})


def _save_index(chunk_dir: Path, index: dict) -> None:
    from .cache import invalidate

    index_path = chunk_dir / "index.json"
    tmp_path = index_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(index, indent=2))
    tmp_path.replace(index_path)  # atomic on POSIX
    invalidate(index_path)


def batch_store_memories(
    entries: list[tuple[str, dict, "MemoryBrick", str]],
    data_dir: Path | None = None,
) -> int:
    """Store multiple memories with one index read+write per chunk.

    Parameters
    ----------
    entries : list of (text, evolve_result, brick, chunk)
    data_dir : Wheeler data directory

    Returns number of successfully stored entries.
    """
    if not entries:
        return 0

    d = _get_data_dir(data_dir)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Group by chunk so we only lock+read+write each index once
    by_chunk: dict[str, list[tuple[str, dict, MemoryBrick]]] = {}
    for text, result, brick, chunk in entries:
        by_chunk.setdefault(chunk, []).append((text, result, brick))

    stored = 0
    for chunk, items in by_chunk.items():
        chunk_dir = get_chunk_dir(d, chunk)
        lock_path = chunk_dir / "index.json.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            index = _load_index(chunk_dir)
            for text, result, brick in items:
                hex_key = text_to_hex(text)
                # Write attractor + brick files (fast, outside lock in ideal world
                # but done here for simplicity since they're independent files)
                np.save(
                    chunk_dir / "attractors" / f"{hex_key}.npy", result["attractor"]
                )
                brick.save(chunk_dir / "bricks" / f"{hex_key}.npz")
                # Build index entry
                meta = dict(result.get("metadata", {}))
                meta["hit_count"] = 0
                meta["last_accessed"] = now_iso
                flat = result["attractor"].flatten()
                meta["att_mean"] = float(flat.mean())
                meta["att_std"] = float(flat.std())
                index[hex_key] = {
                    "text": text,
                    "state": result["state"],
                    "convergence_ticks": result["convergence_ticks"],
                    "timestamp": now_iso,
                    "metadata": meta,
                    "chunk": chunk,
                }
                stored += 1
            _save_index(chunk_dir, index)
        touch_chunk_metadata(chunk_dir, stored=True)

    return stored


def store_memory(
    text: str,
    result: dict,
    brick: MemoryBrick,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    auto_evict: bool = True,
    memory_type: str | None = None,
    grid: str = "corpus",
    experiential_meta: "ExperientialMeta | None" = None,
) -> str:
    """Save attractor, brick, and index entry for a memory.

    Returns the hex hash key used for storage.

    Parameters
    ----------
    grid : "corpus" or "experiential".  Corpus writes to attractors/ (default).
        Experiential writes to experiential/ subdirectory with temporal context.
    experiential_meta : temporal context for experiential memories.
    memory_type : written into metadata (e.g. "polar").
    """
    d = _get_data_dir(data_dir)
    if chunk is None:
        chunk = select_chunk(text)

    chunk_dir = get_chunk_dir(d, chunk)
    hex_key = text_to_hex(text)

    if grid == "experiential":
        from .experiential import (
            ExperientialMeta,
            experiential_bricks_dir,
            experiential_dir,
        )

        exp_dir = experiential_dir(chunk_dir)
        np.save(exp_dir / f"{hex_key}.npy", result["attractor"])
        brick.save(experiential_bricks_dir(chunk_dir) / f"{hex_key}.npz")
        meta = experiential_meta or ExperientialMeta()
        meta.save(exp_dir / f"{hex_key}.meta.json")
    else:
        np.save(chunk_dir / "attractors" / f"{hex_key}.npy", result["attractor"])
        brick.save(chunk_dir / "bricks" / f"{hex_key}.npz")

    lock_path = chunk_dir / "index.json.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        index = _load_index(chunk_dir)
        now_iso = datetime.now(timezone.utc).isoformat()
        base_metadata = result.get("metadata", {})
        base_metadata["hit_count"] = 0
        base_metadata["last_accessed"] = now_iso
        # Cache attractor norm for faster Pearson during recall
        flat = result["attractor"].flatten()
        base_metadata["att_mean"] = float(flat.mean())
        base_metadata["att_std"] = float(flat.std())
        if memory_type is not None:
            base_metadata["memory_type"] = memory_type
        entry = {
            "text": text,
            "state": result["state"],
            "convergence_ticks": result["convergence_ticks"],
            "timestamp": now_iso,
            "metadata": base_metadata,
            "chunk": chunk,
        }
        if grid == "experiential":
            entry["grid"] = "experiential"
        index[hex_key] = entry
        _save_index(chunk_dir, index)
    touch_chunk_metadata(chunk_dir, stored=True)
    build_store_associations(chunk_dir, hex_key)

    if auto_evict:
        from .eviction import evict_for_capacity

        evict_for_capacity(d)

    return hex_key


def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    temperature_boost: float = 0.0,
    use_embedding: bool = False,
    encoder: str | None = None,
    query_frame: "np.ndarray | None" = None,
    reconstruct: bool = False,
    reconstruct_alpha: float = 0.3,
    salience: float | None = None,
    polar_decay: bool = False,
    readonly: bool = False,
) -> list[dict]:
    """Recall stored memories by Pearson correlation with the query's attractor.

    Searches across matched chunks, merges results sorted by effective similarity.
    When temperature_boost > 0, hotter memories get a ranking bonus.
    When use_embedding is True, uses sentence embedding instead of SHA-256 hash
    for the query frame, enabling fuzzy semantic recall.
    When reconstruct is True, each recalled memory's attractor is blended with
    the query attractor and re-evolved through the CA (Darman architecture).
    When salience is set, the query evolution and reconstruction use the
    corresponding attention budget (variable tick rates).
    When polar_decay is True, each polarity link that fires receives an
    incremented decay_count; results include "new_weight" in their
    polar_firing companion dict.

    Polar attractors (memory_type=="polar" or "avoidance") are never surfaced
    as independent results; they only appear as "polar_firing" companions.
    """
    d = _get_data_dir(data_dir)
    budget = compute_attention_budget(
        salience if salience is not None else SALIENCE_DEFAULT
    )

    if chunk is not None:
        chunks_to_search = [chunk]
    else:
        chunks_to_search = select_recall_chunks(text)
        # Also include any existing chunks not already selected
        existing = list_existing_chunks(d)
        for c in existing:
            if c not in chunks_to_search:
                chunks_to_search.append(c)

    if query_frame is None:
        from .rotation import _get_frame_fn

        frame_fn = _get_frame_fn(use_embedding, encoder=encoder)
        query_frame = frame_fn(text)
    query_result = evolve_and_interpret(
        query_frame,
        max_iters=budget.max_iters,
        stability_threshold=budget.stability_threshold,
    )
    query_flat = query_result["attractor"].flatten()

    # Collect work items across all chunks, then score in parallel
    work_items = []  # (hex_key, meta, attractor_path, warmth_entry, chunk_name)
    for c in chunks_to_search:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        if not index:
            continue

        touch_chunk_metadata(chunk_dir)
        warmth_data = load_warmth(chunk_dir)

        for hex_key, meta in index.items():
            if meta.get("metadata", {}).get("memory_type") in ("avoidance", "polar"):
                continue
            # Default recall skips experiential entries (use interference=True for those)
            if meta.get("grid") == "experiential":
                continue
            attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if not attractor_path.exists():
                continue
            ensure_access_fields(meta, meta["timestamp"])
            w = warmth_data.get(hex_key, {})
            work_items.append((hex_key, meta, attractor_path, w, c))

    # Pre-compute query norm for fast Pearson
    query_mean = query_flat.mean()
    query_centered = query_flat - query_mean
    query_std = query_flat.std()

    def _score_item(item):
        hex_key, meta, attractor_path, w, chunk_name = item
        attractor = np.load(attractor_path, mmap_mode="r")
        if attractor.shape != (64, 64):
            return None
        att_flat = attractor.flatten()
        md = meta.get("metadata", {})
        att_mean = md.get("att_mean")
        att_std = md.get("att_std")
        try:
            if (
                att_mean is not None
                and att_std is not None
                and att_std > 0
                and query_std > 0
            ):
                att_centered = att_flat - att_mean
                sim = float(
                    np.dot(query_centered, att_centered)
                    / (len(query_flat) * query_std * att_std)
                )
            else:
                corr, _ = pearsonr(query_flat, att_flat)
                sim = float(corr)
        except Exception:
            return None
        if np.isnan(sim):
            sim = 0.0
        temp = effective_temperature(
            meta["metadata"]["hit_count"],
            meta["metadata"]["last_accessed"],
            warmth_boost=w.get("boost", 0.0),
            warmth_applied_at=w.get("applied_at"),
        )
        tier = temperature_tier(temp)
        effective = sim + temperature_boost * temp
        return {
            "hex_key": hex_key,
            "text": meta["text"],
            "similarity": sim,
            "temperature": temp,
            "temperature_tier": tier,
            "effective_similarity": effective,
            "state": meta["state"],
            "convergence_ticks": meta["convergence_ticks"],
            "timestamp": meta["timestamp"],
            "chunk": chunk_name,
        }

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = [r for r in executor.map(_score_item, work_items) if r is not None]

    top_results = heapq.nlargest(
        top_k,
        results,
        key=lambda r: r["effective_similarity"],
    )

    # Lazy feature extraction: only compute for top_k results
    for r in top_results:
        chunk_dir = d / "chunks" / r["chunk"]
        att_path = chunk_dir / "attractors" / f"{r['hex_key']}.npy"
        att = np.load(att_path, mmap_mode="r")
        feats = compute_attractor_features(att)
        r["grid_entropy"] = feats["grid_entropy"]
        r["cluster_count"] = feats["cluster_count"]
        r["alive_fraction"] = feats["alive_fraction"]

    # Polar companion injection: batch by chunk to minimise disk reads.
    # For top_k results all in the same chunk this reduces 5–10 reads to 1–2.
    from .polarity import apply_polar_decay_in_place, get_polar_companion_from_assoc

    by_chunk: dict[str, list] = {}
    for r in top_results:
        by_chunk.setdefault(r["chunk"], []).append(r)
    for chunk_name, chunk_results in by_chunk.items():
        result_chunk_dir = d / "chunks" / chunk_name
        assoc = _load_associations(result_chunk_dir)
        index = _load_index(result_chunk_dir)
        assoc_dirty = False
        for r in chunk_results:
            companion = get_polar_companion_from_assoc(r["hex_key"], assoc, index)
            if companion is not None:
                r["polar_firing"] = companion
                if polar_decay:
                    new_weight = apply_polar_decay_in_place(r["hex_key"], assoc)
                    r["polar_firing"]["new_weight"] = new_weight
                    assoc_dirty = True
        if assoc_dirty:
            _save_associations(result_chunk_dir, assoc)

    # Reconstructive recall: blend each result with query context
    if reconstruct and top_results:
        from .reconstruction import reconstruct as _reconstruct

        query_att = query_result["attractor"]
        for r in top_results:
            # Load the stored attractor for reconstruction
            chunk_dir = d / "chunks" / r["chunk"]
            att_path = chunk_dir / "attractors" / f"{r['hex_key']}.npy"
            stored_att = np.load(att_path)
            # Derive reconstruction salience from explicit or temperature
            recon_salience = (
                salience
                if salience is not None
                else salience_from_temperature(r["temperature"])
            )
            recon_budget = compute_attention_budget(recon_salience)
            recon = _reconstruct(
                stored_att,
                query_att,
                alpha=reconstruct_alpha,
                max_iters=recon_budget.max_iters,
                stability_threshold=recon_budget.stability_threshold,
            )
            r["reconstructed_attractor"] = recon["attractor"]
            r["reconstruction_state"] = recon["state"]
            r["reconstruction_ticks"] = recon["convergence_ticks"]
            r["reconstruction_alpha"] = recon["alpha"]
            r["correlation_with_stored"] = recon["correlation_with_stored"]
            r["correlation_with_query"] = recon["correlation_with_query"]

    if not readonly:
        _bump_recalled_memories(d, top_results)

    return top_results


def _bump_recalled_memories(data_dir: Path, results: list[dict]) -> None:
    """Increment hit_count and update last_accessed for recalled memories."""
    # Group results by chunk to minimise index loads
    by_chunk: dict[str, list[str]] = {}
    for r in results:
        by_chunk.setdefault(r["chunk"], []).append(r["hex_key"])

    for chunk_name, hex_keys in by_chunk.items():
        chunk_dir = data_dir / "chunks" / chunk_name
        lock_path = chunk_dir / "index.json.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            index = _load_index(chunk_dir)
            changed = False
            for hk in hex_keys:
                if hk in index:
                    bump_access(index[hk])
                    changed = True
            if changed:
                _save_index(chunk_dir, index)
        propagate_warmth(chunk_dir, hex_keys)
        if len(hex_keys) >= 2:
            build_co_recall_associations(chunk_dir, hex_keys)

    # Cross-chunk warmth: build edges and propagate across chunk boundaries
    if len(by_chunk) >= 2:
        build_cross_chunk_co_recall(data_dir, results)
        propagate_warmth_cross_chunk(data_dir, results)


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
        warmth_data = load_warmth(chunk_dir)
        for k, v in index.items():
            ensure_access_fields(v, v["timestamp"])
            w = warmth_data.get(k, {})
            temp = effective_temperature(
                v["metadata"]["hit_count"],
                v["metadata"]["last_accessed"],
                warmth_boost=w.get("boost", 0.0),
                warmth_applied_at=w.get("applied_at"),
            )
            all_memories.append(
                {
                    "hex_key": k,
                    "chunk": c,
                    "temperature": temp,
                    "temperature_tier": temperature_tier(temp),
                    **v,
                }
            )

    return all_memories

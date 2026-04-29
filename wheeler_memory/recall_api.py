"""Two-tier recall API: recognize / reconstruct.

`recognize(query)` returns a `BasinSeed` (basin coordinates + depth + SCM
stability snapshot) without engaging the CA convergence loop.
`reconstruct(seed, query)` warm-starts CA settling from the seed when content
(not just identity) is needed.

This module wraps the existing recall machinery in storage.py without
modifying it. Per-basin Temporal Stability (T) is read from each chunk's
index.json metadata via t_metadata.ensure_t_fields, and updated with an
EMA when learning_enabled=True.
"""

from __future__ import annotations

import fcntl
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .chunking import list_existing_chunks, select_recall_chunks, touch_chunk_metadata
from .constants import (
    BASIN_DRIFT_BASE_RATE,
    RECOGNITION_THRESHOLD,
    SALIENCE_DEFAULT,
    T_EMA_RATE,
)
from .dynamics import (
    apply_ca_dynamics,
    compute_attractor_features,
    evolve_and_interpret,
)
from .t_metadata import ensure_t_fields, update_t_stability
from .temperature import (
    bump_access,
    effective_temperature,
    ensure_access_fields,
    temperature_tier,
)
from .warming import load_warmth


DEFAULT_DATA_DIR = Path.home() / ".wheeler_memory"

_LEARNING_ENABLED_DEFAULT = False


def set_learning_enabled(enabled: bool) -> None:
    """Set the module-level default for learning_enabled.

    Per-call args still take precedence. The default is False — no writes
    to stored attractors or T metadata happen unless explicitly enabled.
    """
    global _LEARNING_ENABLED_DEFAULT
    _LEARNING_ENABLED_DEFAULT = bool(enabled)


@dataclass(frozen=True)
class BasinSeed:
    """Identity-only handle to a recognized basin.

    Returned by recognize(). Sufficient to look up the stored attractor and
    re-warm the CA without re-running it. Does NOT contain the reconstructed
    pattern — call reconstruct(seed) for that.
    """

    hex_key: str
    chunk: str
    similarity: float          # Pearson(query_frame, stored_attractor)
    basin_depth: float         # 'energy' from compute_attractor_features (mean |delta| after 1 step)
    temperature: float
    temperature_tier: str
    t_stability: float         # per-basin Temporal Stability T in [0, 1]
    text: str
    data_dir: Path
    basin_stability: float     # per-basin stability reading in [0,1]; 1 - p99(|one_step - stored|)/2


@dataclass
class Pattern:
    """Reconstructed memory pattern returned by reconstruct(seed)."""

    attractor: np.ndarray
    state: str
    convergence_ticks: int
    correlation_with_stored: float
    correlation_with_query: float
    seed: BasinSeed
    text: str


def _get_data_dir(data_dir: str | Path | None) -> Path:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_chunks(d: Path, text: str, chunk: str | None) -> list[str]:
    if chunk is not None:
        return [chunk]
    chunks = select_recall_chunks(text)
    for c in list_existing_chunks(d):
        if c not in chunks:
            chunks.append(c)
    return chunks


def _frame_fn(use_embedding: bool, encoder: str | None):
    from .rotation import _get_frame_fn

    return _get_frame_fn(use_embedding, encoder=encoder)


def _load_index(chunk_dir: Path) -> dict:
    from .cache import cached_load_json

    return cached_load_json(chunk_dir / "index.json", default={})


def _save_index_with_lock(chunk_dir: Path, index: dict) -> None:
    """Atomic write of index.json with cache invalidation.

    Caller is responsible for holding the chunk's index.json.lock file lock
    via fcntl. Mirrors storage._save_index without importing it.
    """
    from .cache import invalidate

    index_path = chunk_dir / "index.json"
    tmp_path = index_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(index, indent=2))
    tmp_path.replace(index_path)
    invalidate(index_path)


def _basin_stability(stored: np.ndarray, one_step: np.ndarray) -> float:
    """Per-basin stability reading in [0, 1].

    Mirrors the CA's own convergence detection (dynamics.py uses p99 of
    |delta| against a small threshold). Using max() here — as cortex_scm.
    score_energy does — saturates to 0 for converged attractors with sharp
    ±1 boundaries that flip a few cells per CA step. p99 / 2 normalises
    for the [-1, 1] value range so the result is bounded in [0, 1].
    """
    delta = np.abs(one_step.flatten() - stored.flatten())
    p99 = float(np.percentile(delta, 99))
    return float(np.clip(1.0 - p99 / 2.0, 0.0, 1.0))


def _pearson_fast(
    query_flat: np.ndarray,
    query_centered: np.ndarray,
    query_std: float,
    att_flat: np.ndarray,
    att_mean: float | None,
    att_std: float | None,
) -> float:
    n = len(att_flat)
    if att_mean is not None and att_std is not None and att_std > 0 and query_std > 0:
        att_centered = att_flat - att_mean
        return float(np.dot(query_centered, att_centered) / (n * query_std * att_std))
    try:
        corr, _ = pearsonr(query_flat, att_flat)
    except Exception:
        return 0.0
    return float(corr) if not np.isnan(corr) else 0.0


def _scan_chunks(
    d: Path,
    chunks_to_search: list[str],
    query_frame: np.ndarray,
) -> list[tuple[float, str, str, dict]]:
    """Return [(similarity, hex_key, chunk_name, meta), ...] over all candidates.

    Skips polar/avoidance/experiential entries to mirror recall_memory.
    Does NOT call evolve_and_interpret. Pearson is computed against the raw
    query frame, not an evolved query attractor.
    """
    query_flat = query_frame.flatten()
    query_mean = query_flat.mean()
    query_centered = query_flat - query_mean
    query_std = query_flat.std()

    scored: list[tuple[float, str, str, dict]] = []
    for c in chunks_to_search:
        chunk_dir = d / "chunks" / c
        if not chunk_dir.exists():
            continue
        index = _load_index(chunk_dir)
        if not index:
            continue
        touch_chunk_metadata(chunk_dir)
        for hex_key, meta in index.items():
            md = meta.get("metadata", {})
            if md.get("memory_type") in ("avoidance", "polar"):
                continue
            if meta.get("grid") == "experiential":
                continue
            attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
            if not attractor_path.exists():
                continue
            try:
                attractor = np.load(attractor_path, mmap_mode="r")
            except Exception:
                continue
            if attractor.shape != (64, 64):
                continue
            sim = _pearson_fast(
                query_flat,
                query_centered,
                query_std,
                attractor.flatten(),
                md.get("att_mean"),
                md.get("att_std"),
            )
            if np.isnan(sim):
                sim = 0.0
            scored.append((sim, hex_key, c, meta))
    return scored


def _build_seed(
    hex_key: str,
    chunk: str,
    sim: float,
    meta: dict,
    chunk_dir: Path,
    data_dir: Path,
    warmth_entry: dict,
) -> BasinSeed:
    """Construct a BasinSeed for a recognized basin. Includes a 1-step CA probe.

    The probe runs apply_ca_dynamics once on the stored attractor and feeds
    both states to score_energy. This is the per-recall stability reading.
    It is NOT evolve_and_interpret — there is no convergence loop.
    """
    ensure_t_fields(meta)
    ensure_access_fields(meta, meta["timestamp"])
    md = meta["metadata"]

    attractor_path = chunk_dir / "attractors" / f"{hex_key}.npy"
    stored = np.load(attractor_path)

    # 1-step CA probe for the per-basin stability reading. Constant 1 tick.
    one_step = apply_ca_dynamics(stored)
    stability = _basin_stability(stored, one_step)

    basin_depth = float(md.get("energy", compute_attractor_features(stored)["energy"]))
    temp = effective_temperature(
        md["hit_count"],
        md["last_accessed"],
        warmth_boost=warmth_entry.get("boost", 0.0),
        warmth_applied_at=warmth_entry.get("applied_at"),
    )

    return BasinSeed(
        hex_key=hex_key,
        chunk=chunk,
        similarity=float(sim),
        basin_depth=basin_depth,
        temperature=float(temp),
        temperature_tier=temperature_tier(temp),
        t_stability=float(md["t_stability"]),
        text=meta["text"],
        data_dir=data_dir,
        basin_stability=float(stability),
    )


def _apply_recall_learning(
    chunk_dir: Path,
    hex_key: str,
    query_frame: np.ndarray,
) -> None:
    """Update T (EMA) and drift the stored basin in-place. Bumps hit_count.

    Holds the chunk's index.json.lock for the entire transaction. Re-reads
    the index inside the lock so the EMA is computed against the freshest T.
    """
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
        observed_stability = _basin_stability(stored, one_step)

        old_t = float(entry["metadata"]["t_stability"])
        new_t = update_t_stability(old_t, observed_stability, T_EMA_RATE)
        entry["metadata"]["t_stability"] = new_t
        entry["metadata"]["t_recall_count"] = int(entry["metadata"].get("t_recall_count", 0)) + 1

        # Drift: observed = query_frame after one CA step. Captures direction
        # of pull from the query without paying for a full evolve.
        observed = apply_ca_dynamics(query_frame.astype(np.float32))
        plasticity = (1.0 - old_t) * BASIN_DRIFT_BASE_RATE
        new_attractor = (stored.astype(np.float32) + plasticity * (observed - stored.astype(np.float32))).astype(stored.dtype)

        np.save(attractor_path, new_attractor)
        flat = new_attractor.flatten()
        entry["metadata"]["att_mean"] = float(flat.mean())
        entry["metadata"]["att_std"] = float(flat.std())

        bump_access(entry)
        _save_index_with_lock(chunk_dir, index)


def recognize(
    query: str,
    *,
    data_dir: str | Path | None = None,
    chunk: str | None = None,
    encoder: str | None = None,
    use_embedding: bool = False,
    threshold: float | None = None,
    learning_enabled: bool | None = None,
) -> BasinSeed | None:
    """Fast path: recognize a basin without engaging the CA grid.

    Returns the best BasinSeed if max Pearson similarity reaches `threshold`,
    else None ("recognition miss"). Never calls evolve_and_interpret.

    A single apply_ca_dynamics step IS run on the recognized basin to compute
    the SCM stability reading (scm_energy). This is constant 1-tick work, not
    a convergence loop, so recognition still skips grid engagement in the
    sense intended by the architecture.
    """
    if learning_enabled is None:
        learning_enabled = _LEARNING_ENABLED_DEFAULT
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    d = _get_data_dir(data_dir)
    chunks_to_search = _resolve_chunks(d, query, chunk)
    frame_fn = _frame_fn(use_embedding, encoder)
    query_frame = frame_fn(query)

    scored = _scan_chunks(d, chunks_to_search, query_frame)
    if not scored:
        return None

    sim, hex_key, chunk_name, meta = max(scored, key=lambda t: t[0])
    if sim < threshold:
        return None

    chunk_dir = d / "chunks" / chunk_name
    warmth = load_warmth(chunk_dir).get(hex_key, {})
    seed = _build_seed(hex_key, chunk_name, sim, meta, chunk_dir, d, warmth)

    if learning_enabled:
        _apply_recall_learning(chunk_dir, hex_key, query_frame)

    return seed


def recognize_top_k(
    query: str,
    k: int = 5,
    *,
    data_dir: str | Path | None = None,
    chunk: str | None = None,
    encoder: str | None = None,
    use_embedding: bool = False,
    threshold: float | None = None,
    learning_enabled: bool | None = None,
) -> list[BasinSeed]:
    """Like recognize() but returns up to k seeds passing the threshold.

    Drift updates apply only to the top-1 winner (consistent with recognize),
    not to all k. Threshold still gates: results below threshold are dropped.
    """
    if learning_enabled is None:
        learning_enabled = _LEARNING_ENABLED_DEFAULT
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    d = _get_data_dir(data_dir)
    chunks_to_search = _resolve_chunks(d, query, chunk)
    frame_fn = _frame_fn(use_embedding, encoder)
    query_frame = frame_fn(query)

    scored = _scan_chunks(d, chunks_to_search, query_frame)
    if not scored:
        return []

    top = heapq.nlargest(k, scored, key=lambda t: t[0])
    seeds: list[BasinSeed] = []
    for sim, hex_key, chunk_name, meta in top:
        if sim < threshold:
            break
        chunk_dir = d / "chunks" / chunk_name
        warmth = load_warmth(chunk_dir).get(hex_key, {})
        seeds.append(_build_seed(hex_key, chunk_name, sim, meta, chunk_dir, d, warmth))

    if learning_enabled and seeds:
        winner = seeds[0]
        _apply_recall_learning(d / "chunks" / winner.chunk, winner.hex_key, query_frame)

    return seeds


def reconstruct(
    seed: BasinSeed,
    query: str | None = None,
    *,
    alpha: float = 0.3,
    salience: float | None = None,
    encoder: str | None = None,
    use_embedding: bool = False,
) -> Pattern:
    """Slow path: warm-start CA settling from the seed.

    With query=None, returns the stored attractor as-is (no CA work).
    With query=str, blends the stored attractor with the *raw* query frame
    (no query evolve!) and re-evolves. This skips the cold path's query-
    evolve cost, which is the warm-start savings.
    """
    from .attention import compute_attention_budget

    chunk_dir = seed.data_dir / "chunks" / seed.chunk
    attractor_path = chunk_dir / "attractors" / f"{seed.hex_key}.npy"
    stored = np.load(attractor_path)

    if query is None:
        return Pattern(
            attractor=stored,
            state="CONVERGED",
            convergence_ticks=0,
            correlation_with_stored=1.0,
            correlation_with_query=0.0,
            seed=seed,
            text=seed.text,
        )

    frame_fn = _frame_fn(use_embedding, encoder)
    query_frame = frame_fn(query).astype(np.float32)
    stored_f32 = stored.astype(np.float32)

    blended = (1.0 - alpha) * stored_f32 + alpha * query_frame

    budget = compute_attention_budget(salience if salience is not None else SALIENCE_DEFAULT)
    result = evolve_and_interpret(
        blended,
        max_iters=budget.max_iters,
        stability_threshold=budget.stability_threshold,
    )
    reconstructed = result["attractor"]

    def _pearson(a, b):
        a = a.flatten()
        b = b.flatten()
        a = a - a.mean()
        b = b - b.mean()
        norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / norm) if norm > 0 else 0.0

    return Pattern(
        attractor=reconstructed,
        state=result["state"],
        convergence_ticks=int(result["convergence_ticks"]),
        correlation_with_stored=_pearson(reconstructed, stored_f32),
        correlation_with_query=_pearson(reconstructed, query_frame),
        seed=seed,
        text=seed.text,
    )

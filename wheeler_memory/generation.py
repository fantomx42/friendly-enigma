"""Generative engine: IT from BIT.

Tristan Wheeler's thesis: language generation emerges from CA information dynamics.
Stored attractors ARE the model. The CA trajectory IS the forward pass.
Each tick = one BIT decision. The sequence of resonances across ticks = the IT (output).

Usage:
    from wheeler_memory.generation import trajectory_resonance
    result = trajectory_resonance("GPU debugging", chunk="code")
    for text in result.sequence:
        print(text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .attention import compute_attention_budget, salience_from_label
from .chunking import get_chunk_dir, list_existing_chunks, select_recall_chunks
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame
from .temperature import SALIENCE_DEFAULT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TickResult:
    """Per-tick resonance data."""
    tick: int
    peak_text: str
    peak_sim: float
    emitted: bool


@dataclass
class GenerationResult:
    """Full output from trajectory_resonance."""
    query: str
    query_attractor: np.ndarray
    convergence_state: str
    convergence_ticks: int
    sequence: list[str]          # deduplicated output sequence (texts)
    ticks: list[TickResult]      # per-tick data for introspection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_chunk_attractors(chunk_dir: Path) -> dict[str, dict]:
    """Load all attractors from a chunk directory.

    Returns a dict keyed by hex_key, value is:
      {"text": str, "flat": np.ndarray, "mean": float, "std": float}
    Only entries with existing .npy files and valid shapes are included.
    """
    from .cache import cached_load_json

    index_path = chunk_dir / "index.json"
    index = cached_load_json(index_path, default={})
    if not index:
        return {}

    attractors = {}
    for hex_key, meta in index.items():
        # Skip polar/avoidance entries
        if meta.get("metadata", {}).get("memory_type") in ("avoidance", "polar"):
            continue
        att_path = chunk_dir / "attractors" / f"{hex_key}.npy"
        if not att_path.exists():
            continue
        try:
            att = np.load(att_path, mmap_mode='r')
            if att.shape != (64, 64):
                continue
            flat = att.flatten().copy()  # copy so we own the data
            md = meta.get("metadata", {})
            att_mean = md.get("att_mean", float(flat.mean()))
            att_std = md.get("att_std", float(flat.std()))
            attractors[hex_key] = {
                "text": meta["text"],
                "flat": flat,
                "mean": att_mean,
                "std": att_std,
            }
        except Exception:
            continue

    return attractors


def _fast_pearson(
    frame_flat: np.ndarray,
    frame_centered: np.ndarray,
    frame_std: float,
    att_flat: np.ndarray,
    att_mean: float,
    att_std: float,
) -> float:
    """Fast Pearson correlation reusing cached attractor mean/std."""
    if att_std <= 0 or frame_std <= 0:
        return 0.0
    att_centered = att_flat - att_mean
    sim = float(
        np.dot(frame_centered, att_centered) / (len(frame_flat) * frame_std * att_std)
    )
    return 0.0 if np.isnan(sim) else sim


def _dedup(ticks: list[TickResult], strategy: str) -> list[str]:
    """Apply deduplication strategy and return ordered list of emitted texts.

    Strategies:
      "first"  — emit on the first tick where a text appears above threshold
      "last"   — emit on the last tick of each consecutive run
      "peak"   — emit on the tick where sim is max within each consecutive run
      "none"   — emit all ticks above threshold (no dedup)
    """
    if strategy == "none":
        return [t.peak_text for t in ticks if t.emitted]

    # Group consecutive runs of the same text
    runs: list[list[TickResult]] = []
    current_text: str | None = None
    current_run: list[TickResult] = []

    for t in ticks:
        if not t.emitted:
            if current_run:
                runs.append(current_run)
                current_run = []
                current_text = None
            continue
        if t.peak_text == current_text:
            current_run.append(t)
        else:
            if current_run:
                runs.append(current_run)
            current_run = [t]
            current_text = t.peak_text

    if current_run:
        runs.append(current_run)

    seen: set[str] = set()
    result: list[str] = []

    for run in runs:
        if strategy == "first":
            chosen = run[0]
        elif strategy == "last":
            chosen = run[-1]
        elif strategy == "peak":
            chosen = max(run, key=lambda t: t.peak_sim)
        else:
            chosen = run[0]

        text = chosen.peak_text
        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def trajectory_resonance(
    query_text: str,
    data_dir=None,
    chunk: str | None = None,
    top_k: int = 1,
    min_resonance: float = 0.15,
    dedup: str = "first",
    use_embedding: bool = False,
    salience: float | None = None,
) -> GenerationResult:
    """Generate a sequence by resonating the query trajectory against stored attractors.

    The query text is encoded into a 64×64 frame and evolved through the CA.
    At each tick, all stored attractors are scored by Pearson correlation with the
    current frame. Ticks above min_resonance emit the best-matching memory text.
    The sequence of emissions (after deduplication) is the generated output.

    Parameters
    ----------
    query_text:
        The query / prompt to generate from.
    data_dir:
        Root data directory (default: ~/.wheeler_memory).
    chunk:
        Restrict search to this chunk. None = auto-select + all existing chunks.
    top_k:
        How many attractors to emit per tick (default: 1 = argmax only).
    min_resonance:
        Minimum Pearson correlation to emit at a tick (default: 0.15).
    dedup:
        Deduplication strategy: "first" | "last" | "peak" | "none".
    use_embedding:
        Use sentence-transformer embedding instead of hash for query frame.
    salience:
        Attention budget salience [0,1]. None uses SALIENCE_DEFAULT (0.5).
    """
    from .storage import _get_data_dir

    d = _get_data_dir(data_dir)
    budget = compute_attention_budget(
        salience if salience is not None else SALIENCE_DEFAULT
    )

    # Determine chunks to search
    if chunk is not None:
        chunks_to_search = [chunk]
    else:
        chunks_to_search = select_recall_chunks(query_text)
        existing = list_existing_chunks(d)
        for c in existing:
            if c not in chunks_to_search:
                chunks_to_search.append(c)

    # Load all attractors into memory
    all_attractors: dict[str, dict] = {}
    for c in chunks_to_search:
        chunk_dir = d / "chunks" / c
        if chunk_dir.exists():
            all_attractors.update(_load_chunk_attractors(chunk_dir))

    # Encode query → initial frame
    if use_embedding:
        from .embedding import embed_to_frame
        query_frame = embed_to_frame(query_text)
    else:
        query_frame = hash_to_frame(query_text)

    # Evolve query frame through CA, collecting full history
    result = evolve_and_interpret(
        query_frame,
        max_iters=budget.max_iters,
        stability_threshold=budget.stability_threshold,
    )
    history: list[np.ndarray] = result["history"]

    tick_results: list[TickResult] = []

    if not all_attractors:
        # No stored memories — return empty generation
        return GenerationResult(
            query=query_text,
            query_attractor=result["attractor"],
            convergence_state=result["state"],
            convergence_ticks=result["convergence_ticks"],
            sequence=[],
            ticks=[],
        )

    # Pre-flatten attractor arrays for vectorised scoring
    att_keys = list(all_attractors.keys())
    att_matrix = np.stack([all_attractors[k]["flat"] for k in att_keys])  # (N, 4096)
    att_means = np.array([all_attractors[k]["mean"] for k in att_keys])   # (N,)
    att_stds = np.array([all_attractors[k]["std"] for k in att_keys])     # (N,)

    # Score each frame in history
    for tick_i, frame in enumerate(history):
        frame_flat = frame.flatten()
        frame_mean = float(frame_flat.mean())
        frame_std = float(frame_flat.std())

        if frame_std <= 0:
            tick_results.append(TickResult(tick=tick_i, peak_text="", peak_sim=0.0, emitted=False))
            continue

        frame_centered = frame_flat - frame_mean

        # Vectorised Pearson: sim[j] = dot(frame_c, att_c[j]) / (n * std_f * std_a[j])
        att_centered = att_matrix - att_means[:, None]  # (N, 4096)
        valid_mask = att_stds > 0
        sims = np.zeros(len(att_keys))
        if valid_mask.any():
            dots = att_centered[valid_mask] @ frame_centered  # (K,)
            sims[valid_mask] = dots / (len(frame_flat) * frame_std * att_stds[valid_mask])
        # NaN guard
        sims = np.where(np.isnan(sims), 0.0, sims)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_text = all_attractors[att_keys[best_idx]]["text"]

        emitted = best_sim >= min_resonance
        tick_results.append(TickResult(
            tick=tick_i,
            peak_text=best_text,
            peak_sim=best_sim,
            emitted=emitted,
        ))

    sequence = _dedup(tick_results, dedup)

    return GenerationResult(
        query=query_text,
        query_attractor=result["attractor"],
        convergence_state=result["state"],
        convergence_ticks=result["convergence_ticks"],
        sequence=sequence,
        ticks=tick_results,
    )

"""Corpus crystallization pipeline for Wheeler Memory pre-training.

Feeds a text corpus through Wheeler's encode → evolve → store pipeline at
scale, forming an attractor landscape before deployment.  This is the
offline equivalent of runtime memory formation — the system boots with
crystallized knowledge rather than starting empty.

Supported corpus formats:
  - JSONL: one JSON object per line with a "text" field
  - CSV:   a column named "text"
  - TXT:   one text per line

Usage
-----
>>> from wheeler_memory.crystallization import crystallize
>>> result = crystallize(Path("corpus.jsonl"), max_items=10_000, verbose=True)
>>> print(result)

Or via CLI:
    wheeler-crystallize corpus.jsonl --verbose
    wheeler-crystallize corpus.csv --chunk science --max-items 10000
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# CPU topology — detected once at import time
# ---------------------------------------------------------------------------


def _detect_core_groups() -> tuple[list[int], list[int]]:
    """Return (p_cores, e_cores) by reading cpufreq max-freq from /sys.

    P-cores have higher max frequency than E-cores on Intel hybrid chips.
    Falls back to (all_cores, []) on non-hybrid or non-Linux systems.
    """
    try:
        freq_map: dict[int, int] = {}
        n = os.cpu_count() or 1
        for i in range(n):
            p = f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq"
            try:
                freq_map[i] = int(open(p).read().strip())
            except OSError:
                freq_map[i] = 0
        if not freq_map:
            return list(range(n)), []
        freqs = sorted(set(freq_map.values()), reverse=True)
        if len(freqs) < 2:
            return list(freq_map), []
        # Highest-frequency group = P-cores; rest = E-cores
        top = freqs[0]
        p_cores = [c for c, f in freq_map.items() if f == top or f >= freqs[1]]
        e_cores = [c for c, f in freq_map.items() if f < freqs[1]]
        # If split is extreme (>3:1 E:P), use E-cores for embedding (cluster L2)
        return p_cores, e_cores
    except Exception:
        n = os.cpu_count() or 1
        return list(range(n)), []


_P_CORES, _E_CORES = _detect_core_groups()
# Embedding benefits from E-core cluster L2 cache; use E-cores when available
_EMBED_CORES: list[int] = _E_CORES if _E_CORES else _P_CORES
_IO_CORES: list[int] = _E_CORES if _E_CORES else _P_CORES


def _pin(cpus: list[int]) -> None:
    """Pin current thread to the given CPU list (Linux only, silently ignored elsewhere)."""
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError):
        pass


def _configure_torch_threads(n: int) -> None:
    """Set PyTorch intra-op thread count (silently ignored if torch unavailable)."""
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


from .brick import MemoryBrick
from .chunking import get_chunk_dir, list_existing_chunks, select_chunk
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .storage import _get_data_dir, _load_index, batch_store_memories
from .temperature import MAX_ATTRACTORS


# ── Corpus loading ────────────────────────────────────────────────────────────


def load_corpus(path: Path, fmt: str = "auto") -> Iterator[str]:
    """Stream texts from a corpus file.

    Parameters
    ----------
    path : Path
        Path to the corpus file.
    fmt : str
        Format hint: "jsonl", "csv", "txt", "parquet", or "auto" (detect from extension).

    Yields
    ------
    str
        Individual text entries, stripped of surrounding whitespace.
    """
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            fmt = "jsonl"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix == ".parquet":
            fmt = "parquet"
        else:
            fmt = "txt"

    if fmt == "jsonl":
        yield from _load_jsonl(path)
    elif fmt == "csv":
        yield from _load_csv(path)
    elif fmt == "parquet":
        yield from _load_parquet(path)
    else:
        yield from _load_txt(path)


def _load_jsonl(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if text:
                yield text


def _load_csv(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                yield text


def _load_txt(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def _load_parquet(path: Path) -> Iterator[str]:
    import pandas as pd

    df = pd.read_parquet(path)
    # Try common text column names
    for col in ("text", "prompt", "problem_statement", "description", "input"):
        if col in df.columns:
            for text in df[col].dropna():
                text = str(text).strip()
                if text:
                    yield text
            return
    raise ValueError(
        f"No text column found in {path}. "
        f"Columns: {list(df.columns)}. "
        f"Expected one of: text, prompt, problem_statement, description, input"
    )


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CrystallizationResult:
    """Summary of a crystallization run."""

    stored: int = 0
    skipped: int = 0
    errors: int = 0
    saturation_pct: float = 0.0
    elapsed_seconds: float = 0.0
    chunks_used: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Crystallization complete:",
            f"  stored:     {self.stored}",
            f"  skipped:    {self.skipped}",
            f"  errors:     {self.errors}",
            f"  saturation: {self.saturation_pct:.1f}%",
            f"  elapsed:    {self.elapsed_seconds:.1f}s",
        ]
        if self.chunks_used:
            lines.append("  chunks:")
            for name, count in sorted(self.chunks_used.items()):
                lines.append(f"    {name}: {count}")
        return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def _count_stored(data_dir: Path) -> int:
    """Count total stored attractors across all chunks."""
    total = 0
    for chunk_name in list_existing_chunks(data_dir):
        chunk_dir = get_chunk_dir(data_dir, chunk_name)
        index = _load_index(chunk_dir)
        total += len(index)
    return total


def _existing_keys(data_dir: Path) -> set[str]:
    """Collect all hex keys already stored across all chunks."""
    keys: set[str] = set()
    for chunk_name in list_existing_chunks(data_dir):
        chunk_dir = get_chunk_dir(data_dir, chunk_name)
        index = _load_index(chunk_dir)
        keys.update(index.keys())
    return keys


def crystallize(
    corpus_path: Path,
    data_dir: Path | None = None,
    batch_size: int = 32,
    chunk: str | None = None,
    use_embedding: bool = True,
    encoder: str | None = None,
    max_items: int | None = None,
    resume: bool = True,
    fmt: str = "auto",
    verbose: bool = False,
) -> CrystallizationResult:
    """Feed a text corpus through Wheeler to crystallize attractor landscapes.

    Parameters
    ----------
    corpus_path : Path
        Path to the corpus file (JSONL, CSV, or TXT).
    data_dir : Path, optional
        Wheeler data directory.  Defaults to ~/.wheeler_memory.
    batch_size : int
        Number of texts to process per batch (for embedding efficiency).
    chunk : str, optional
        Force all texts into this chunk.  If None, auto-routes via select_chunk().
    use_embedding : bool
        Legacy flag — if True, use embed_to_frame (semantic).  If False, use hash_to_frame.
        Prefer the ``encoder`` parameter instead.
    encoder : str, optional
        Encoder name: "hash", "hippocampus", "embedding", "language", "blended".
        Overrides ``use_embedding`` when set.
    max_items : int, optional
        Stop after processing this many texts (for validation runs).
    resume : bool
        If True, skip texts whose hex key is already stored.
    fmt : str
        Corpus format: "jsonl", "csv", "txt", or "auto".
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    CrystallizationResult
        Summary statistics of the crystallization run.
    """
    d = _get_data_dir(data_dir)
    result = CrystallizationResult()
    t0 = time.monotonic()

    # Gather existing keys for resume
    existing = _existing_keys(d) if resume else set()

    # Resolve encoder
    _resolved_encoder = (
        encoder if encoder is not None else ("embedding" if use_embedding else "hash")
    )

    # Lazy-load batch functions
    _batch_fn = None
    if _resolved_encoder == "embedding":
        from .embedding import embed_to_frame_batch

        _batch_fn = embed_to_frame_batch
    elif _resolved_encoder == "hippocampus":
        from .hippocampus import hippocampus_to_frame_batch

        _batch_fn = hippocampus_to_frame_batch
    elif _resolved_encoder in ("language", "blended"):
        from .rotation import _get_frame_fn

        _single_fn = _get_frame_fn(encoder=_resolved_encoder)
        _batch_fn = lambda texts: [_single_fn(t) for t in texts]

    if verbose:
        n_embed = len(_EMBED_CORES)
        label = "E-cores" if _E_CORES else "all-cores"
        print(
            f"  [crystallize] embed→{label}({n_embed}t)  CA→GPU  store→threads",
            file=sys.stderr,
        )

    # ── Collect all batches upfront (corpus is a stream, needs buffering) ──
    batches: list[list[str]] = []
    current: list[str] = []
    total_processed = 0

    for text in load_corpus(corpus_path, fmt=fmt):
        if max_items is not None and total_processed >= max_items:
            break
        hex_key = text_to_hex(text)
        if resume and hex_key in existing:
            result.skipped += 1
            total_processed += 1
            continue
        current.append(text)
        total_processed += 1
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    # ── Pipelined execution ────────────────────────────────────────────────
    # Stage A (embed thread, E-cores):   texts → frames
    # Stage B (main thread, GPU):        frames → ca_results
    # Stage C (store thread, E-cores):   ca_results + texts → disk
    #
    # Queues carry (texts, frames) and (texts, ca_results) between stages.
    # sentinel=None signals end-of-stream.

    SENTINEL = None
    embed_q: Queue = Queue(maxsize=2)  # A→B
    store_q: Queue = Queue(maxsize=2)  # B→C

    def _embed_worker() -> None:
        _pin(_EMBED_CORES)
        _configure_torch_threads(len(_EMBED_CORES))
        try:
            for batch in batches:
                if _batch_fn is not None:
                    frames = _batch_fn(batch)
                else:
                    frames = [hash_to_frame(t) for t in batch]
                embed_q.put((batch, frames))
        finally:
            embed_q.put(SENTINEL)

    def _store_worker() -> None:
        _pin(_IO_CORES)
        while True:
            item = store_q.get()
            if item is SENTINEL:
                break
            texts_b, ca_results_b = item
            entries = []
            for text, frame_result in zip(texts_b, ca_results_b):
                try:
                    text_chunk = chunk if chunk is not None else select_chunk(text)
                    brick = MemoryBrick.from_evolution_result(frame_result)
                    entries.append((text, frame_result, brick, text_chunk))
                except Exception as exc:
                    result.errors += 1
                    if verbose:
                        print(
                            f"  error preparing '{text[:50]}...': {exc}",
                            file=sys.stderr,
                        )
            try:
                n = batch_store_memories(entries, data_dir=data_dir)
                result.stored += n
                for _, _, _, text_chunk in entries[:n]:
                    result.chunks_used[text_chunk] = (
                        result.chunks_used.get(text_chunk, 0) + 1
                    )
                if verbose and result.stored % 100 < len(entries):
                    milestone = (result.stored // 100) * 100
                    print(f"  crystallized {milestone} memories...", file=sys.stderr)
            except Exception as exc:
                result.errors += len(entries)
                if verbose:
                    print(f"  error in batch store: {exc}", file=sys.stderr)

    embed_thread = Thread(target=_embed_worker, daemon=True)
    store_thread = Thread(target=_store_worker, daemon=True)
    embed_thread.start()
    store_thread.start()

    # Main thread: CA evolution on GPU (stage B)
    _pin(_P_CORES)  # P-cores for GPU dispatch / HIP overhead
    while True:
        item = embed_q.get()
        if item is SENTINEL:
            break
        texts_b, frames = item
        ca_results = [evolve_and_interpret(f) for f in frames]
        store_q.put((texts_b, ca_results))
    store_q.put(SENTINEL)

    embed_thread.join()
    store_thread.join()

    # Restore full affinity
    _pin(list(range(os.cpu_count() or 20)))

    # Final stats
    result.elapsed_seconds = time.monotonic() - t0
    total_stored = _count_stored(d)
    result.saturation_pct = (total_stored / MAX_ATTRACTORS) * 100.0

    if verbose:
        print(result, file=sys.stderr)

    return result

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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .brick import MemoryBrick
from .chunking import get_chunk_dir, list_existing_chunks, select_chunk
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .storage import _get_data_dir, _load_index, store_memory
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
        If True, use embed_to_frame (semantic).  If False, use hash_to_frame.
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

    # Lazy-load embedding if needed
    if use_embedding:
        from .embedding import embed_to_frame_batch

    # Collect texts into batches
    batch: list[str] = []
    total_processed = 0

    for text in load_corpus(corpus_path, fmt=fmt):
        if max_items is not None and total_processed >= max_items:
            break

        hex_key = text_to_hex(text)
        if resume and hex_key in existing:
            result.skipped += 1
            total_processed += 1
            continue

        batch.append(text)
        total_processed += 1

        if len(batch) >= batch_size:
            _process_batch(batch, d, chunk, use_embedding, result, verbose)
            batch.clear()

    # Process remaining
    if batch:
        _process_batch(batch, d, chunk, use_embedding, result, verbose)

    # Final stats
    result.elapsed_seconds = time.monotonic() - t0
    total_stored = _count_stored(d)
    result.saturation_pct = (total_stored / MAX_ATTRACTORS) * 100.0

    if verbose:
        print(result, file=sys.stderr)

    return result


def _process_batch(
    texts: list[str],
    data_dir: Path,
    chunk: str | None,
    use_embedding: bool,
    result: CrystallizationResult,
    verbose: bool,
) -> None:
    """Process a single batch of texts through the crystallization pipeline."""
    # 1. Generate frames
    if use_embedding:
        from .embedding import embed_to_frame_batch

        frames = embed_to_frame_batch(texts)
    else:
        frames = [hash_to_frame(t) for t in texts]

    # 2. Evolve each frame through CA
    # Use evolve_and_interpret which handles GPU dispatch + history synthesis
    results = [evolve_and_interpret(f) for f in frames]

    # 3. Store each memory
    for text, frame_result in zip(texts, results):
        try:
            text_chunk = chunk if chunk is not None else select_chunk(text)
            brick = MemoryBrick.from_evolution_result(frame_result)
            store_memory(
                text,
                frame_result,
                brick,
                data_dir=data_dir,
                chunk=text_chunk,
                auto_evict=False,  # skip per-item eviction during bulk load
            )
            result.stored += 1
            result.chunks_used[text_chunk] = result.chunks_used.get(text_chunk, 0) + 1

            if verbose and result.stored % 100 == 0:
                print(
                    f"  crystallized {result.stored} memories...",
                    file=sys.stderr,
                )
        except Exception as exc:
            result.errors += 1
            if verbose:
                print(f"  error storing '{text[:50]}...': {exc}", file=sys.stderr)

#!/usr/bin/env python3
"""Backfill trajectory signatures for all existing stored memories.

This is a one-time batch script that computes trajectory signatures for memories
stored in Wheeler Memory. Each signature is computed by re-evolving the seed frame
(evolution_history[0]) on CPU to capture trajectory dynamics.

Signatures are saved to chunks/{chunk}/signatures/{hex_key}.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.storage import DEFAULT_DATA_DIR, list_memories
from wheeler_memory.trajectory import compute_signature, save_signature


def main() -> int:
    """Parse args and backfill all signatures."""
    parser = argparse.ArgumentParser(
        description="Backfill trajectory signatures for all stored memories"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Wheeler data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--chunk",
        type=str,
        default=None,
        help="Only backfill a specific chunk (optional)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing signatures",
    )

    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()

    # List all memories
    try:
        memories = list_memories(data_dir=data_dir, chunk=args.chunk)
    except Exception as e:
        print(f"Error listing memories: {e}", file=sys.stderr)
        return 1

    if not memories:
        print("No memories found.")
        return 0

    print(f"Found {len(memories)} memories to process.")
    successes = 0
    failures = 0
    skipped = 0

    for i, memory in enumerate(memories, 1):
        hex_key = memory["hex_key"]
        chunk = memory["chunk"]
        chunk_dir = data_dir / "chunks" / chunk

        # Check if signature already exists
        sig_path = chunk_dir / "signatures" / f"{hex_key}.npz"
        if sig_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            # Load brick
            brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
            if not brick_path.exists():
                print(
                    f"[{i}/{len(memories)}] {chunk}/{hex_key} SKIPPED (brick not found)"
                )
                failures += 1
                continue

            brick = MemoryBrick.load(brick_path)

            # Extract seed frame
            if not brick.evolution_history or len(brick.evolution_history) == 0:
                print(
                    f"[{i}/{len(memories)}] {chunk}/{hex_key} SKIPPED (no evolution history)"
                )
                failures += 1
                continue

            seed_frame = brick.evolution_history[0]
            if seed_frame.shape != (64, 64):
                print(
                    f"[{i}/{len(memories)}] {chunk}/{hex_key} SKIPPED (invalid seed shape: {seed_frame.shape})"
                )
                failures += 1
                continue

            # Compute signature
            start_time = time.time()
            signature = compute_signature(seed_frame)
            elapsed = time.time() - start_time

            # Create signatures directory if needed
            sig_path.parent.mkdir(parents=True, exist_ok=True)

            # Save signature
            save_signature(signature, sig_path)

            print(f"[{i}/{len(memories)}] {chunk}/{hex_key} computed in {elapsed:.3f}s")
            successes += 1

        except Exception as e:
            print(
                f"[{i}/{len(memories)}] {chunk}/{hex_key} ERROR: {e}",
                file=sys.stderr,
            )
            failures += 1

    # Final summary
    print()
    print("=" * 60)
    print(f"Backfill complete:")
    print(f"  Successes:  {successes}")
    print(f"  Failures:   {failures}")
    print(f"  Skipped:    {skipped}")
    print(f"  Total:      {len(memories)}")
    print("=" * 60)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

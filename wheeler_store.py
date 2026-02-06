#!/usr/bin/env python3
"""
Store iteration output in Wheeler Memory.

Persists text and its Wheeler frame representation to ~/.wheeler_memory/
with metadata for stability scoring and eviction.

Usage:
    python wheeler_store.py output_file.txt
    python wheeler_store.py output_file.txt --type iteration
    python wheeler_store.py --text "direct text to store" --type lesson
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Import from wheeler_ai_training
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "wheeler_ai_training"))

from wheeler_ai import TextCodec
from wheeler_cpu import WheelerMemoryCPU

# Storage configuration
WHEELER_MEMORY_DIR = Path.home() / ".wheeler_memory"
MEMORY_INDEX_FILE = WHEELER_MEMORY_DIR / "memory.json"
FRAMES_DIR = WHEELER_MEMORY_DIR / "frames"
CHECKPOINT_FILE = WHEELER_MEMORY_DIR / "checkpoint.pkl"

# Limits
MAX_MEMORIES = 1000
MAX_TEXT_LENGTH = 4096
CHECKPOINT_SIZE_TRIGGER = 10  # Save checkpoint after this many new memories
CHECKPOINT_TIME_TRIGGER = 300  # Save checkpoint after this many seconds


def ensure_storage_dirs():
    """Create storage directories if they don't exist."""
    WHEELER_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def load_memory_index() -> dict:
    """Load the memory index from disk."""
    if not MEMORY_INDEX_FILE.exists():
        return {"memories": [], "last_checkpoint": time.time()}
    try:
        with open(MEMORY_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"memories": [], "last_checkpoint": time.time()}


def save_memory_index(index: dict):
    """Save the memory index to disk."""
    ensure_storage_dirs()
    with open(MEMORY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def save_frame(memory_id: str, frame: np.ndarray):
    """Save a frame to disk as .npy file."""
    ensure_storage_dirs()
    frame_path = FRAMES_DIR / f"{memory_id}.npy"
    np.save(frame_path, frame.astype(np.float32))


def generate_memory_id(text: str) -> str:
    """Generate a unique ID for a memory based on content hash."""
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    timestamp_part = hex(int(time.time() * 1000))[-6:]
    return f"{content_hash}_{timestamp_part}"


def compute_eviction_score(memory: dict, now: float) -> float:
    """
    Compute eviction score for a memory (lower = more likely to evict).

    Factors:
    - Age (older = lower score)
    - Hit count (more hits = higher score)
    - Stability (more stable = higher score)
    """
    age = now - memory.get("timestamp", now)
    age_hours = age / 3600

    hits = memory.get("hits", 1)
    stability = memory.get("stability", 0.5)

    # Score formula: hits and stability boost, age penalty
    score = (hits * 0.4 + stability * 0.4) / (1 + age_hours * 0.1)

    return score


def evict_if_needed(index: dict) -> dict:
    """Evict lowest-value memories if over capacity."""
    memories = index.get("memories", [])

    if len(memories) <= MAX_MEMORIES:
        return index

    now = time.time()

    # Score all memories
    scored = [(i, compute_eviction_score(m, now)) for i, m in enumerate(memories)]
    scored.sort(key=lambda x: x[1])

    # Remove lowest scoring until under capacity
    num_to_remove = len(memories) - MAX_MEMORIES + 50  # Remove extra for buffer
    indices_to_remove = set(i for i, _ in scored[:num_to_remove])

    # Remove frames
    for i in indices_to_remove:
        mem = memories[i]
        frame_path = FRAMES_DIR / f"{mem.get('id', '')}.npy"
        if frame_path.exists():
            try:
                frame_path.unlink()
            except IOError:
                pass

    # Filter memories
    index["memories"] = [m for i, m in enumerate(memories) if i not in indices_to_remove]

    return index


def save_checkpoint(index: dict, force: bool = False):
    """Save a checkpoint of the hot buffer state."""
    now = time.time()
    last_checkpoint = index.get("last_checkpoint", 0)
    memories_since_checkpoint = index.get("memories_since_checkpoint", 0)

    should_save = (
        force
        or (now - last_checkpoint) > CHECKPOINT_TIME_TRIGGER
        or memories_since_checkpoint >= CHECKPOINT_SIZE_TRIGGER
    )

    if not should_save:
        return

    ensure_storage_dirs()

    # Save rotating checkpoint
    checkpoint_data = {
        "timestamp": now,
        "memory_count": len(index.get("memories", [])),
        "index_snapshot": index,
    }

    # Rotate checkpoints (keep 2)
    checkpoint_backup = WHEELER_MEMORY_DIR / "checkpoint_backup.pkl"
    if CHECKPOINT_FILE.exists():
        try:
            CHECKPOINT_FILE.rename(checkpoint_backup)
        except IOError:
            pass

    try:
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(checkpoint_data, f)
        index["last_checkpoint"] = now
        index["memories_since_checkpoint"] = 0
    except IOError as e:
        print(f"Warning: Failed to save checkpoint: {e}", file=sys.stderr)


def store_output(
    text: str,
    output_type: str = "iteration",
    outcome: str = "pending",
    summary: Optional[str] = None,
    errors: Optional[list] = None,
) -> str:
    """
    Store output in Wheeler Memory as a new experience.

    Returns the memory ID.
    """
    # Truncate text if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "...[truncated]"

    # Initialize Wheeler components
    width, height = 128, 128
    codec = TextCodec(width=width, height=height)
    dynamics = WheelerMemoryCPU(width=width, height=height)

    # Encode text to frame
    input_frame = codec.encode(text)

    # Run dynamics to get stable attractor
    trajectory = dynamics.run_dynamics(input_frame, max_ticks=50)
    attractor_frame = trajectory.final_frame

    # Generate memory ID
    memory_id = generate_memory_id(text)

    # Create memory entry
    memory_entry = {
        "id": memory_id,
        "text": text,
        "type": output_type,
        "outcome": outcome,
        "summary": summary,
        "errors": errors or [],
        "timestamp": time.time(),
        "hits": 1,
        "stability": 0.5,  # Initial stability, will be updated on recall
    }

    # Load index and check for duplicates
    index = load_memory_index()
    memories = index.get("memories", [])

    # Check if similar text already exists (by hash prefix)
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    for i, existing in enumerate(memories):
        existing_hash = hashlib.sha256(existing.get("text", "").encode()).hexdigest()[:16]
        if existing_hash == text_hash:
            # Update existing memory instead
            memories[i]["hits"] = existing.get("hits", 1) + 1
            memories[i]["timestamp"] = time.time()
            if outcome != "pending":
                memories[i]["outcome"] = outcome
            index["memories"] = memories
            save_memory_index(index)
            return existing.get("id", memory_id)

    # Add new memory
    memories.append(memory_entry)
    index["memories"] = memories
    index["memories_since_checkpoint"] = index.get("memories_since_checkpoint", 0) + 1

    # Save frame
    save_frame(memory_id, attractor_frame)

    # Evict if needed
    index = evict_if_needed(index)

    # Save index
    save_memory_index(index)

    # Save checkpoint if needed
    save_checkpoint(index)

    return memory_id


def store_from_file(file_path: str, output_type: str = "iteration") -> str:
    """Store contents of a file in Wheeler Memory."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    # Determine outcome from content
    outcome = "pending"
    if "<promise>COMPLETE</promise>" in text:
        outcome = "success"
    elif "ERROR" in text or "error" in text.lower():
        outcome = "failure"

    return store_output(text, output_type=output_type, outcome=outcome)


def main():
    parser = argparse.ArgumentParser(
        description="Store iteration output in Wheeler Memory"
    )
    parser.add_argument(
        "file", nargs="?",
        help="Path to file containing output to store"
    )
    parser.add_argument(
        "--text", "-t",
        help="Direct text to store (alternative to file)"
    )
    parser.add_argument(
        "--type", default="iteration",
        choices=["iteration", "input", "lesson", "error"],
        help="Type of memory (default: iteration)"
    )
    parser.add_argument(
        "--outcome", default="pending",
        choices=["success", "failure", "pending"],
        help="Outcome of this iteration (default: pending)"
    )
    parser.add_argument(
        "--summary",
        help="Optional summary of the content"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print debug information"
    )
    args = parser.parse_args()

    # Get text to store
    if args.text:
        text = args.text
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        # Read from stdin
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No content to store", file=sys.stderr)
        sys.exit(1)

    try:
        memory_id = store_output(
            text,
            output_type=args.type,
            outcome=args.outcome,
            summary=args.summary,
        )
        if args.verbose:
            print(f"Stored memory: {memory_id}", file=sys.stderr)
            print(f"Type: {args.type}", file=sys.stderr)
            print(f"Text length: {len(text)}", file=sys.stderr)
        print(memory_id)
    except Exception as e:
        print(f"Error storing memory: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

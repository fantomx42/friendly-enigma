"""CLI: Store text as a Wheeler Memory attractor.

Usage:
    wheeler-store "some text to remember"
    echo "piped text" | wheeler-store -
    wheeler-store --chunk code "fix the bug"
"""

import argparse
import sys

from wheeler_memory import store_with_rotation_retry
from wheeler_memory.chunking import select_chunk


def _embed_store(text, chunk, data_dir):
    """Store using embedding-based frame generation."""
    import time
    from wheeler_memory.embedding import embed_to_frame
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.storage import store_memory

    start = time.time()
    frame = embed_to_frame(text)
    result = evolve_and_interpret(frame)
    wall_time = time.time() - start

    result["metadata"]["rotation_used"] = 0
    result["metadata"]["attempts"] = 1
    result["metadata"]["wall_time_seconds"] = wall_time
    result["metadata"]["frame_mode"] = "embedding"

    if result["state"] == "CONVERGED":
        brick = MemoryBrick.from_evolution_result(result, {"input_text": text})
        store_memory(text, result, brick, data_dir, chunk=chunk)

    return result


def main():
    parser = argparse.ArgumentParser(description="Store text as Wheeler Memory")
    parser.add_argument("text", help="Text to store (use '-' for stdin)")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Target chunk (default: auto-route)")
    parser.add_argument("--embed", action="store_true", help="Use sentence embedding instead of SHA-256 hash")
    args = parser.parse_args()

    text = sys.stdin.read().strip() if args.text == "-" else args.text

    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    auto = args.chunk is None
    chunk = args.chunk if args.chunk else select_chunk(text)

    if args.embed:
        result = _embed_store(text, chunk, args.data_dir)
    else:
        result = store_with_rotation_retry(text, data_dir=args.data_dir, chunk=chunk)

    state = result["state"]
    ticks = result["convergence_ticks"]
    angle = result["metadata"].get("rotation_used", 0)
    attempts = result["metadata"].get("attempts", 1)
    wall = result["metadata"].get("wall_time_seconds", 0)

    chunk_label = f"{chunk} (auto)" if auto else chunk
    print(f"Chunk:    {chunk_label}")
    print(f"State:    {state}")
    print(f"Ticks:    {ticks}")
    print(f"Rotation: {angle}° (attempt {attempts})")
    print(f"Time:     {wall:.3f}s")

    if state == "CONVERGED":
        print("Memory stored successfully.")
    elif state == "FAILED_ALL_ROTATIONS":
        print("Warning: failed to converge on all rotations.", file=sys.stderr)


if __name__ == "__main__":
    main()

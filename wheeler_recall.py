#!/usr/bin/env python3
"""
Recall relevant context from Wheeler Memory with stability scoring.

Stability scoring weights memories by:
- hit_count (40%): Activation frequency (sigmoid normalized)
- frame_persistence (30%): Re-encode matches stored frame
- compression_survival (30%): Pattern survives dynamics compression

Usage:
    python wheeler_recall.py "query text"
    python wheeler_recall.py "query text" --top-k 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Import from wheeler_ai_training
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "wheeler_ai_training"))

from wheeler_ai import WheelerAI, TextCodec, KnowledgeStore, Memory
from wheeler_cpu import WheelerMemoryCPU

# Default storage location
WHEELER_MEMORY_DIR = Path.home() / ".wheeler_memory"
MEMORY_INDEX_FILE = WHEELER_MEMORY_DIR / "memory.json"
FRAMES_DIR = WHEELER_MEMORY_DIR / "frames"


def sigmoid(x: float) -> float:
    """Sigmoid function for normalizing counts."""
    return 1 / (1 + math.exp(-x))


def load_memory_index() -> dict:
    """Load the memory index from disk."""
    if not MEMORY_INDEX_FILE.exists():
        return {"memories": []}
    try:
        with open(MEMORY_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"memories": []}


def load_frame(memory_id: str) -> np.ndarray | None:
    """Load a frame from disk."""
    frame_path = FRAMES_DIR / f"{memory_id}.npy"
    if not frame_path.exists():
        return None
    try:
        return np.load(frame_path)
    except IOError:
        return None


def compute_stability_score(
    query_frame: np.ndarray,
    memory: dict,
    stored_frame: np.ndarray,
    dynamics: WheelerMemoryCPU,
    codec: TextCodec,
) -> float:
    """
    Compute stability score for a memory (0.0 to 1.0).

    Components:
    - hit_count (40%): sigmoid(0.1 * hits)
    - frame_persistence (30%): correlation between re-encoded text and stored frame
    - compression_survival (30%): correlation after 10 ticks of dynamics
    """
    # 1. Hit count score (40%)
    hits = memory.get("hits", 1)
    hit_score = sigmoid(0.1 * hits) - 0.5  # Shift to center around 0.5
    hit_score = max(0.0, min(1.0, hit_score * 2))  # Normalize to 0-1

    # 2. Frame persistence score (30%)
    # Re-encode the text and compare with stored frame
    text = memory.get("text", "")
    if text:
        re_encoded = codec.encode(text)
        # Correlation between re-encoded and stored
        correlation = np.sum(re_encoded * stored_frame) / (re_encoded.size + 1e-8)
        persistence_score = (correlation + 1) / 2  # Normalize to 0-1
    else:
        persistence_score = 0.5

    # 3. Compression survival score (30%)
    # Run 10 ticks of dynamics on stored frame, measure how much survives
    trajectory = dynamics.run_dynamics(stored_frame, max_ticks=10)
    compressed = trajectory.final_frame
    # Correlation between original and compressed
    survival_corr = np.sum(stored_frame * compressed) / (stored_frame.size + 1e-8)
    survival_score = (survival_corr + 1) / 2  # Normalize to 0-1

    # Combine with weights: 40% hit, 30% persistence, 30% survival
    stability = 0.4 * hit_score + 0.3 * persistence_score + 0.3 * survival_score

    return max(0.0, min(1.0, stability))


def recall_with_stability(query: str, top_k: int = 3) -> str:
    """
    Recall memories weighted by stability scoring.

    Returns formatted context string for injection into prompts.
    """
    index = load_memory_index()
    memories = index.get("memories", [])

    if not memories:
        return ""

    # Initialize Wheeler components
    width, height = 128, 128
    codec = TextCodec(width=width, height=height)
    dynamics = WheelerMemoryCPU(width=width, height=height)

    # Encode query
    query_frame = codec.encode(query)

    # Run dynamics on query to get attractor
    query_traj = dynamics.run_dynamics(query_frame, max_ticks=30)
    query_attractor = query_traj.final_frame

    # Score each memory
    scored_memories = []
    for mem in memories:
        mem_id = mem.get("id", "")
        stored_frame = load_frame(mem_id)

        if stored_frame is None:
            continue

        # Compute similarity to query
        similarity = np.sum(query_attractor * stored_frame) / (query_attractor.size + 1e-8)
        similarity = (similarity + 1) / 2  # Normalize to 0-1

        # Compute stability score
        stability = compute_stability_score(
            query_attractor, mem, stored_frame, dynamics, codec
        )

        # Combined score: similarity weighted by stability
        combined_score = similarity * (0.5 + 0.5 * stability)

        scored_memories.append({
            "memory": mem,
            "similarity": similarity,
            "stability": stability,
            "score": combined_score,
        })

    # Sort by combined score
    scored_memories.sort(key=lambda x: x["score"], reverse=True)

    # Format output
    if not scored_memories:
        return ""

    output_lines = []
    for i, item in enumerate(scored_memories[:top_k]):
        mem = item["memory"]
        text = mem.get("text", "")[:200]  # Truncate long texts
        mem_type = mem.get("type", "unknown")
        stability = item["stability"]
        score = item["score"]

        # Format: [type] (stability: X.XX) text
        output_lines.append(
            f"[{mem_type}] (stability: {stability:.2f}, relevance: {score:.2f}) {text}"
        )

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Recall context from Wheeler Memory with stability scoring"
    )
    parser.add_argument("query", help="Query text to find relevant memories")
    parser.add_argument(
        "--top-k", "-k", type=int, default=3,
        help="Number of memories to return (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print debug information to stderr"
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Query: {args.query}", file=sys.stderr)
        print(f"Memory dir: {WHEELER_MEMORY_DIR}", file=sys.stderr)

    result = recall_with_stability(args.query, top_k=args.top_k)

    if result:
        print(result)
    elif args.verbose:
        print("No relevant memories found", file=sys.stderr)


if __name__ == "__main__":
    main()

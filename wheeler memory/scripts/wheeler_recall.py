"""CLI: Recall similar memories by Pearson correlation.

Usage:
    wheeler-recall "query text" --top-k 5
    wheeler-recall --chunk code "python bug"
"""

import argparse

from wheeler_memory import recall_memory


def main():
    parser = argparse.ArgumentParser(description="Recall similar Wheeler memories")
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Search only this chunk (default: auto-select)")
    args = parser.parse_args()

    results = recall_memory(args.query, top_k=args.top_k, data_dir=args.data_dir, chunk=args.chunk)

    if not results:
        print("No memories stored yet.")
        return

    print(f"{'Rank':<5} {'Similarity':>10}  {'Chunk':<12} {'State':<12} {'Ticks':>5}  Text")
    print("-" * 82)
    for i, r in enumerate(results, 1):
        text_preview = r["text"][:40] + "..." if len(r["text"]) > 40 else r["text"]
        chunk_name = r.get("chunk", "?")
        print(f"{i:<5} {r['similarity']:>10.4f}  {chunk_name:<12} {r['state']:<12} {r['convergence_ticks']:>5}  {text_preview}")


if __name__ == "__main__":
    main()

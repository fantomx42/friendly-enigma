#!/usr/bin/env python3
"""
Recall relevant context from Wheeler Memory using the new core library.
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Ensure we can find the bridge
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from ralph_core.wheeler_bridge import get_bridge

async def run_recall(query: str, top_k: int = 3):
    bridge = get_bridge()
    return await bridge.recall(query, limit=top_k)

def main():
    parser = argparse.ArgumentParser(
        description="Recall context from Wheeler Memory"
    )
    parser.add_argument("query", help="Query text to find relevant memories")
    parser.add_argument(
        "--top-k", "-k", type=int, default=3,
        help="Number of memories to return (default: 3)"
    )
    args = parser.parse_args()

    result = asyncio.run(run_recall(args.query, top_k=args.top_k))

    if result:
        print(result)

if __name__ == "__main__":
    main()
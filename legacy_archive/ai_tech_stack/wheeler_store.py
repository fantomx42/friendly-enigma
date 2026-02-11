#!/usr/bin/env python3
"""
Store iteration output in Wheeler Memory using the new core library.
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Ensure we can find the bridge
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from ralph_core.wheeler_bridge import get_bridge

async def run_store(text: str, memory_type: str = "iteration"):
    bridge = get_bridge()
    return await bridge.store(text, memory_type=memory_type)

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
        help="Type of memory (default: iteration)"
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
        text = sys.stdin.read()

    if not text.strip():
        sys.exit(0)

    memory_id = asyncio.run(run_store(text, memory_type=args.type))
    print(memory_id)

if __name__ == "__main__":
    main()
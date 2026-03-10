"""wheeler-crystallize — offline corpus crystallization for Wheeler Memory.

Pre-trains Wheeler by feeding a text corpus through the encode → evolve → store
pipeline at scale, forming attractor landscapes before deployment.

Usage
-----
    wheeler-crystallize corpus.jsonl
    wheeler-crystallize corpus.csv --chunk science
    wheeler-crystallize corpus.txt --max-items 10000 --verbose
    wheeler-crystallize corpus.jsonl --batch-size 64 --no-resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wheeler_memory.crystallization import crystallize


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wheeler-crystallize",
        description="Pre-train Wheeler Memory by crystallizing a text corpus.",
    )
    p.add_argument(
        "corpus",
        type=Path,
        help="Path to corpus file (JSONL, CSV, or TXT).",
    )
    p.add_argument(
        "--chunk",
        default=None,
        help="Force all texts into this chunk (default: auto-route).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Texts per batch for embedding efficiency (default: 32).",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N texts (for validation runs).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all texts even if already stored.",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        help="Use SHA-256 hashing instead of sentence embeddings.",
    )
    p.add_argument(
        "--fmt",
        choices=["jsonl", "csv", "txt", "auto"],
        default="auto",
        help="Corpus format (default: auto-detect from extension).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Wheeler Memory data directory (default: ~/.wheeler_memory).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to stderr.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.corpus.exists():
        print(f"Error: corpus file not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    try:
        result = crystallize(
            corpus_path=args.corpus,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            chunk=args.chunk,
            use_embedding=not args.no_embed,
            max_items=args.max_items,
            resume=not args.no_resume,
            fmt=args.fmt,
            verbose=args.verbose,
        )
    except ImportError as e:
        if "sentence_transformers" in str(e):
            print(
                "Error: Embedding requires sentence-transformers.\n"
                "Install with: pip install -e '.[embed]'",
                file=sys.stderr,
            )
        else:
            print(f"Error: Missing dependency — {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.verbose:
        # Print summary even without verbose
        print(result)


if __name__ == "__main__":
    main()

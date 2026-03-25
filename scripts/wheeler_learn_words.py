"""CLI: Train word vectors from stored Wheeler memories.

Usage:
    wheeler-learn-words
    wheeler-learn-words --data-dir /path/to/data --window 5 --dim 384
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train word co-occurrence vectors from stored Wheeler memories"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Wheeler data directory (default: ~/.wheeler_memory)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Co-occurrence window size (default: 5)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Vector dimensionality (default: 384)",
    )
    args = parser.parse_args()

    try:
        from wheeler_memory.word_encoder import (
            train_word_vectors,
            save_word_vectors,
        )

        print("Training word vectors from stored memories...")

        # Train vectors
        vectors, vocab = train_word_vectors(
            data_dir=args.data_dir,
            dim=args.dim,
            window=args.window,
        )

        # Save vectors
        save_word_vectors(vectors, vocab, data_dir=args.data_dir)

        # Determine actual data dir for reporting
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = Path.home() / ".wheeler_memory"

        save_path = data_dir / "word_vectors.npz"

        # Print summary
        print(f"Vocab size:      {len(vocab)}")
        print(f"Vector dim:      {args.dim}")
        print(f"Saved to:        {save_path}")
        print("Word vector training complete.")

    except ImportError as e:
        if "sentence_transformers" in str(e):
            print(
                "Error: Word encoding requires sentence-transformers.\n"
                "Install with: pip install -e '.[embed]'",
                file=sys.stderr,
            )
        else:
            print(f"Error: Missing dependency — {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(
            f"Error: Data directory not found or no memories found — {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

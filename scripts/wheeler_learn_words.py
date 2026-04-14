"""CLI: Train word vectors from stored Wheeler memories.

Usage:
    wheeler-learn-words
    wheeler-learn-words --method svd --data-dir /path/to/data --window 5 --dim 384
    wheeler-learn-words --method context-ri --corpus datasets/corpus.jsonl
"""

import argparse
import sys
from pathlib import Path

from wheeler_memory.constants import CONTEXT_RI_REMOVE_TOP_K, CONTEXT_RI_SUBSAMPLE_T


def _train_svd(args):
    """Train word vectors via PMI + SVD (original method)."""
    from wheeler_memory.word_encoder import train_word_vectors, save_word_vectors

    print("Training word vectors (PMI + SVD) from stored memories...")

    vectors, vocab = train_word_vectors(
        data_dir=args.data_dir,
        dim=args.dim,
        window=args.window,
    )

    save_word_vectors(vectors, vocab, data_dir=args.data_dir)

    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".wheeler_memory"
    save_path = data_dir / "word_vectors.npz"

    print(f"Vocab size:      {len(vocab)}")
    print(f"Vector dim:      {args.dim}")
    print(f"Saved to:        {save_path}")
    print("Word vector training complete.")


def _train_context_ri(args):
    """Train context vectors via Random Indexing with hippocampus n-gram index vectors."""
    from wheeler_memory.word_encoder import train_context_ri, save_context_ri_vectors

    corpus_str = f" + corpus {args.corpus}" if args.corpus else ""
    print(f"Training context-RI vectors (window={args.window}){corpus_str}...")

    vectors, vocab = train_context_ri(
        data_dir=args.data_dir,
        window=args.window,
        corpus_path=args.corpus,
        subsample_t=args.subsample_t,
        remove_top_k=args.remove_top_k,
    )

    save_context_ri_vectors(vectors, vocab, data_dir=args.data_dir)

    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".wheeler_memory"
    save_path = data_dir / "context_ri_vectors.npz"

    print(f"Vocab size:      {len(vocab)}")
    print(f"Vector dim:      384")
    print(f"Saved to:        {save_path}")
    print("Context-RI training complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Train word vectors from stored Wheeler memories"
    )
    parser.add_argument(
        "--method",
        choices=["svd", "context-ri"],
        default="svd",
        help="Training method: svd (PMI+SVD, default) or context-ri (random indexing)",
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
        help="Vector dimensionality for SVD method (default: 384)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        metavar="FILE",
        help="JSONL corpus file for context-ri training (optional, supplements stored memories)",
    )
    parser.add_argument(
        "--subsample-t",
        type=float,
        default=CONTEXT_RI_SUBSAMPLE_T,
        help=f"Word2Vec-style subsampling threshold (default: {CONTEXT_RI_SUBSAMPLE_T}, 0=disable)",
    )
    parser.add_argument(
        "--remove-top-k",
        type=int,
        default=CONTEXT_RI_REMOVE_TOP_K,
        help=f"Remove top-k singular components post-training (default: {CONTEXT_RI_REMOVE_TOP_K}, 0=skip)",
    )
    args = parser.parse_args()

    try:
        if args.method == "context-ri":
            _train_context_ri(args)
        else:
            _train_svd(args)

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

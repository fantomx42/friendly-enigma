"""Download OpenWebText subsample and convert to JSONL for context-RI training."""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Download OpenWebText subsample to JSONL")
    parser.add_argument("-o", "--output", default="datasets/openwebtext_500m.jsonl")
    parser.add_argument(
        "--max-words",
        type=int,
        default=500_000_000,
        help="Stop after accumulating this many words (default: 500M)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading OpenWebText from HuggingFace (streaming)...")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    # Shuffle with buffer for diversity
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    total_words = 0
    written = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text or len(text) < 50:
                continue

            words = len(text.split())
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
            total_words += words

            if written % 50_000 == 0:
                print(f"  {written:>8,} docs, {total_words:>12,} words...")

            if total_words >= args.max_words:
                break

    print(f"Done. Wrote {written:,} docs ({total_words:,} words) to {args.output}")


if __name__ == "__main__":
    main()

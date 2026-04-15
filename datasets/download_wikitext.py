"""Download WikiText-103-raw-v1 and convert to JSONL for context-RI training."""

import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download WikiText-103 to JSONL")
    parser.add_argument("-o", "--output", default="datasets/wikitext103.jsonl")
    args = parser.parse_args()

    print("Loading WikiText-103-raw-v1 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    written = 0
    with open(args.output, "w") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1

    print(f"Done. Wrote {written:,} lines to {args.output}")


if __name__ == "__main__":
    main()

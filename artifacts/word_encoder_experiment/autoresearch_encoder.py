"""Autoresearch: encoder strategy sweep for MMLU ternary scoring.

Sweeps encoder strategies and WORD_HIPPO_BLEND values, logs results to TSV.
Keeps the best constants.py setting; reverts regressions.

Usage:
    python scripts/autoresearch_encoder.py
    python scripts/autoresearch_encoder.py --samples 20 --subjects virology sociology
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSTANTS = REPO / "wheeler_memory" / "constants.py"
RESULTS_TSV = REPO / "encoder_sweep_results.tsv"

# Experiments: (encoder, WORD_HIPPO_BLEND value or None if n/a)
EXPERIMENTS = [
    # Baselines
    ("hippocampus", None),
    ("word", None),
    ("word-blended", None),
    # hippo-word blends: sweep WORD_HIPPO_BLEND
    ("hippo-word", 0.1),
    ("hippo-word", 0.2),
    ("hippo-word", 0.3),
    ("hippo-word", 0.4),
    ("hippo-word", 0.5),
    ("hippo-word", 0.6),
    ("hippo-word", 0.7),
]


def set_word_hippo_blend(value: float) -> None:
    """Update WORD_HIPPO_BLEND in constants.py."""
    text = CONSTANTS.read_text()
    import re
    text = re.sub(
        r"WORD_HIPPO_BLEND\s*=\s*[\d.]+",
        f"WORD_HIPPO_BLEND = {value}",
        text,
    )
    CONSTANTS.write_text(text)


def run_mmlu(encoder: str, subjects: list[str], samples: int | None) -> dict:
    """Run MMLU ternary benchmark, return parsed results."""
    cmd = [
        sys.executable, str(REPO / "scripts" / "wheeler_mmlu.py"),
        "--mode", "ternary",
        "--encoder", encoder,
        "--subjects", *subjects,
    ]
    if samples:
        cmd.extend(["--samples", str(samples)])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    output = result.stdout + result.stderr

    # Parse overall accuracy from "OVERALL: X/Y = Z%"
    import re
    m = re.search(r"OVERALL:\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", output)
    if not m:
        return {"correct": 0, "total": 0, "accuracy": 0.0, "time": elapsed, "error": output[-200:]}

    # Parse per-subject
    subject_scores = {}
    for sm in re.finditer(r"(\w+)\s+([\d.]+)%\s+[█ ]+", output):
        subject_scores[sm.group(1)] = float(sm.group(2))

    return {
        "correct": int(m.group(1)),
        "total": int(m.group(2)),
        "accuracy": float(m.group(3)),
        "time": elapsed,
        "subjects": subject_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Encoder strategy autoresearch sweep")
    parser.add_argument("--subjects", nargs="+", default=["sociology", "virology"])
    parser.add_argument("--samples", type=int, default=None, help="Samples per subject (default: all)")
    args = parser.parse_args()

    # Read original WORD_HIPPO_BLEND to restore later if needed
    import re
    orig_text = CONSTANTS.read_text()
    orig_blend = float(re.search(r"WORD_HIPPO_BLEND\s*=\s*([\d.]+)", orig_text).group(1))

    best_accuracy = 0.0
    best_config = None
    rows = []

    print(f"{'='*70}")
    print(f"  ENCODER AUTORESEARCH SWEEP")
    print(f"  Subjects: {args.subjects}")
    print(f"  Samples: {args.samples or 'all'}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"{'='*70}\n")

    for i, (encoder, blend) in enumerate(EXPERIMENTS, 1):
        label = encoder
        if blend is not None:
            label = f"{encoder} (blend={blend})"
            set_word_hippo_blend(blend)

        print(f"  [{i}/{len(EXPERIMENTS)}] {label} ...", end=" ", flush=True)

        try:
            result = run_mmlu(encoder, args.subjects, args.samples)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            result = {"correct": 0, "total": 0, "accuracy": 0.0, "time": 1800, "error": "timeout"}

        acc = result["accuracy"]
        elapsed = result["time"]
        print(f"{acc:.1f}% ({elapsed:.1f}s)")

        if "subjects" in result:
            for subj, subj_acc in result["subjects"].items():
                print(f"    {subj}: {subj_acc:.1f}%")

        row = {
            "encoder": encoder,
            "blend": blend if blend is not None else "",
            "accuracy": acc,
            "correct": result["correct"],
            "total": result["total"],
            "time_s": f"{elapsed:.1f}",
        }
        # Add per-subject columns
        for subj in args.subjects:
            row[subj] = result.get("subjects", {}).get(subj, "")
        rows.append(row)

        if acc > best_accuracy:
            best_accuracy = acc
            best_config = (encoder, blend)

        print()

    # Restore best blend value
    if best_config and best_config[1] is not None:
        set_word_hippo_blend(best_config[1])
        print(f"  Set WORD_HIPPO_BLEND = {best_config[1]} (best config)")
    else:
        set_word_hippo_blend(orig_blend)
        print(f"  Restored WORD_HIPPO_BLEND = {orig_blend}")

    # Write results TSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(RESULTS_TSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"  Results saved to {RESULTS_TSV}")

    print(f"\n{'='*70}")
    print(f"  BEST: {best_config[0]}", end="")
    if best_config[1] is not None:
        print(f" (blend={best_config[1]})", end="")
    print(f" → {best_accuracy:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

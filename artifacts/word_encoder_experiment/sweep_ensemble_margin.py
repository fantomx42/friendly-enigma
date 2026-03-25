"""Sweep ENSEMBLE_MARGIN_THRESHOLD (word-count routing cutoff)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSTANTS = REPO / "wheeler_memory" / "constants.py"

# Word count thresholds to try
THRESHOLDS = [8, 10, 12, 14, 16, 18, 20, 25, 30, 50]


def set_threshold(value: int):
    text = CONSTANTS.read_text()
    text = re.sub(
        r"ENSEMBLE_MARGIN_THRESHOLD\s*=\s*\d+",
        f"ENSEMBLE_MARGIN_THRESHOLD = {value}",
        text,
    )
    CONSTANTS.write_text(text)


def run(threshold: int) -> dict:
    set_threshold(threshold)
    cmd = [
        sys.executable, str(REPO / "scripts" / "wheeler_mmlu.py"),
        "--mode", "ternary-ensemble",
        "--subjects", "sociology", "virology",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    m = re.search(r"OVERALL:\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", output)
    overall = float(m.group(3)) if m else 0.0

    subjects = {}
    for sm in re.finditer(r"(\w+)\s+([\d.]+)%\s+[█ ]+", output):
        subjects[sm.group(1)] = float(sm.group(2))

    return {"overall": overall, **subjects}


def main():
    print(f"{'WordCount':>10} {'Overall':>8} {'Sociology':>10} {'Virology':>10}")
    print("-" * 42)

    best_threshold = 0
    best_overall = 0.0

    for t in THRESHOLDS:
        r = run(t)
        soc = r.get("sociology", 0)
        vir = r.get("virology", 0)
        marker = " <--" if r["overall"] > 31.6 else ""
        print(f"{t:>10} {r['overall']:>7.1f}% {soc:>9.1f}% {vir:>9.1f}%{marker}")
        if r["overall"] > best_overall:
            best_overall = r["overall"]
            best_threshold = t

    print(f"\nBest: word_count>={best_threshold} → {best_overall:.1f}%")
    print(f"(Hippocampus baseline: 31.6%)")
    set_threshold(best_threshold)


if __name__ == "__main__":
    main()

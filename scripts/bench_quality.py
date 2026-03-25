"""Autoresearch-style quality benchmark for Wheeler Memory.

Measures CA dynamics quality across 20 deterministic test inputs.
Outputs a single composite score (lower = better) suitable for tracking
parameter experiments in results.tsv.

Usage
-----
    wheeler-bench              # print JSON result + human summary
    wheeler-bench --json       # print JSON only (for scripting)
    wheeler-bench --no-save    # don't append to results.tsv

Composite metric
----------------
    score = 0.6 * avg_abs_correlation
          + 0.2 * (1 - convergence_ratio)
          + 0.1 * (median_ticks / SALIENCE_MAX_ITERS_MED)
          + 0.1 * (1 - avg_alive_fraction)

Each component is in [0, 1]; overall score is in [0, 1].
Lower is better.  A healthy baseline is ~0.30–0.35.

Components
----------
avg_abs_correlation : mean |Pearson r| across all attractor pairs
                      (primary diversity signal; target < 0.5)
convergence_ratio   : fraction of inputs reaching CONVERGED (want > 0.90)
median_ticks        : median ticks to convergence, normalized by MED budget
avg_alive_fraction  : mean fraction of non-neutral cells per attractor
                      (want > 0.5 — attractors should be richly structured)
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED
from wheeler_memory.dynamics import compute_attractor_features, evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame

# ---------------------------------------------------------------------------
# Fixed test corpus — deterministic (same hash → same initial frame, always)
# ---------------------------------------------------------------------------
TEST_INPUTS = [
    "Fix authentication bug in login flow",
    "Deploy Kubernetes cluster on AWS",
    "Buy groceries: milk, eggs, bread",
    "Schedule dentist appointment for Thursday",
    "Quantum entanglement violates Bell inequalities",
    "The mitochondria is the powerhouse of the cell",
    "Review pull request #42 for memory leaks",
    "Plan birthday party for next Saturday",
    "Configure NGINX reverse proxy with TLS",
    "Water the garden every morning at 7am",
    "Implement binary search tree in Rust",
    "Book flight to Tokyo for March conference",
    "Dark matter comprises 27% of the universe",
    "Refactor database schema for multi-tenancy",
    "Practice piano scales for 30 minutes daily",
    "Debug segfault in GPU kernel launch",
    "Write unit tests for payment processing",
    "Organize closet by season and color",
    "Black holes emit Hawking radiation",
    "Compile FFmpeg with hardware acceleration",
]

_RESULTS_TSV = Path(__file__).parent.parent / "results.tsv"
_TSV_HEADER = (
    "iteration\ttimestamp\tscore\tavg_correlation\tconvergence_ratio"
    "\tmedian_ticks\talive_fraction\telapsed_s\timproved\tcommit\tchanged\tnotes\n"
)


def _next_iteration(path: Path) -> int:
    if not path.exists():
        return 1
    with open(path) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    return len(rows) + 1


def _last_score(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    try:
        return float(rows[-1]["score"])
    except (KeyError, ValueError):
        return None


def run_benchmark(verbose: bool = True) -> dict:
    """Run the quality benchmark and return a result dict."""
    n = len(TEST_INPUTS)
    attractors_flat = []
    states = []
    ticks_list = []
    alive_fractions = []

    t0 = time.time()

    for i, text in enumerate(TEST_INPUTS):
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        flat = result["attractor"].flatten()
        attractors_flat.append(flat)
        states.append(result["state"])
        ticks_list.append(result["convergence_ticks"])
        features = compute_attractor_features(result["attractor"])
        alive_fractions.append(features["alive_fraction"])

        if verbose:
            print(
                f"  [{i + 1:2d}/{n}] {result['state']:<11} "
                f"{result['convergence_ticks']:>4} ticks  {text[:45]}"
            )

    elapsed = time.time() - t0

    # --- Correlation matrix (off-diagonal only) ---
    off_diag_abs = []
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(attractors_flat[i], attractors_flat[j])
            off_diag_abs.append(abs(r))

    avg_abs_correlation = float(np.mean(off_diag_abs))
    max_abs_correlation = float(np.max(off_diag_abs))

    # --- Convergence ---
    n_converged = states.count("CONVERGED")
    convergence_ratio = n_converged / n

    # --- Ticks (normalized) ---
    converged_ticks = [t for s, t in zip(states, ticks_list) if s == "CONVERGED"]
    median_ticks = (
        float(np.median(converged_ticks))
        if converged_ticks
        else float(SALIENCE_MAX_ITERS_MED)
    )
    ticks_norm = median_ticks / SALIENCE_MAX_ITERS_MED

    # --- Alive fraction ---
    avg_alive_fraction = float(np.mean(alive_fractions))

    # --- Composite score (lower = better) ---
    score = (
        0.6 * avg_abs_correlation
        + 0.2 * (1.0 - convergence_ratio)
        + 0.1 * ticks_norm
        + 0.1 * (1.0 - avg_alive_fraction)
    )

    return {
        "score": round(score, 6),
        "avg_correlation": round(avg_abs_correlation, 6),
        "max_correlation": round(max_abs_correlation, 6),
        "convergence_ratio": round(convergence_ratio, 4),
        "median_ticks": round(median_ticks, 1),
        "avg_alive_fraction": round(avg_alive_fraction, 4),
        "elapsed_seconds": round(elapsed, 2),
        "n_converged": n_converged,
        "n_total": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def append_result(
    result: dict, improved: bool | None, commit: str, changed: str, notes: str
) -> None:
    """Append one row to results.tsv (creates the file with header if absent)."""
    if not _RESULTS_TSV.exists():
        _RESULTS_TSV.write_text(_TSV_HEADER)

    iteration = _next_iteration(_RESULTS_TSV)
    improved_str = "" if improved is None else ("1" if improved else "0")

    row = (
        "\t".join(
            [
                str(iteration),
                result["timestamp"],
                str(result["score"]),
                str(result["avg_correlation"]),
                str(result["convergence_ratio"]),
                str(result["median_ticks"]),
                str(result["avg_alive_fraction"]),
                str(result["elapsed_seconds"]),
                improved_str,
                commit,
                changed,
                notes,
            ]
        )
        + "\n"
    )

    with open(_RESULTS_TSV, "a") as f:
        f.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheeler Memory quality benchmark")
    parser.add_argument("--json", action="store_true", help="Print JSON result only")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not append to results.tsv"
    )
    parser.add_argument("--commit", default="", help="Git commit hash for results.tsv")
    parser.add_argument(
        "--changed", default="", help="Parameter(s) changed for results.tsv"
    )
    parser.add_argument("--notes", default="", help="Free-text notes for results.tsv")
    args = parser.parse_args()

    verbose = not args.json
    if verbose:
        print(f"Running quality benchmark ({len(TEST_INPUTS)} inputs)...\n")

    result = run_benchmark(verbose=verbose)

    if args.json:
        print(json.dumps(result))
        return

    prev_score = _last_score(_RESULTS_TSV)
    improved = (
        None
        if prev_score is None
        else (
            True
            if result["score"] < prev_score
            else None
            if result["score"] == prev_score
            else False
        )
    )

    # Human-readable summary
    print(f"\n{'=' * 55}")
    print(f"  QUALITY BENCHMARK RESULT")
    print(f"{'=' * 55}")
    print(f"  Score          : {result['score']:.6f}  (lower = better)")
    if prev_score is not None:
        delta = result["score"] - prev_score
        arrow = "▼" if delta < 0 else "▲"
        print(f"  vs. previous   : {prev_score:.6f}  {arrow} {abs(delta):.6f}")
    print(f"  Avg |r|        : {result['avg_correlation']:.4f}  (target < 0.50)")
    print(f"  Max |r|        : {result['max_correlation']:.4f}  (target < 0.85)")
    print(
        f"  Convergence    : {result['n_converged']}/{result['n_total']}  ({result['convergence_ratio']:.0%})"
    )
    print(f"  Median ticks   : {result['median_ticks']:.0f}")
    print(f"  Alive fraction : {result['avg_alive_fraction']:.4f}")
    print(f"  Elapsed        : {result['elapsed_seconds']:.1f}s")
    print(f"{'=' * 55}")

    if improved is True:
        print("  IMPROVED — keep these parameters")
    elif improved is False:
        print("  REGRESSION — consider reverting constants.py")

    if not args.no_save:
        append_result(result, improved, args.commit, args.changed, args.notes)
        print(f"\n  Appended to {_RESULTS_TSV.name}")


if __name__ == "__main__":
    main()

"""Cross-cube interference probe (CANON open item #4, §6.4).

Characterizes what it means for a nested cube³:0 to interfere with its parent.
For each parent attractor we spawn its 6 children through the FCAS portal and
measure, via fcas.cross_cube_interference, two things:

  1. depth coherence  — mean |Pearson(parent, child)| across all parent×face
     pairs. Because expand_cube seeds children with sha256(portal_hash + face),
     children are decorrelated from the parent by construction, so this should
     sit at the noise floor for two independent length-4096 vectors, ≈ 1/√4096
     ≈ 0.0156. That is the decisive orthogonality test.
  2. GROUNDED fraction vs the null expectation parent_peak_rate ×
     child_peak_rate. Converged attractors are dense (peak almost everywhere),
     so both observed GROUNDED and the null product land near 1.0 — co-peaking
     reflects attractor density, not interference structure, and is *also*
     consistent with independence.

Conclusion the numbers support: the fractal address space is an orthogonal
decomposition — a child cannot constructively interfere with its parent.

Usage
-----
    python scripts/bench/cross_cube_probe.py
    python scripts/bench/cross_cube_probe.py --json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from wheeler_memory.constants import INTERFERENCE_PEAK_THRESHOLD
from wheeler_memory.dynamics import evolve_batch
from wheeler_memory.fcas import cross_cube_interference, evolve_cube, portal_hash
from wheeler_memory.hashing import hash_to_frame

# Imported read-only — does not modify the sacred benchmark corpus.
from scripts.bench_quality import TEST_INPUTS


def _peak_rate(att: np.ndarray) -> float:
    return float((np.abs(att) > INTERFERENCE_PEAK_THRESHOLD).mean())


def run_probe(verbose: bool = True) -> dict:
    """Aggregate cross-cube interference over the fixed corpus."""
    parents = [r["attractor"] for r in evolve_batch([hash_to_frame(t) for t in TEST_INPUTS])]

    coherences: list[float] = []
    grounded: list[float] = []
    signals: list[float] = []
    parent_peaks: list[float] = []
    child_peaks: list[float] = []

    for parent in parents:
        reading = cross_cube_interference(parent)
        coherences.extend(f["correlation"] for f in reading["per_face"].values())
        grounded.append(reading["grounded_fraction"])
        signals.append(reading["mean_signal_strength"])
        parent_peaks.append(_peak_rate(parent))
        for child_result in evolve_cube(portal_hash(parent)).values():
            child_peaks.append(_peak_rate(child_result["attractor"]))

    mean_parent_peak = float(np.mean(parent_peaks))
    mean_child_peak = float(np.mean(child_peaks))
    null_grounded = mean_parent_peak * mean_child_peak
    noise_floor = 1.0 / np.sqrt(4096)  # expected std of r for independent vectors

    return {
        "n_parents": len(parents),
        "n_pairs": len(coherences),
        "mean_abs_correlation": round(float(np.mean(coherences)), 5),
        "max_abs_correlation": round(float(np.max(coherences)), 5),
        "noise_floor": round(float(noise_floor), 5),
        "observed_grounded_fraction": round(float(np.mean(grounded)), 5),
        "null_grounded_fraction": round(null_grounded, 5),
        "mean_parent_peak_rate": round(mean_parent_peak, 5),
        "mean_child_peak_rate": round(mean_child_peak, 5),
        "mean_signal_strength": round(float(np.mean(signals)), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-cube interference probe (§6.4)")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    if not args.json:
        print(
            f"Cross-cube interference probe "
            f"({len(TEST_INPUTS)} parents × 6 children)...\n"
        )
    result = run_probe(verbose=not args.json)

    if args.json:
        print(json.dumps(result))
        return

    orthogonal = result["mean_abs_correlation"] < 3 * result["noise_floor"]
    grounded_consistent = (
        abs(result["observed_grounded_fraction"] - result["null_grounded_fraction"])
        < 0.05
    )

    print(f"{'=' * 60}")
    print("  CROSS-CUBE INTERFERENCE PROBE")
    print(f"{'=' * 60}")
    print(f"  parent×child pairs        : {result['n_pairs']}")
    print(
        f"  depth coherence (mean |r|): {result['mean_abs_correlation']:.5f}  "
        f"(noise floor 1/√4096 = {result['noise_floor']:.5f})"
    )
    print(f"  max |r|                   : {result['max_abs_correlation']:.5f}")
    print(
        f"  GROUNDED  observed → null : "
        f"{result['observed_grounded_fraction']:.4f} → {result['null_grounded_fraction']:.4f}"
        f"  (peak rates {result['mean_parent_peak_rate']:.3f} / "
        f"{result['mean_child_peak_rate']:.3f})"
    )
    print(f"  mean signal strength      : {result['mean_signal_strength']:.5f}")
    print(f"{'-' * 60}")
    verdict = (
        "ORTHOGONAL — children are independent of parent"
        if orthogonal and grounded_consistent
        else "NON-TRIVIAL interference detected — investigate"
    )
    print(f"  verdict: {verdict}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

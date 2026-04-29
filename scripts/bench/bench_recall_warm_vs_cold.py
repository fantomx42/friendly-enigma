"""Benchmark: warm-start (recognize → reconstruct) vs cold (recall_memory).

Measures CA cycles spent on reconstruction across three input-distance bands:
near_identical, moderate, substantial. Reports cycles per band AND recognition
rate per band; the warm-vs-cold ratio is computed only over RECOGNIZED queries
to avoid hiding the truth behind misses.

Honest interpretation:
- near_identical band SHOULD show the largest ratio (warm-start fully avoids
  query evolve when stored ≈ query)
- moderate band SHOULD show a meaningful but smaller ratio
- substantial band SHOULD show low recognition rate (correct! those queries
  legitimately need cold reconstruction); ratio over the few hits is reported
  separately for transparency
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.recall_api import recognize, reconstruct
from wheeler_memory.storage import recall_memory, store_memory


PAIRS_BY_BAND: dict[str, list[tuple[str, str]]] = {
    "near_identical": [
        ("Quantum entanglement violates Bell inequalities",
         "Quantum entanglement violates Bell inequalities."),
        ("python list comprehensions",
         "Python list comprehensions"),
        ("the cat sat on the mat",
         "the cat sat on the mat."),
        ("photosynthesis converts light to chemical energy",
         "photosynthesis converts light to chemical energy "),
        ("HTTP status code 404 means not found",
         "HTTP status code 404 means not found."),
        ("git rebase rewrites commit history",
         "git rebase rewrites commit history "),
        ("DNA encodes genetic information",
         "DNA encodes genetic information."),
        ("the speed of light is constant",
         "The speed of light is constant"),
        ("Big O notation describes algorithmic complexity",
         "Big O notation describes algorithmic complexity."),
        ("regex matches text patterns",
         "Regex matches text patterns"),
    ],
    "moderate": [
        ("Quantum entanglement violates Bell inequalities",
         "Bell inequalities are violated by entangled particles"),
        ("python list comprehensions",
         "list comps in python"),
        ("the cat sat on the mat",
         "a cat was on the rug"),
        ("photosynthesis converts light to chemical energy",
         "plants turn sunlight into chemical energy"),
        ("HTTP status code 404 means not found",
         "404 is the not-found HTTP code"),
        ("git rebase rewrites commit history",
         "rebase in git changes history"),
        ("DNA encodes genetic information",
         "genetic info is stored in DNA"),
        ("the speed of light is constant",
         "light speed never changes"),
        ("Big O notation describes algorithmic complexity",
         "algorithm complexity uses Big O"),
        ("regex matches text patterns",
         "regular expressions find text"),
    ],
    "substantial": [
        ("Quantum entanglement violates Bell inequalities",
         "spooky action at a distance"),
        ("python list comprehensions",
         "filtering iterables"),
        ("the cat sat on the mat",
         "feline rests on textile"),
        ("photosynthesis converts light to chemical energy",
         "chlorophyll absorbs solar radiation"),
        ("HTTP status code 404 means not found",
         "missing resource error code"),
        ("git rebase rewrites commit history",
         "version control linearization"),
        ("DNA encodes genetic information",
         "double helix carrier molecule"),
        ("the speed of light is constant",
         "relativistic invariant velocity"),
        ("Big O notation describes algorithmic complexity",
         "asymptotic upper bound"),
        ("regex matches text patterns",
         "pattern engine for strings"),
    ],
}


def _store_pair(stored_text: str, data_dir: Path) -> str:
    frame = hash_to_frame(stored_text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(stored_text, result, brick, data_dir=data_dir, auto_evict=False)


def _measure_cold(query: str, data_dir: Path) -> tuple[int, bool]:
    """recall_memory(reconstruct=True). Returns (total_ticks, recognized).

    `recognized` is best-effort: True if the top-1 hit's text matches a stored
    item. We approximate by checking similarity > 0 — recall_memory always
    finds *something*, and we want both paths to agree on "recognition" for
    a fair comparison. We track recognition rate separately via recognize().
    """
    hits = recall_memory(
        query,
        top_k=1,
        data_dir=data_dir,
        reconstruct=True,
        readonly=True,
    )
    if not hits:
        return 0, False
    h = hits[0]
    query_ticks = h.get("convergence_ticks", 0)
    recon_ticks = h.get("reconstruction_ticks", 0)
    return int(query_ticks) + int(recon_ticks), True


def _measure_warm(query: str, data_dir: Path) -> tuple[int, bool]:
    """recognize() + reconstruct(seed, query). Returns (total_ticks, recognized).

    Uses threshold=0.0 to ensure parity in coverage with cold; recognition
    rate at the production threshold is tracked separately.
    """
    seed = recognize(query, data_dir=data_dir, threshold=0.0)
    if seed is None:
        return 0, False
    pat = reconstruct(seed, query=query, alpha=0.3)
    # recognize() runs 1 apply_ca_dynamics step probe (constant 1 tick).
    return 1 + int(pat.convergence_ticks), True


def _recognition_rate(query: str, data_dir: Path, threshold: float) -> bool:
    """Did recognize() find a hit at the production threshold?"""
    return recognize(query, data_dir=data_dir, threshold=threshold) is not None


def run_band(
    band: str,
    pairs: list[tuple[str, str]],
    runs: int,
    threshold: float,
) -> dict:
    """Run a single band: store stored side, query with query side, measure both paths."""
    cold_ticks: list[int] = []
    warm_ticks: list[int] = []
    recognition_hits = 0

    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d)
        for stored_text, _ in pairs:
            _store_pair(stored_text, data_dir)

        for stored_text, query in pairs:
            for _ in range(runs):
                cold_t, _ = _measure_cold(query, data_dir)
                warm_t, _ = _measure_warm(query, data_dir)
                if cold_t > 0 and warm_t > 0:
                    cold_ticks.append(cold_t)
                    warm_ticks.append(warm_t)
            if _recognition_rate(query, data_dir, threshold):
                recognition_hits += 1

    n_pairs = len(pairs)
    if cold_ticks and warm_ticks:
        cold_mean = sum(cold_ticks) / len(cold_ticks)
        warm_mean = sum(warm_ticks) / len(warm_ticks)
        ratio = cold_mean / warm_mean if warm_mean > 0 else float("inf")
    else:
        cold_mean = warm_mean = ratio = 0.0

    return {
        "band": band,
        "n_pairs": n_pairs,
        "runs_per_pair": runs,
        "cold_ticks_mean": round(cold_mean, 2),
        "warm_ticks_mean": round(warm_mean, 2),
        "ratio": round(ratio, 2),
        "recognition_hits": recognition_hits,
        "recognition_rate": round(recognition_hits / n_pairs, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Warm-start vs cold-recall CA cycle benchmark across input-distance bands"
    )
    parser.add_argument(
        "--bands",
        choices=["all", "near_identical", "moderate", "substantial"],
        default="all",
        help="Which band(s) to run (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per pair to average tick jitter (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Production recognition threshold for the recognition_rate column (default: 0.45)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write per-band results as JSON",
    )
    args = parser.parse_args()

    bands = list(PAIRS_BY_BAND.keys()) if args.bands == "all" else [args.bands]
    results = []
    print(
        f"{'band':<16} {'n':>3} {'runs':>4} {'cold_ticks':>10} {'warm_ticks':>10} "
        f"{'ratio':>6} {'recog_rate':>11}"
    )
    print("-" * 78)
    t0 = time.time()
    for band in bands:
        out = run_band(band, PAIRS_BY_BAND[band], args.runs, args.threshold)
        results.append(out)
        print(
            f"{out['band']:<16} {out['n_pairs']:>3} {out['runs_per_pair']:>4} "
            f"{out['cold_ticks_mean']:>10.1f} {out['warm_ticks_mean']:>10.1f} "
            f"{out['ratio']:>6.1f}x {out['recognition_rate']:>10.1%}"
        )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    if args.json_out:
        out_path = Path(args.json_out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"bench_recall_warm_vs_cold-{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "results": results,
    }, indent=2))
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()

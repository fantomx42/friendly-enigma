"""Wheeler-native reconstruction-fidelity benchmark (CANON.md §8.3).

Where ``wheeler-bench`` measures attractor *diversity* across the corpus, this
benchmark measures attractor *self-recovery* — the defining behaviour of an
attractor-reconstruction memory. The procedure, per §8.3:

    perturb a known attractor, measure settling time and final-state fidelity.

For each input we evolve it to its attractor A (the "known attractor"), add
Gaussian noise at a sweep of magnitudes ε, clip back to [-1, 1], and re-evolve.
Fidelity is the Pearson correlation between the re-settled attractor and A;
settling time is the convergence tick count. Sweeping ε traces a fidelity curve
whose collapse point estimates the basin's **capture radius** — a pure
architecture signal that MMLU (corpus-limited, §8.2) cannot expose.

Determinism: the known attractors come from the fixed, sacred ``TEST_INPUTS``
corpus (imported read-only), and each perturbation is seeded from the input hash
plus ε, so the whole benchmark is reproducible and comparable across runs.

Usage
-----
    wheeler-recon-bench            # human summary + fidelity curve
    wheeler-recon-bench --json     # JSON only (for scripting)
    wheeler-recon-bench --no-save  # don't append to reconstruction.tsv
    wheeler-recon-bench --cpu-only # force CPU dispatch for a reproducible baseline

Composite metric (lower = better, matching wheeler-bench convention)
--------------------------------------------------------------------
    score = 0.7 * (1 - mean_fidelity) + 0.3 * mean_ticks_norm

where mean_fidelity is averaged (clipped to [0,1]) over every (input, ε) trial
and mean_ticks_norm normalises settling time by the MED iteration budget.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED
from wheeler_memory.dynamics import evolve_and_interpret, evolve_batch
from wheeler_memory.hashing import hash_to_frame

# Imported read-only — this does NOT modify the sacred benchmark corpus.
from scripts.bench_quality import TEST_INPUTS

# Perturbation magnitudes (std-dev of additive Gaussian noise on the [-1,1] grid).
EPS_LEVELS = (0.1, 0.25, 0.5, 0.75, 1.0)
# A trial counts as "recovered" if it re-settles within this correlation of A.
FIDELITY_THRESHOLD = 0.9

_RESULTS_TSV = Path(__file__).parent.parent / "reconstruction.tsv"
_TSV_HEADER = (
    "iteration\ttimestamp\tscore\tmean_fidelity\tcapture_radius"
    "\tmean_ticks\trecovery_rate\telapsed_s\tnotes\n"
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    try:
        r, _ = pearsonr(a.flatten(), b.flatten())
    except Exception:
        return 0.0
    return float(r) if not np.isnan(r) else 0.0


def perturb(attractor: np.ndarray, eps: float, seed: int) -> np.ndarray:
    """Additive Gaussian perturbation of an attractor, clipped to [-1, 1].

    Deterministic for a given (attractor-text, eps): the caller supplies a seed
    derived from both, so repeated runs reproduce the same perturbed frame.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.normal(0.0, eps, size=attractor.shape).astype(np.float32)
    return np.clip(attractor + noise, -1.0, 1.0)


def _seed_for(text: str, eps: float) -> int:
    digest = hashlib.sha256(f"{text}|{eps}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def run_benchmark(eps_levels=EPS_LEVELS, verbose: bool = True) -> dict:
    """Run the reconstruction benchmark and return a result dict."""
    t0 = time.time()

    # 1. Evolve the fixed corpus to its "known" attractors (one batch dispatch).
    originals = evolve_batch([hash_to_frame(text) for text in TEST_INPUTS])
    references = [r["attractor"] for r in originals]

    # 2. For each ε, perturb every reference and re-evolve (one batch per ε).
    per_eps = {}
    for eps in eps_levels:
        perturbed = [
            perturb(ref, eps, _seed_for(text, eps))
            for text, ref in zip(TEST_INPUTS, references)
        ]
        resettled = evolve_batch(perturbed)

        fidelities = np.array(
            [
                float(np.clip(_pearson(res["attractor"], ref), 0.0, 1.0))
                for res, ref in zip(resettled, references)
            ]
        )
        ticks = np.array([res["convergence_ticks"] for res in resettled], dtype=float)
        recovered = fidelities >= FIDELITY_THRESHOLD

        per_eps[eps] = {
            "mean_fidelity": float(fidelities.mean()),
            "min_fidelity": float(fidelities.min()),
            "mean_ticks": float(ticks.mean()),
            "recovery_rate": float(recovered.mean()),
        }

        if verbose:
            bar = "#" * int(round(per_eps[eps]["mean_fidelity"] * 20))
            print(
                f"  ε={eps:<4}  fidelity={per_eps[eps]['mean_fidelity']:.3f}  "
                f"recovered={per_eps[eps]['recovery_rate']:.0%}  "
                f"ticks={per_eps[eps]['mean_ticks']:.0f}  |{bar:<20}|"
            )

    elapsed = time.time() - t0

    # 3. Aggregate across all (input, ε) trials.
    mean_fidelity = float(np.mean([m["mean_fidelity"] for m in per_eps.values()]))
    mean_ticks = float(np.mean([m["mean_ticks"] for m in per_eps.values()]))
    recovery_rate = float(np.mean([m["recovery_rate"] for m in per_eps.values()]))
    ticks_norm = mean_ticks / SALIENCE_MAX_ITERS_MED

    # Capture radius: largest ε whose mean fidelity still clears the threshold.
    # 0.0 means even the gentlest perturbation breaks recovery.
    passing = [e for e in eps_levels if per_eps[e]["mean_fidelity"] >= FIDELITY_THRESHOLD]
    capture_radius = float(max(passing)) if passing else 0.0

    score = 0.7 * (1.0 - mean_fidelity) + 0.3 * ticks_norm

    return {
        "score": round(score, 6),
        "mean_fidelity": round(mean_fidelity, 6),
        "capture_radius": round(capture_radius, 4),
        "mean_ticks": round(mean_ticks, 1),
        "recovery_rate": round(recovery_rate, 4),
        "fidelity_threshold": FIDELITY_THRESHOLD,
        "per_eps": {str(e): m for e, m in per_eps.items()},
        "elapsed_seconds": round(elapsed, 2),
        "n_inputs": len(TEST_INPUTS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _next_iteration(path: Path) -> int:
    if not path.exists():
        return 1
    with open(path) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    return len(rows) + 1


def _last_score(path: Path) -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None
    try:
        return float(rows[-1]["score"])
    except (KeyError, ValueError):
        return None


def append_result(result: dict, notes: str) -> None:
    """Append one row to reconstruction.tsv (creates header if absent)."""
    if not _RESULTS_TSV.exists():
        _RESULTS_TSV.write_text(_TSV_HEADER)
    iteration = _next_iteration(_RESULTS_TSV)
    row = (
        "\t".join(
            [
                str(iteration),
                result["timestamp"],
                str(result["score"]),
                str(result["mean_fidelity"]),
                str(result["capture_radius"]),
                str(result["mean_ticks"]),
                str(result["recovery_rate"]),
                str(result["elapsed_seconds"]),
                notes,
            ]
        )
        + "\n"
    )
    with open(_RESULTS_TSV, "a") as f:
        f.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wheeler Memory reconstruction-fidelity benchmark (CANON §8.3)"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result only")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not append to reconstruction.tsv"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU dispatch (sets WHEELER_DISABLE_GPU=1) for a reproducible baseline",
    )
    parser.add_argument("--notes", default="", help="Free-text notes for reconstruction.tsv")
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["WHEELER_DISABLE_GPU"] = "1"

    verbose = not args.json
    if verbose:
        print(
            f"Running reconstruction benchmark "
            f"({len(TEST_INPUTS)} attractors × {len(EPS_LEVELS)} ε levels)...\n"
        )

    result = run_benchmark(verbose=verbose)

    if args.json:
        print(json.dumps(result))
        return

    prev_score = _last_score(_RESULTS_TSV)

    print(f"\n{'=' * 55}")
    print("  RECONSTRUCTION BENCHMARK RESULT")
    print(f"{'=' * 55}")
    print(f"  Score          : {result['score']:.6f}  (lower = better)")
    if prev_score is not None:
        delta = result["score"] - prev_score
        arrow = "▼" if delta < 0 else "▲"
        print(f"  vs. previous   : {prev_score:.6f}  {arrow} {abs(delta):.6f}")
    print(f"  Mean fidelity  : {result['mean_fidelity']:.4f}  (target > 0.80)")
    print(
        f"  Capture radius : ε≤{result['capture_radius']:.2f}  "
        f"(largest ε with mean fidelity ≥ {FIDELITY_THRESHOLD})"
    )
    print(f"  Recovery rate  : {result['recovery_rate']:.0%}")
    print(f"  Mean ticks     : {result['mean_ticks']:.0f}")
    print(f"  Elapsed        : {result['elapsed_seconds']:.1f}s")
    print(f"{'=' * 55}")

    env = (
        f"[env: np={np.__version__} "
        f"py={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}]"
    )
    notes = f"{args.notes} {env}".strip() if args.notes else env

    if not args.no_save:
        append_result(result, notes)
        print(f"\n  Appended to {_RESULTS_TSV.name}")


if __name__ == "__main__":
    main()

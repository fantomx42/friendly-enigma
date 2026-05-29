"""Pre-registered substrate head-to-head: Wheeler CA vs classical Hopfield.

Runs both engines through the *exact* perturbation regime of
``wheeler-recon-bench`` (same TEST_INPUTS, same Gaussian noise levels, same
Pearson fidelity threshold). The Wheeler attractors and the Hopfield
attractors are scored on the natural form of each engine's stored pattern:
Wheeler's continuous attractor for the Wheeler track, sign-snapped attractor
for the Hopfield track. Each engine wins or loses on its own substrate.

Pre-registered primary criterion (set before running, see
``plans/im-not-sure-what-steady-church.md``):

    Wheeler's capture radius (largest eps with mean fidelity >= 0.9) is
    STRICTLY GREATER than Hopfield's capture radius. Pass / tie / fail.

Wheeler's CA is spatially local (von Neumann, dynamics.py:54-71); classical
Hopfield is all-to-all. Locality normally lowers capacity, so a Wheeler win
overcomes a handicap; a Wheeler loss is ambiguous (could be the locality
tax). The plan calls out the asymmetry honestly.

Usage
-----
    python scripts/bench/bench_substrate_vs_hopfield.py
    python scripts/bench/bench_substrate_vs_hopfield.py --cpu-only --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the canonical perturbation regime and fidelity metric so the two
# tracks are scored identically. Importing as a library; no duplication.
from scripts.bench_reconstruction import (
    EPS_LEVELS,
    FIDELITY_THRESHOLD,
    _pearson,
    _seed_for,
    perturb,
)
from scripts.bench_quality import TEST_INPUTS

from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED
from wheeler_memory.dynamics import evolve_batch
from wheeler_memory.hashing import hash_to_frame

_RESULTS_TSV = Path(__file__).resolve().parents[2] / "substrate_comparison.tsv"
_TSV_HEADER = (
    "iteration\ttimestamp\twheeler_capture\thopfield_capture\twheeler_fidelity"
    "\thopfield_fidelity\twheeler_recovery\thopfield_recovery"
    "\twheeler_spurious\thopfield_spurious\tverdict\telapsed_s\tnotes\n"
)


# --- Hopfield baseline ------------------------------------------------------
#
# The Wheeler attractors are heavily biased (~77% +1, ~23% -1, mean ~0.54).
# Vanilla Hebbian Hopfield on biased patterns collapses to the all-+1 fixed
# point — a well-known failure mode unrelated to substrate quality. The
# textbook correction is to center patterns before training and threshold
# the update by the same bias on the way back out. We use that variant
# everywhere so Hopfield gets the fair fight the comparison requires.


def hopfield_train(patterns: np.ndarray) -> tuple[np.ndarray, float]:
    """Centered Hebbian outer-product Hopfield weight matrix.

    Patterns: (P, N) sign-valued in {-1, +1}.
    Returns (W, bias) where W: (N, N) zero-diagonal, bias = scalar mean of
    the pattern bank used to re-bias the sign update.
    """
    bias = float(patterns.mean())
    centered = patterns - bias
    w = centered.T @ centered
    np.fill_diagonal(w, 0.0)
    return (w / float(centered.shape[1])).astype(np.float32), bias


def hopfield_settle(
    w: np.ndarray,
    bias: float,
    x0: np.ndarray,
    max_iters: int = SALIENCE_MAX_ITERS_MED,
) -> dict:
    """Synchronous sign update x <- sign(W (x - bias)) + bias.

    The +bias re-injection puts the output back on the biased manifold the
    patterns live on. Returns dict shape-compatible with evolve_batch.
    """
    x = np.sign(x0.flatten().astype(np.float32))
    x = np.where(x == 0.0, 1.0, x).astype(np.float32)
    for t in range(1, max_iters + 1):
        field = w @ (x - bias)
        x_new = np.sign(field)
        # Re-bias the output: at zero field, default to +1 (the dominant role).
        x_new = np.where(x_new == 0.0, 1.0, x_new).astype(np.float32)
        if np.array_equal(x_new, x):
            return {
                "attractor": x.reshape(64, 64),
                "convergence_ticks": t,
                "state": "CONVERGED",
            }
        x = x_new
    return {
        "attractor": x.reshape(64, 64),
        "convergence_ticks": max_iters,
        "state": "CHAOTIC",
    }


# --- Comparison harness -----------------------------------------------------


def _sign_snap(frame: np.ndarray) -> np.ndarray:
    out = np.sign(frame).astype(np.float32)
    return np.where(out == 0.0, 1.0, out)


def _spurious_rate(
    settled: list[np.ndarray],
    references: list[np.ndarray],
    fidelities: np.ndarray,
) -> float:
    """Fraction of failed-recovery trials that settled to a *non-stored* fixed point.

    A trial is spurious if its fidelity-vs-correct < threshold AND its max
    fidelity against ALL stored references is also < threshold. That is: it
    settled, but to a state outside the entire stored attractor set.
    """
    failed_mask = fidelities < FIDELITY_THRESHOLD
    if not failed_mask.any():
        return 0.0
    refs = np.stack([r.flatten() for r in references]).astype(np.float32)
    spurious = 0
    failures = 0
    for i, is_fail in enumerate(failed_mask):
        if not is_fail:
            continue
        failures += 1
        s = settled[i].flatten().astype(np.float32)
        best = max(_pearson(s, r) for r in refs)
        if best < FIDELITY_THRESHOLD:
            spurious += 1
    return spurious / failures if failures else 0.0


def _capture_radius(per_eps: dict[float, dict]) -> float:
    passing = [e for e in per_eps if per_eps[e]["mean_fidelity"] >= FIDELITY_THRESHOLD]
    return float(max(passing)) if passing else 0.0


def run_wheeler_track(references: list[np.ndarray]) -> dict:
    per_eps = {}
    for eps in EPS_LEVELS:
        perturbed = [
            perturb(ref, eps, _seed_for(text, eps))
            for text, ref in zip(TEST_INPUTS, references)
        ]
        resettled = evolve_batch(perturbed)
        settled_frames = [r["attractor"] for r in resettled]
        fidelities = np.array(
            [
                float(np.clip(_pearson(s, r), 0.0, 1.0))
                for s, r in zip(settled_frames, references)
            ]
        )
        ticks = np.array(
            [r["convergence_ticks"] for r in resettled], dtype=float
        )
        per_eps[eps] = {
            "mean_fidelity": float(fidelities.mean()),
            "recovery_rate": float((fidelities >= FIDELITY_THRESHOLD).mean()),
            "mean_ticks": float(ticks.mean()),
            "spurious_rate": _spurious_rate(settled_frames, references, fidelities),
        }
    return per_eps


def run_hopfield_track(references: list[np.ndarray]) -> dict:
    sign_refs = [_sign_snap(r) for r in references]
    patterns = np.stack([r.flatten() for r in sign_refs]).astype(np.float32)
    w, bias = hopfield_train(patterns)

    per_eps = {}
    for eps in EPS_LEVELS:
        # Use the same perturbation function on the sign-snapped references so
        # noise magnitude matches the Wheeler track exactly.
        perturbed = [
            perturb(ref, eps, _seed_for(text, eps))
            for text, ref in zip(TEST_INPUTS, sign_refs)
        ]
        resettled = [hopfield_settle(w, bias, p) for p in perturbed]
        settled_frames = [r["attractor"] for r in resettled]
        fidelities = np.array(
            [
                float(np.clip(_pearson(s, r), 0.0, 1.0))
                for s, r in zip(settled_frames, sign_refs)
            ]
        )
        ticks = np.array(
            [r["convergence_ticks"] for r in resettled], dtype=float
        )
        per_eps[eps] = {
            "mean_fidelity": float(fidelities.mean()),
            "recovery_rate": float((fidelities >= FIDELITY_THRESHOLD).mean()),
            "mean_ticks": float(ticks.mean()),
            "spurious_rate": _spurious_rate(settled_frames, sign_refs, fidelities),
        }
    return per_eps


def run_comparison() -> dict:
    t0 = time.time()
    originals = evolve_batch([hash_to_frame(text) for text in TEST_INPUTS])
    references = [r["attractor"] for r in originals]

    wheeler = run_wheeler_track(references)
    hopfield = run_hopfield_track(references)

    def _agg(per_eps):
        return {
            "capture_radius": _capture_radius(per_eps),
            "mean_fidelity": float(
                np.mean([m["mean_fidelity"] for m in per_eps.values()])
            ),
            "recovery_rate": float(
                np.mean([m["recovery_rate"] for m in per_eps.values()])
            ),
            "mean_ticks": float(np.mean([m["mean_ticks"] for m in per_eps.values()])),
            "spurious_rate": float(
                np.mean([m["spurious_rate"] for m in per_eps.values()])
            ),
            "per_eps": {str(e): m for e, m in per_eps.items()},
        }

    w_agg = _agg(wheeler)
    h_agg = _agg(hopfield)

    if w_agg["capture_radius"] > h_agg["capture_radius"]:
        verdict = "PASS"
    elif w_agg["capture_radius"] == h_agg["capture_radius"]:
        verdict = "TIE"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "wheeler": w_agg,
        "hopfield": h_agg,
        "n_patterns": len(TEST_INPUTS),
        "n_cells": 4096,
        "alpha": len(TEST_INPUTS) / 4096.0,
        "classical_alpha_limit": 0.138,
        "fidelity_threshold": FIDELITY_THRESHOLD,
        "eps_levels": list(EPS_LEVELS),
        "elapsed_seconds": round(time.time() - t0, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- IO ---------------------------------------------------------------------


def _next_iteration(path: Path) -> int:
    if not path.exists():
        return 1
    with open(path) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    return len(rows) + 1


def append_result(result: dict, notes: str) -> None:
    if not _RESULTS_TSV.exists():
        _RESULTS_TSV.write_text(_TSV_HEADER)
    iteration = _next_iteration(_RESULTS_TSV)
    w = result["wheeler"]
    h = result["hopfield"]
    row = (
        "\t".join(
            [
                str(iteration),
                result["timestamp"],
                f"{w['capture_radius']:.4f}",
                f"{h['capture_radius']:.4f}",
                f"{w['mean_fidelity']:.4f}",
                f"{h['mean_fidelity']:.4f}",
                f"{w['recovery_rate']:.4f}",
                f"{h['recovery_rate']:.4f}",
                f"{w['spurious_rate']:.4f}",
                f"{h['spurious_rate']:.4f}",
                result["verdict"],
                f"{result['elapsed_seconds']:.2f}",
                notes,
            ]
        )
        + "\n"
    )
    with open(_RESULTS_TSV, "a") as f:
        f.write(row)


def _print_human(result: dict) -> None:
    w, h = result["wheeler"], result["hopfield"]
    print("\n" + "=" * 68)
    print("  SUBSTRATE HEAD-TO-HEAD  (Wheeler CA vs classical Hopfield)")
    print("=" * 68)
    print(
        f"  N patterns: {result['n_patterns']}  |  N cells: {result['n_cells']}  "
        f"|  alpha = {result['alpha']:.4f}  "
        f"(classical limit {result['classical_alpha_limit']:.3f})"
    )
    print(f"  Fidelity threshold: {result['fidelity_threshold']}")
    print()
    print(
        f"  {'eps':>5}  {'WHEELER fid':>12} {'recovery':>10}  "
        f"{'HOPFIELD fid':>13} {'recovery':>10}"
    )
    print(f"  {'-' * 5}  {'-' * 12} {'-' * 10}  {'-' * 13} {'-' * 10}")
    for eps in result["eps_levels"]:
        we = w["per_eps"][str(eps)]
        he = h["per_eps"][str(eps)]
        print(
            f"  {eps:>5}  {we['mean_fidelity']:>12.3f} {we['recovery_rate']:>10.0%}  "
            f"{he['mean_fidelity']:>13.3f} {he['recovery_rate']:>10.0%}"
        )
    print()
    print(f"  Capture radius  : Wheeler eps<={w['capture_radius']:.2f}  "
          f"Hopfield eps<={h['capture_radius']:.2f}")
    print(f"  Mean fidelity   : Wheeler {w['mean_fidelity']:.3f}  "
          f"Hopfield {h['mean_fidelity']:.3f}")
    print(f"  Recovery rate   : Wheeler {w['recovery_rate']:.0%}  "
          f"Hopfield {h['recovery_rate']:.0%}")
    print(f"  Spurious rate   : Wheeler {w['spurious_rate']:.0%}  "
          f"Hopfield {h['spurious_rate']:.0%}")
    print(f"  Mean ticks      : Wheeler {w['mean_ticks']:.1f}  "
          f"Hopfield {h['mean_ticks']:.1f}")
    print()
    print(f"  Pre-registered verdict (capture radius): {result['verdict']}")
    print(f"  Elapsed: {result['elapsed_seconds']}s")
    print("=" * 68)


# --- CLI --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Wheeler CA vs classical Hopfield — pre-registered head-to-head"
    )
    p.add_argument("--json", action="store_true", help="Emit JSON only")
    p.add_argument(
        "--no-save", action="store_true", help="Do not append to substrate_comparison.tsv"
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU dispatch (sets WHEELER_DISABLE_GPU=1)",
    )
    p.add_argument("--notes", default="", help="Free-text notes for the TSV row")
    args = p.parse_args()

    if args.cpu_only:
        os.environ["WHEELER_DISABLE_GPU"] = "1"

    result = run_comparison()

    if args.json:
        print(json.dumps(result))
        return

    _print_human(result)

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

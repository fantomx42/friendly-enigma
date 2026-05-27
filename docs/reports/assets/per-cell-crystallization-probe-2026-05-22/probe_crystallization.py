"""Probe: does per-cell crystallization produce web-like clumps?

Runs evolve_and_interpret() twice on the same seed frame:
  1. baseline — current global-relaxation behavior (CA_CELL_HARDENING_ENABLED=False)
  2. probe    — per-cell soft-freeze + hit-hardening + T-erosion (flag=True)

Reports three metrics and renders a 2x3 matplotlib comparison.

Decision tree (per plan):
  Legs:  n_components >= 5, largest_pct < 0.5, Moran's I in ~[0.2, 0.6]
  Dud:   n_components <= 2 AND largest_pct > 0.8, OR probe ≈ baseline frame.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force CPU dispatch (probe path is CPU-only — GPU kernel is unaware of hardness).
os.environ["WHEELER_DISABLE_GPU"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from wheeler_memory import constants
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.hippocampus import hippocampus_to_frame


ENCODERS = {
    "hash": hash_to_frame,
    "hippocampus": hippocampus_to_frame,
}

SWEEP_SCENARIOS = [
    {"name": "baseline",    "hardening": False, "encoder": "hash"},
    {"name": "probe-def",   "hardening": True,  "encoder": "hash"},
    {"name": "probe-fast",  "hardening": True,  "encoder": "hash",
     "CA_HARDEN_RATE": 0.30, "CA_SETTLE_EPSILON": 0.01},
    {"name": "probe-hippo", "hardening": True,  "encoder": "hippocampus"},
    {"name": "probe-tight", "hardening": True,  "encoder": "hash",
     "stability_threshold": 1e-6, "max_iters": 200},
    {"name": "probe-max",   "hardening": True,  "encoder": "hash",
     "CA_HARDEN_RATE": 0.30, "CA_SETTLE_EPSILON": 0.01,
     "stability_threshold": 1e-6, "max_iters": 200},
]


SEED_TEXT = "the quick brown fox jumps over the lazy dog"
OUT_DIR = Path(__file__).parent.parent / "probe_out"


def morans_I(frame: np.ndarray) -> float:
    """Rook-neighbor (4-conn) spatial autocorrelation via vectorized np.roll.

    Returns Moran's I in [-1, 1]. ~0 = no spatial structure, ~1 = strong
    contiguous clumping, saturated attractors trend high.
    """
    x = frame - frame.mean()
    var = (x * x).sum()
    if var == 0.0:
        return 0.0
    # 4 neighbor products (wrap boundaries — matches CA's np.roll dynamics).
    num = (
        (x * np.roll(x, 1, axis=0)).sum()
        + (x * np.roll(x, -1, axis=0)).sum()
        + (x * np.roll(x, 1, axis=1)).sum()
        + (x * np.roll(x, -1, axis=1)).sum()
    )
    n = frame.size
    w_sum = 4 * n  # 4 neighbors per cell, all weight 1
    return float((n / w_sum) * (num / var))


def component_stats(mask: np.ndarray) -> tuple[int, float, int]:
    """Return (n_components, largest_component_fraction, total_frozen_cells).

    Uses 4-connectivity (rook adjacency). Largest fraction is over total frozen,
    not total grid — answers "is there one dominant blob?".
    """
    total = int(mask.sum())
    if total == 0:
        return 0, 0.0, 0
    labeled, n = label(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    sizes = np.bincount(labeled.ravel())[1:]  # drop background bucket
    return int(n), float(sizes.max() / total), total


def run_one(seed: np.ndarray, hardening_on: bool) -> dict:
    constants.CA_CELL_HARDENING_ENABLED = hardening_on
    t0 = time.perf_counter()
    result = evolve_and_interpret(seed.copy())
    result["_wall_ms"] = (time.perf_counter() - t0) * 1000
    return result


def run_scenario(scenario: dict) -> dict:
    """Apply scenario overrides, run, restore originals, return result.

    Mutates constants in-place for the duration of the call. Always restores
    the originals so subsequent scenarios start clean.
    """
    overrides = {k: v for k, v in scenario.items()
                 if k.startswith("CA_") and hasattr(constants, k)}
    originals = {k: getattr(constants, k) for k in overrides}
    constants.CA_CELL_HARDENING_ENABLED = scenario["hardening"]
    for k, v in overrides.items():
        setattr(constants, k, v)

    encoder = ENCODERS[scenario["encoder"]]
    seed = encoder(SEED_TEXT)

    kwargs = {}
    if "stability_threshold" in scenario:
        kwargs["stability_threshold"] = scenario["stability_threshold"]
    if "max_iters" in scenario:
        kwargs["max_iters"] = scenario["max_iters"]

    t0 = time.perf_counter()
    try:
        result = evolve_and_interpret(seed.copy(), **kwargs)
        result["_wall_ms"] = (time.perf_counter() - t0) * 1000
        result["_seed_frame"] = seed
    finally:
        constants.CA_CELL_HARDENING_ENABLED = False
        for k, v in originals.items():
            setattr(constants, k, v)
    return result


def summarize(label_: str, result: dict) -> dict:
    frame = result["attractor"]
    hardness = result.get("_probe_hardness")
    md = result["metadata"]
    morans = morans_I(frame)
    if hardness is not None:
        frozen = hardness > constants.CA_FROZEN_THRESHOLD
        n_comp, largest_pct, total_frozen = component_stats(frozen)
    else:
        n_comp, largest_pct, total_frozen = 0, 0.0, 0
    return {
        "label": label_,
        "state": result["state"],
        "ticks": result["convergence_ticks"],
        "wall_ms": result["_wall_ms"],
        "alive_fraction": md.get("alive_fraction", float("nan")),
        "frozen_fraction": md.get("frozen_fraction", 0.0),
        "mean_hardness": md.get("mean_hardness", 0.0),
        "n_components": n_comp,
        "largest_pct": largest_pct,
        "total_frozen": total_frozen,
        "morans_I": morans,
        "exit_reason": md.get("exit_reason", "p99_alive"),
        "frame": frame,
        "hardness": hardness,
    }


def render(base: dict, probe: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(base["frame"], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title(f"baseline frame\nticks={base['ticks']} state={base['state']}")
    axes[0, 1].imshow(probe["frame"], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title(
        f"probe frame\nticks={probe['ticks']} state={probe['state']} ({probe['exit_reason']})"
    )
    diff = np.abs(probe["frame"] - base["frame"])
    axes[0, 2].imshow(diff, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title(f"|probe - baseline|\nmean={diff.mean():.3f}")

    hardness = probe["hardness"] if probe["hardness"] is not None else np.zeros_like(probe["frame"])
    axes[1, 0].imshow(hardness, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title(f"probe hardness\nmean={probe['mean_hardness']:.3f}")
    frozen = (hardness > constants.CA_FROZEN_THRESHOLD).astype(float)
    axes[1, 1].imshow(frozen, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"frozen mask (>{constants.CA_FROZEN_THRESHOLD})\nn_comp={probe['n_components']} "
        f"largest={probe['largest_pct']:.2f}"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.95,
        f"seed: {SEED_TEXT!r}\n\n"
        f"Moran's I\n  baseline: {base['morans_I']:+.3f}\n  probe:    {probe['morans_I']:+.3f}\n\n"
        f"alive_fraction\n  baseline: {base['alive_fraction']:.3f}\n  probe:    {probe['alive_fraction']:.3f}\n\n"
        f"frozen_fraction (probe): {probe['frozen_fraction']:.3f}\n"
        f"|Δframe| mean: {diff.mean():.3f}",
        family="monospace", fontsize=9, va="top",
    )
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-cell crystallization probe — baseline vs. probe", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def verdict(base: dict, probe: dict) -> str:
    """Apply the plan's decision rule. Strings only — no side effects."""
    diff_mean = float(np.abs(probe["frame"] - base["frame"]).mean())
    legs = (
        probe["n_components"] >= 5
        and probe["largest_pct"] < 0.5
        and 0.2 <= probe["morans_I"] <= 0.6
        and diff_mean > 0.05
    )
    dud = (
        (probe["n_components"] <= 2 and probe["largest_pct"] > 0.8)
        or diff_mean < 0.01
        or probe["frozen_fraction"] < 0.2
    )
    if legs and not dud:
        return "LEGS — proceed to follow-on persistence plan."
    if dud:
        return "DUD — revert constants block + dynamics edits."
    return "AMBIGUOUS — sweep more inputs or tune CA_HARDEN_RATE / CA_SETTLE_EPSILON."


def print_table(rows: list[dict]) -> None:
    cols = [
        ("label", 14), ("state", 13), ("ticks", 6), ("wall_ms", 9),
        ("alive_fraction", 16), ("frozen_fraction", 17), ("mean_hardness", 15),
        ("n_components", 14), ("largest_pct", 13), ("morans_I", 10),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    print(header); print("-" * len(header))
    for row in rows:
        line = "  ".join(
            f"{row[name]:<{w}.4g}" if isinstance(row[name], float) else f"{str(row[name]):<{w}}"
            for name, w in cols
        )
        print(line)
    print()


def run_sweep() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for scenario in SWEEP_SCENARIOS:
        result = run_scenario(scenario)
        rows.append(summarize(scenario["name"], result))
    print_table(rows)

    base_frame = rows[0]["frame"]
    print(f"{'label':<14}  {'|Δframe| mean':<14}  {'exit_reason':<20}")
    print("-" * 54)
    for row in rows:
        diff = float(np.abs(row["frame"] - base_frame).mean())
        print(f"{row['label']:<14}  {diff:<14.4g}  {row['exit_reason']:<20}")
    print()

    # Coarse legs/dud across the sweep
    probes = [r for r in rows if r["label"] != "baseline"]
    any_legs = any(r["n_components"] >= 5 and r["largest_pct"] < 0.5
                   and r["frozen_fraction"] > 0.1 for r in probes)
    any_strong = any(r["frozen_fraction"] > 0.5 for r in probes)
    if any_legs and any_strong:
        verdict = "LEGS — at least one scenario produces multiple disjoint frozen clumps with substantial coverage."
    elif any_legs:
        verdict = "WEAK LEGS — clumps form but coverage stays low. Worth a follow-on tuning pass before committing."
    else:
        verdict = "DUD across sweep — even aggressive params don't produce clumps. Revert is justified."
    print(f"sweep verdict: {verdict}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-cell crystallization probe")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multi-scenario parameter sweep instead of single comparison.")
    args = parser.parse_args()
    if args.sweep:
        return run_sweep()

    OUT_DIR.mkdir(exist_ok=True)
    seed = hash_to_frame(SEED_TEXT)

    base = summarize("baseline", run_one(seed, hardening_on=False))
    probe = summarize("probe",    run_one(seed, hardening_on=True))

    # Always restore the flag — leaving it on would pollute later test runs.
    constants.CA_CELL_HARDENING_ENABLED = False

    cols = [
        ("label", 10), ("state", 13), ("ticks", 6), ("wall_ms", 9),
        ("alive_fraction", 16), ("frozen_fraction", 17), ("mean_hardness", 15),
        ("n_components", 14), ("largest_pct", 13), ("morans_I", 10),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    print(header); print("-" * len(header))
    for row in (base, probe):
        line = "  ".join(
            f"{row[name]:<{w}.4g}" if isinstance(row[name], float) else f"{str(row[name]):<{w}}"
            for name, w in cols
        )
        print(line)
    print()

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = OUT_DIR / f"crystallization_{ts}.png"
    render(base, probe, out_path)
    print(f"plot: {out_path}")
    print(f"verdict: {verdict(base, probe)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

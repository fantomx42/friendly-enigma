"""Pre-registered SCM-as-waveguide ablation: full Wheeler vs SCM-frozen-open.

Tests whether the SCM gating mechanism earns its complexity. Stores
physics + history facts mixed in one substrate, then runs physics-question
recall in two conditions: full Wheeler (warmed SCM, gating active) vs.
Wheeler with the SCM forced to all-zeros throughout (fully permissive,
no gating). Bootstrap 95% CI on the gap between recall accuracies.

Pre-registered primary criterion (see ``plans/im-not-sure-what-steady-church.md``):

    Full-SCM physics recall is STRICTLY GREATER than SCM-frozen physics
    recall, with the gap excluding zero under bootstrapped 95% CI.

If the gap exceeds CI noise -> SCM earns its keep (the waveguide claim
survives). If the gap is null or negative -> the SCM is decorative on
this task and this configuration.

The hippocampus encoder gives intra-class correlation ~0.43 for
physics-flavoured text and cross-class correlation ~0.07 vs history,
which is the crosstalk regime the SCM is meant to dampen.

Usage
-----
    python scripts/bench/bench_scm_ablation.py
    python scripts/bench/bench_scm_ablation.py --json --bootstrap-n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wheeler_memory.hippocampus import hippocampus_to_frame
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.brick import MemoryBrick
from wheeler_memory.storage import store_memory
from wheeler_memory.interference import recall_with_interference
from wheeler_memory.scm_grid import SCMGrid


# Synthetic facts: 10 physics + 10 history. The hippocampus encoder gives
# ~0.43 intra-class correlation, ~0.07 cross-class -- the SCM is intended
# to dampen the intra-class crosstalk while leaving cross-class alone.

PHYSICS_FACTS = [
    "physics measures the motion of objects under applied force",
    "newton second law states force equals mass times acceleration always",
    "energy is conserved in a closed physical system over time",
    "momentum equals mass multiplied by velocity in classical mechanics",
    "gravity attracts two masses inversely with square of distance",
    "kinetic energy depends on the mass and velocity squared of object",
    "friction force opposes the relative motion between two contact surfaces",
    "work done equals force times displacement along motion direction always",
    "pressure equals force divided by area of contact in fluids",
    "wave frequency multiplied by wavelength equals propagation speed of wave",
]

HISTORY_FACTS = [
    "the battle of waterloo ended napoleon military career in 1815",
    "world war two concluded with allied victory in september 1945",
    "magna carta limited king authority in england during the year 1215",
    "the french revolution began with bastille storming on july 1789",
    "american independence was declared from britain on july fourth 1776",
    "the roman empire fell to germanic tribes around the year 476",
    "columbus arrived in caribbean islands while seeking western route to asia",
    "the cold war divided europe between soviet and western blocs",
    "renaissance art flourished in italy during the fifteenth and sixteenth centuries",
    "the berlin wall fell in november 1989 ending east west division",
]


def _corrupt(text: str, fraction: float, rng: random.Random) -> str:
    """Drop a fraction of words from text -- partial recall cue."""
    words = text.split()
    keep = [w for w in words if rng.random() > fraction]
    return " ".join(keep) if keep else words[0]


def _store_experiential(text: str, data_dir: Path, chunk: str) -> None:
    """Write an experiential attractor npy without touching the corpus index.

    Mirrors ``scm_ab_eval._store_experiential``. store_memory(grid='experiential')
    overwrites the corpus index entry (recall then skips it, storage.py:305-306),
    so we write the npy directly. Without this the SCM has nothing to gate: the
    recall path leaves ``stored_exp = None`` -> the experiential arg to
    ``update_from_recall`` falls back to zeros -> credit = |corpus*exp| = 0
    everywhere -> the gate never fires regardless of the kappa_base/homeostasis
    logic. This is the harness gap the --with-experiential flag closes.
    """
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.dynamics import evolve_with_params
    from wheeler_memory.experiential import experiential_dir
    from wheeler_memory.hashing import text_to_hex
    from wheeler_memory.storage import get_chunk_dir

    frame = hippocampus_to_frame(text)
    result = evolve_with_params(frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW)
    if result["state"] != "CONVERGED":
        return
    chunk_dir = get_chunk_dir(data_dir, chunk)
    exp_dir = experiential_dir(chunk_dir)
    np.save(exp_dir / f"{text_to_hex(text)}.npy", result["attractor"])


def _store_fact(
    text: str, data_dir: Path, chunk: str, with_experiential: bool = False
) -> None:
    frame = hippocampus_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)
    if with_experiential:
        _store_experiential(text, data_dir, chunk)


def _query_top1_hit(
    text: str, data_dir: Path, apply_learning: bool, force_scm_zero: bool
) -> str | None:
    """Run a single interference-based recall; return the top-1 hit text or None.

    If force_scm_zero, we re-zero the persisted SCM grid file BEFORE the
    recall so the loaded grid is fully permissive. Combined with
    apply_learning=False this keeps the SCM out of the comparison entirely.
    """
    if force_scm_zero:
        scm = SCMGrid.load_or_create(data_dir)
        scm.grid = np.zeros_like(scm.grid)
        scm.hardening = np.zeros_like(scm.hardening)
        scm.save()

    results, _state, _openness = recall_with_interference(
        text,
        top_k=1,
        data_dir=data_dir,
        encoder="hippocampus",
        apply_learning=apply_learning,
    )
    if not results:
        return None
    return results[0].get("text")


def _accuracy(
    queries: list[tuple[str, str]],  # (cue, expected_text)
    data_dir: Path,
    apply_learning: bool,
    force_scm_zero: bool,
) -> list[int]:
    """Return per-query 0/1 indicators of top-1 correctness."""
    hits = []
    for cue, expected in queries:
        top = _query_top1_hit(cue, data_dir, apply_learning, force_scm_zero)
        hits.append(1 if top == expected else 0)
    return hits


def _bootstrap_gap_ci(
    a: list[int], b: list[int], n: int, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on mean(a) - mean(b), paired."""
    rng = np.random.default_rng(seed)
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(samples, 2.5)), float(
        np.percentile(samples, 97.5)
    )


def run_ablation(
    corruption: float = 0.3,
    bootstrap_n: int = 1000,
    seed: int = 42,
    warmup_epochs: int = 1,
    with_experiential: bool = False,
) -> dict:
    t0 = time.time()
    rng = random.Random(seed)

    # Build the test queries: each physics fact paired with a corrupted cue.
    queries = [(_corrupt(f, corruption, rng), f) for f in PHYSICS_FACTS]

    # --- Condition 1: physics-only baseline ---------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for f in PHYSICS_FACTS:
            _store_fact(f, d, chunk="science", with_experiential=with_experiential)
        for _ in range(warmup_epochs):
            for cue, _expected in queries:
                _query_top1_hit(cue, d, apply_learning=True, force_scm_zero=False)
        phys_only_hits = _accuracy(queries, d, apply_learning=False, force_scm_zero=False)

    # --- Condition 2: mixed storage, full SCM -------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for f in PHYSICS_FACTS:
            _store_fact(f, d, chunk="science", with_experiential=with_experiential)
        for f in HISTORY_FACTS:
            _store_fact(f, d, chunk="general", with_experiential=with_experiential)
        for _ in range(warmup_epochs):
            for cue, _expected in queries:
                _query_top1_hit(cue, d, apply_learning=True, force_scm_zero=False)
        # Record whether the SCM actually accumulated any state during warmup
        # -- the comparison is only meaningful if the two conditions diverge.
        scm = SCMGrid.load_or_create(d)
        scm_nonzero_cells = int((np.abs(scm.grid) > 0).sum())
        scm_max_abs = float(np.abs(scm.grid).max())
        scm_kappa_base = float(scm.kappa_base)
        mixed_full_hits = _accuracy(queries, d, apply_learning=False, force_scm_zero=False)

    # --- Condition 3: mixed storage, SCM frozen open ------------------------
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for f in PHYSICS_FACTS:
            _store_fact(f, d, chunk="science", with_experiential=with_experiential)
        for f in HISTORY_FACTS:
            _store_fact(f, d, chunk="general", with_experiential=with_experiential)
        # No warmup -- and we force-zero on every query for total isolation.
        mixed_frozen_hits = _accuracy(
            queries, d, apply_learning=False, force_scm_zero=True
        )

    gap, lo, hi = _bootstrap_gap_ci(mixed_full_hits, mixed_frozen_hits, bootstrap_n)
    if lo > 0:
        verdict = "PASS"
    elif hi < 0:
        verdict = "FAIL_NEGATIVE"
    else:
        verdict = "FAIL_NULL"

    # If the SCM never actually accumulated state, the comparison is degenerate
    # (both conditions had SCM=zeros). Flag that explicitly in the result.
    if scm_nonzero_cells == 0:
        verdict = "FAIL_INERT_SCM"

    return {
        "verdict": verdict,
        "physics_only_acc": float(np.mean(phys_only_hits)),
        "mixed_full_acc": float(np.mean(mixed_full_hits)),
        "mixed_frozen_acc": float(np.mean(mixed_frozen_hits)),
        "scm_gap": gap,
        "scm_gap_ci_lo": lo,
        "scm_gap_ci_hi": hi,
        "scm_nonzero_cells": scm_nonzero_cells,
        "scm_max_abs": scm_max_abs,
        "scm_kappa_base": scm_kappa_base,
        "n_physics": len(PHYSICS_FACTS),
        "n_history": len(HISTORY_FACTS),
        "corruption": corruption,
        "warmup_epochs": warmup_epochs,
        "bootstrap_n": bootstrap_n,
        "phys_only_per_query": phys_only_hits,
        "mixed_full_per_query": mixed_full_hits,
        "mixed_frozen_per_query": mixed_frozen_hits,
        "elapsed_seconds": round(time.time() - t0, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- IO ---------------------------------------------------------------------

_RESULTS_TSV = Path(__file__).resolve().parents[2] / "scm_ablation.tsv"
_TSV_HEADER = (
    "iteration\ttimestamp\tcorruption\twarmup_epochs\tphys_only_acc\tmixed_full_acc"
    "\tmixed_frozen_acc\tscm_gap\tci_lo\tci_hi\tscm_nonzero_cells\tscm_max_abs"
    "\tscm_kappa_base\tverdict\telapsed_s\tnotes\n"
)


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
    row = (
        "\t".join(
            [
                str(iteration),
                result["timestamp"],
                f"{result['corruption']:.2f}",
                str(result["warmup_epochs"]),
                f"{result['physics_only_acc']:.4f}",
                f"{result['mixed_full_acc']:.4f}",
                f"{result['mixed_frozen_acc']:.4f}",
                f"{result['scm_gap']:+.4f}",
                f"{result['scm_gap_ci_lo']:+.4f}",
                f"{result['scm_gap_ci_hi']:+.4f}",
                str(result["scm_nonzero_cells"]),
                f"{result['scm_max_abs']:.4g}",
                f"{result['scm_kappa_base']:+.4f}",
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
    print("\n" + "=" * 68)
    print("  SCM-AS-WAVEGUIDE ABLATION  (full Wheeler vs SCM-frozen-open)")
    print("=" * 68)
    print(
        f"  N physics: {result['n_physics']}  N history: {result['n_history']}  "
        f"corruption: {result['corruption']:.0%}  bootstrap N: {result['bootstrap_n']}"
    )
    print()
    print(
        f"  Condition 1 (physics-only, baseline)   : "
        f"{result['physics_only_acc']:.1%}"
    )
    print(
        f"  Condition 2 (mixed, full SCM)          : "
        f"{result['mixed_full_acc']:.1%}"
    )
    print(
        f"  Condition 3 (mixed, SCM frozen open)   : "
        f"{result['mixed_frozen_acc']:.1%}"
    )
    print()
    print(
        f"  SCM gap (cond.2 - cond.3) : {result['scm_gap']:+.1%}  "
        f"[95% CI {result['scm_gap_ci_lo']:+.1%}, {result['scm_gap_ci_hi']:+.1%}]"
    )
    print(
        f"  SCM state after warmup    : nonzero cells = "
        f"{result['scm_nonzero_cells']}/4096  max|x| = {result['scm_max_abs']:.4g}  "
        f"kappa_base = {result['scm_kappa_base']:+.3f}"
    )
    print(f"  Pre-registered verdict    : {result['verdict']}")
    print(f"  Elapsed                   : {result['elapsed_seconds']}s")
    print("=" * 68)


# --- CLI --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="SCM-as-waveguide pre-registered ablation"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument(
        "--corruption", type=float, default=0.3, help="Word-drop fraction (default 0.3)"
    )
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--with-experiential",
        action="store_true",
        help=(
            "Also store experiential attractors so the SCM has interference to "
            "gate (credit>0). Default off keeps the original corpus-only harness "
            "byte-for-byte; with this flag the gate can actually fire."
        ),
    )
    p.add_argument("--notes", default="")
    args = p.parse_args()

    if args.cpu_only:
        os.environ["WHEELER_DISABLE_GPU"] = "1"

    result = run_ablation(
        corruption=args.corruption,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        with_experiential=args.with_experiential,
    )

    if args.json:
        # Strip per-query lists from JSON output -- they're only for human debug.
        slim = {k: v for k, v in result.items() if not k.endswith("_per_query")}
        print(json.dumps(slim))
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

"""CA vs raw-MiniLM reconstruction-delta harness.

Measures whether the three-grid CA recall path adds value over a raw
MiniLM cosine-NN baseline on a context-conditioned retrieval task
(sciq question -> support paragraph). Three arms, same corpus, same queries:

    floor   : raw 384-dim MiniLM cosine-NN  (no projection, no CA)
    ca      : recall_memory(encoder="embedding")  (the production path)
    recon   : recall_memory(..., reconstruct=True, reconstruct_alpha=0.3)

This bench is a *comparator*, not a fallback inside recall. CANON §1.2
forbids the latter; this script never modifies the recall path.

Usage
-----
    pip install -e ".[embed]"
    wheeler-baseline --n 50 --seed 42
    wheeler-baseline --n 5 --seed 42 --no-save   # smoke
    wheeler-baseline --n 50 --seed 42 --json     # machine-readable
    wheeler-baseline --n 50 --seed 42 --output trials.csv

Output
------
Appends one row to baseline.tsv at repo root (mirrors wheeler-bench
convention). Columns: iteration, timestamp, n, seed, dataset, then
{floor,ca,recon}_{top1,top5,mrr}, delta_{top1,top5,mrr} (ca - floor),
{floor,ca,recon}_mean_s, commit, changed, notes.

Notes
-----
- The contender corpus is populated into a tempdir
  (tempfile.mkdtemp(prefix="wheeler-baseline-")). Cleanup in `finally`.
  Orphan dirs: `find /tmp -name 'wheeler-baseline-*' -mtime +1 -delete`.
- chunk="science" is forced for both populate and query, bypassing the
  auto-router. This isolates the CA-vs-encoder question from the routing
  question.
- auto_evict=False is load-bearing: default eviction at higher N would
  silently shrink the contender corpus while the floor stays whole.
- If ~/.wheeler_memory/learned_projection.npy exists, embed_to_frame uses
  it while the floor arm uses raw 384-dim MiniLM. That's the correct
  comparison; do not "fix" it.
- Below --n 50 the score is volatile and rows are not cross-comparable.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np


_REPO_ROOT = Path(__file__).parent.parent
_BASELINE_TSV = _REPO_ROOT / "baseline.tsv"
_DATASETS_DIR = _REPO_ROOT.parent / "datasets"

_TSV_HEADER = (
    "iteration\ttimestamp\tn\tseed\tdataset"
    "\tfloor_top1\tfloor_top5\tfloor_mrr"
    "\tca_top1\tca_top5\tca_mrr"
    "\trecon_top1\trecon_top5\trecon_mrr"
    "\tdelta_top1\tdelta_top5\tdelta_mrr"
    "\tfloor_mean_s\tca_mean_s\trecon_mean_s"
    "\tcommit\tchanged\tnotes\n"
)

# Regex: "Q: <question> A: <answer-sentence>. <support paragraph>"
# The answer is the first token-block ending in "." after "A:".
# Support is everything after that. DOTALL so support may span lines.
_SCIQ_RE = re.compile(r"^Q:\s*(.+?)\s*A:\s*([^.]+\.)\s*(.+)$", re.DOTALL)

_MIN_SUPPORT_LEN = 80  # filter degenerate/short rows


def _detect_git_head() -> str:
    """Return short-8 hash of HEAD, or '' if not a git repo / git missing."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""


def _hash_accel_binaries() -> str:
    accel_root = _REPO_ROOT / "wheeler_memory" / "accel"
    if not accel_root.exists():
        return "none"
    so_files = sorted(accel_root.rglob("*.so"))
    if not so_files:
        return "none"
    h = hashlib.sha256()
    for f in so_files:
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


def _collect_env_info() -> str:
    from wheeler_memory.accel.ca import gpu_available

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return (
        f"[env: gpu={gpu_available()} np={np.__version__} "
        f"py={py} accel={_hash_accel_binaries()}]"
    )


def _next_iteration(path: Path) -> int:
    if not path.exists():
        return 1
    with open(path) as f:
        rows = [l for l in f if l.strip() and not l.startswith("iteration")]
    return len(rows) + 1


def _require_embed() -> None:
    from wheeler_memory.embedding import embed_available

    if not embed_available():
        print(
            "Error: wheeler-baseline requires sentence-transformers.\n"
            "Install with: pip install -e '.[embed]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_sciq_pairs(path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Load N deterministic (question, support) pairs from sciq.jsonl.

    Rejects rows where the Q:/A:/<support> regex fails or support is
    shorter than _MIN_SUPPORT_LEN. Deduplicates on support so each query
    has exactly one truth.
    """
    if not path.exists():
        print(f"Error: dataset not found at {path}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[str, str]] = []
    seen_supports: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            m = _SCIQ_RE.match(text)
            if not m:
                continue
            question = m.group(1).strip()
            support = m.group(3).strip()
            if len(support) < _MIN_SUPPORT_LEN:
                continue
            if support in seen_supports:
                continue
            seen_supports.add(support)
            pairs.append((question, support))

    if len(pairs) < n:
        print(
            f"Error: only {len(pairs)} usable rows in {path}, requested n={n}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    return rng.sample(pairs, n)


def _load_arc_pairs(path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """ARC fallback. Most rows lack a support paragraph; retrieval will
    be degenerate. Provided for symmetry with --dataset arc only.
    """
    if not path.exists():
        print(f"Error: dataset not found at {path}", file=sys.stderr)
        sys.exit(1)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            # Treat full row text as both query and support — degenerate
            # but lets the harness run without error.
            if len(text) < _MIN_SUPPORT_LEN or text in seen:
                continue
            seen.add(text)
            pairs.append((text, text))
    if len(pairs) < n:
        print(
            f"Error: only {len(pairs)} usable rows in {path}, requested n={n}",
            file=sys.stderr,
        )
        sys.exit(1)
    rng = random.Random(seed)
    return rng.sample(pairs, n)


def _build_floor_index(supports: list[str]) -> np.ndarray:
    """Encode supports with raw MiniLM, L2-normalize rows.

    Returns (N, 384) float32 matrix. Cosine similarity later is just
    `matrix @ q_normalized`.
    """
    from wheeler_memory.embedding import embed_text_batch

    mat = embed_text_batch(supports)  # (N, 384)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _query_floor(
    question: str, mat: np.ndarray, supports: list[str]
) -> list[str]:
    """Return supports ranked by raw-MiniLM cosine similarity (high to low)."""
    from wheeler_memory.embedding import embed_text

    q = embed_text(question).astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = mat @ q  # (N,)
    order = np.argsort(-scores)
    return [supports[i] for i in order]


def _populate_contender(supports: list[str], data_dir: Path) -> None:
    """Store supports into data_dir/chunks/science/ via the embedding path."""
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.embedding import embed_to_frame
    from wheeler_memory.storage import store_memory

    for text in supports:
        frame = embed_to_frame(text)
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)
        store_memory(
            text,
            result,
            brick,
            data_dir=data_dir,
            chunk="science",
            auto_evict=False,
        )


def _query_contender(
    question: str,
    data_dir: Path,
    top_k: int,
    reconstruct: bool,
) -> list[str]:
    """Return supports ranked by the CA recall path."""
    from wheeler_memory.storage import recall_memory

    results = recall_memory(
        question,
        top_k=top_k,
        data_dir=data_dir,
        chunk="science",
        encoder="embedding",
        reconstruct=reconstruct,
        reconstruct_alpha=0.3,
        readonly=True,
    )
    return [r["text"] for r in results]


def _rank_of_truth(ranked: list[str], truth: str) -> int | None:
    """1-based rank of truth in ranked list, or None if absent."""
    try:
        return ranked.index(truth) + 1
    except ValueError:
        return None


def _metrics(ranks: list[int | None], latencies: list[float]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "mean_s": 0.0}
    top1 = sum(1 for r in ranks if r == 1) / n
    top5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr = sum((1.0 / r) for r in ranks if r is not None) / n
    mean_s = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "mrr": round(mrr, 4),
        "mean_s": round(mean_s, 4),
    }


def _append_tsv_row(
    row: dict,
    commit: str,
    changed: str,
    notes: str,
) -> None:
    if not _BASELINE_TSV.exists():
        _BASELINE_TSV.write_text(_TSV_HEADER)
    iteration = _next_iteration(_BASELINE_TSV)
    recon = row.get("recon") or {"top1": "", "top5": "", "mrr": "", "mean_s": ""}
    line = (
        "\t".join(
            [
                str(iteration),
                row["timestamp"],
                str(row["n"]),
                str(row["seed"]),
                row["dataset"],
                str(row["floor"]["top1"]),
                str(row["floor"]["top5"]),
                str(row["floor"]["mrr"]),
                str(row["ca"]["top1"]),
                str(row["ca"]["top5"]),
                str(row["ca"]["mrr"]),
                str(recon["top1"]),
                str(recon["top5"]),
                str(recon["mrr"]),
                str(round(row["ca"]["top1"] - row["floor"]["top1"], 4)),
                str(round(row["ca"]["top5"] - row["floor"]["top5"], 4)),
                str(round(row["ca"]["mrr"] - row["floor"]["mrr"], 4)),
                str(row["floor"]["mean_s"]),
                str(row["ca"]["mean_s"]),
                str(recon["mean_s"]),
                commit,
                changed,
                notes,
            ]
        )
        + "\n"
    )
    with open(_BASELINE_TSV, "a") as f:
        f.write(line)


def _print_table(row: dict) -> None:
    floor = row["floor"]
    ca = row["ca"]
    recon = row.get("recon")
    print(f"\n{'=' * 72}")
    print(f"  WHEELER-BASELINE  (n={row['n']}, seed={row['seed']}, dataset={row['dataset']})")
    print(f"{'=' * 72}")
    header = f"  {'metric':<10} {'floor':>10} {'ca':>10}"
    if recon:
        header += f" {'recon':>10}"
    header += f" {'delta_ca':>10}"
    if recon:
        header += f" {'delta_rec':>10}"
    print(header)
    print(f"  {'-' * 70}")
    for key in ("top1", "top5", "mrr"):
        line = f"  {key:<10} {floor[key]:>10.4f} {ca[key]:>10.4f}"
        if recon:
            line += f" {recon[key]:>10.4f}"
        line += f" {ca[key] - floor[key]:>+10.4f}"
        if recon:
            line += f" {recon[key] - floor[key]:>+10.4f}"
        print(line)
    print(f"  {'-' * 70}")
    line = f"  {'mean_s':<10} {floor['mean_s']:>10.4f} {ca['mean_s']:>10.4f}"
    if recon:
        line += f" {recon['mean_s']:>10.4f}"
    print(line)
    print(f"{'=' * 72}")
    if floor["mrr"] < 0.5:
        print("  Note: floor MRR < 0.5 — bench corpus may be harder than the")
        print("        encoder's effective ceiling; delta is informative but")
        print("        bounded by floor headroom.")


def run_baseline(
    n: int,
    seed: int,
    dataset: str,
    top_k: int,
    reconstruct_arm: bool,
    output_csv: Path | None,
    verbose: bool,
) -> dict:
    """Run the harness. Returns the row dict ready for TSV/JSON emission."""
    # ------------------------------------------------------------------
    # This bench measures the system. Do not tune the system to the bench.
    # ------------------------------------------------------------------
    if dataset == "sciq":
        pairs = _load_sciq_pairs(_DATASETS_DIR / "sciq.jsonl", n, seed)
    elif dataset == "arc":
        pairs = _load_arc_pairs(_DATASETS_DIR / "arc.jsonl", n, seed)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    questions = [q for q, _ in pairs]
    supports = [s for _, s in pairs]

    if verbose:
        print(f"  Loaded {n} ({dataset}) pairs, seed={seed}")

    # --- Floor ---
    if verbose:
        print("  Building floor (raw MiniLM cosine-NN)...")
    floor_mat = _build_floor_index(supports)

    floor_ranks: list[int | None] = []
    floor_latencies: list[float] = []
    for q, truth in zip(questions, supports):
        t0 = time.time()
        ranked = _query_floor(q, floor_mat, supports)
        floor_latencies.append(time.time() - t0)
        floor_ranks.append(_rank_of_truth(ranked, truth))

    # --- Contender populate ---
    tmpdir = Path(tempfile.mkdtemp(prefix="wheeler-baseline-"))
    contender_rows: list[dict] = []
    try:
        if verbose:
            print(f"  Populating contender corpus in {tmpdir.name}/ ...")
        _populate_contender(supports, tmpdir)

        # Warmup: first recall amortizes one-time costs (warmth files,
        # association graph init, etc.) outside the timed loop.
        if verbose:
            print("  Warming up recall path...")
        _query_contender(questions[0], tmpdir, top_k=n, reconstruct=False)

        # --- CA arm ---
        if verbose:
            print("  Running CA recall arm...")
        ca_ranks: list[int | None] = []
        ca_latencies: list[float] = []
        ca_top1_texts: list[str] = []
        for q, truth in zip(questions, supports):
            t0 = time.time()
            ranked = _query_contender(q, tmpdir, top_k=n, reconstruct=False)
            ca_latencies.append(time.time() - t0)
            ca_ranks.append(_rank_of_truth(ranked, truth))
            ca_top1_texts.append(ranked[0] if ranked else "")

        # --- CA + reconstruction arm (optional) ---
        recon_ranks: list[int | None] = []
        recon_latencies: list[float] = []
        if reconstruct_arm:
            if verbose:
                print("  Running CA+reconstruction arm...")
            for q, truth in zip(questions, supports):
                t0 = time.time()
                ranked = _query_contender(
                    q, tmpdir, top_k=n, reconstruct=True
                )
                recon_latencies.append(time.time() - t0)
                recon_ranks.append(_rank_of_truth(ranked, truth))

        # Per-query CSV (before tmpdir teardown is fine, but it doesn't
        # depend on tmpdir contents — writing here keeps grouping tidy)
        if output_csv is not None:
            with open(output_csv, "w", newline="") as f:
                w = csv.writer(f)
                hdr = ["q_idx", "floor_rank", "ca_rank"]
                if reconstruct_arm:
                    hdr.append("recon_rank")
                hdr += ["ca_top1_text", "truth"]
                w.writerow(hdr)
                for i in range(n):
                    rec = [
                        i,
                        floor_ranks[i] if floor_ranks[i] is not None else "",
                        ca_ranks[i] if ca_ranks[i] is not None else "",
                    ]
                    if reconstruct_arm:
                        rec.append(
                            recon_ranks[i] if recon_ranks[i] is not None else ""
                        )
                    rec += [ca_top1_texts[i], supports[i]]
                    w.writerow(rec)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    floor_m = _metrics(floor_ranks, floor_latencies)
    ca_m = _metrics(ca_ranks, ca_latencies)
    recon_m = _metrics(recon_ranks, recon_latencies) if reconstruct_arm else None

    return {
        "n": n,
        "seed": seed,
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "floor": floor_m,
        "ca": ca_m,
        "recon": recon_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CA vs raw-MiniLM reconstruction-delta harness"
    )
    parser.add_argument("--n", type=int, default=50, help="number of (q, support) pairs (default 50)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    parser.add_argument(
        "--dataset",
        choices=["sciq", "arc"],
        default="sciq",
        help="dataset (default sciq; arc has no support paragraphs and gives degenerate retrieval)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="(unused for ranking; metrics always computed over all N) — kept for forward compat",
    )
    parser.add_argument(
        "--no-reconstruct",
        action="store_true",
        help="skip the CA+reconstruction arm",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--no-save", action="store_true", help="do not append to baseline.tsv")
    parser.add_argument("--commit", default="", help="git commit hash (auto-detected if empty)")
    parser.add_argument("--changed", default="", help="parameters changed since last run")
    parser.add_argument("--notes", default="", help="free-text notes")
    parser.add_argument("--output", type=Path, default=None, help="per-query CSV path")
    args = parser.parse_args()

    _require_embed()

    verbose = not args.json
    if verbose:
        print(f"\n  wheeler-baseline  n={args.n}  seed={args.seed}  dataset={args.dataset}")

    row = run_baseline(
        n=args.n,
        seed=args.seed,
        dataset=args.dataset,
        top_k=args.top_k,
        reconstruct_arm=not args.no_reconstruct,
        output_csv=args.output,
        verbose=verbose,
    )

    if args.json:
        print(json.dumps(row))
        return

    _print_table(row)

    commit = args.commit or _detect_git_head()
    env_suffix = _collect_env_info()
    notes = f"{args.notes} {env_suffix}".strip() if args.notes else env_suffix

    if not args.no_save:
        _append_tsv_row(row, commit, args.changed, notes)
        print(f"\n  Appended to {_BASELINE_TSV.name}")


if __name__ == "__main__":
    main()

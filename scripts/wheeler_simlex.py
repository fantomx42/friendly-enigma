"""Wheeler SimLex-999 word similarity benchmark.

Evaluates whether CA attractor similarity correlates with human judgments
of word-pair similarity.  SimLex-999 (Hill et al., 2015) tests true
semantic similarity — "car"/"automobile" (similar) vs "car"/"road" (related
but not similar).

Usage
-----
    wheeler-simlex                              # hippocampus + pearson
    wheeler-simlex --encoder embedding          # MiniLM baseline
    wheeler-simlex --mode spatial               # hybrid Pearson+spatial
    wheeler-simlex --sweep --samples 50         # all encoders x modes
    wheeler-simlex --no-evolve                  # raw frames, no CA
    wheeler-simlex --scatter plot.png           # save scatter plot
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

from wheeler_memory.constants import (
    SALIENCE_MAX_ITERS_MED,
    SALIENCE_THRESHOLD_MED,
    SPATIAL_PEARSON_WEIGHT,
)
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.rotation import _get_frame_fn
from wheeler_memory.similarity import (
    hybrid_similarity,
    pearson_similarity,
    spatial_cosine_similarity,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"
SIMLEX_FILENAME = "SimLex-999.txt"
CACHE_SUBDIR = "datasets"

ALL_ENCODERS = [
    "hash",
    "hippocampus",
    "word",
    "hippo-word",
    "context",
    "context-blended",
    "blended",
    "word-blended",
    "language",
    "embedding",
]

# Encoders that don't require optional dependencies
CORE_ENCODERS = [e for e in ALL_ENCODERS if e != "embedding"]

ALL_MODES = ["pearson", "spatial", "features"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _download_simlex(cache_dir: Path) -> Path:
    """Download SimLex-999 zip and extract the TSV."""
    datasets_dir = cache_dir / CACHE_SUBDIR
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = datasets_dir / "simlex999.tsv"

    if tsv_path.exists():
        return tsv_path

    print(f"Downloading SimLex-999 from {SIMLEX_URL} ...")
    try:
        resp = urllib.request.urlopen(SIMLEX_URL, timeout=30)
        data = resp.read()
    except Exception as exc:
        print(
            f"\nError downloading SimLex-999: {exc}\n"
            f"Manual download: {SIMLEX_URL}\n"
            f"Extract {SIMLEX_FILENAME} and save as:\n"
            f"  {tsv_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        # Find the TSV inside the zip (may be in a subdirectory)
        tsv_name = next((n for n in names if n.endswith(SIMLEX_FILENAME)), None)
        if tsv_name is None:
            print(f"Could not find {SIMLEX_FILENAME} in zip. Contents: {names}", file=sys.stderr)
            sys.exit(1)
        content = zf.read(tsv_name)

    tsv_path.write_bytes(content)
    print(f"Cached at {tsv_path}")
    return tsv_path


def load_simlex(cache_dir: Path) -> list[tuple[str, str, float, str]]:
    """Load SimLex-999 as (word1, word2, human_score, pos) tuples.

    Downloads on first call, caches under cache_dir/datasets/.
    """
    tsv_path = _download_simlex(cache_dir)
    pairs = []
    with open(tsv_path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        if "word1" not in header.lower():
            # No header — rewind
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            word1, word2, pos = parts[0], parts[1], parts[2]
            score = float(parts[3])
            pairs.append((word1, word2, score, pos))
    return pairs


# ---------------------------------------------------------------------------
# Attractor cache (per-word, within a single run)
# ---------------------------------------------------------------------------


class WordAttractorCache:
    """Cache evolved attractors by word to avoid redundant CA evolution.

    SimLex-999 has ~1028 unique words across 999 pairs.  Each word is
    evolved at most once, cutting total evolutions roughly in half.
    """

    def __init__(self, frame_fn, evolve: bool = True):
        self._frame_fn = frame_fn
        self._evolve = evolve
        self._cache: dict[str, np.ndarray] = {}

    def get(self, word: str) -> np.ndarray:
        if word not in self._cache:
            frame = self._frame_fn(word)
            if self._evolve:
                result = evolve_and_interpret(
                    frame,
                    max_iters=SALIENCE_MAX_ITERS_MED,
                    stability_threshold=SALIENCE_THRESHOLD_MED,
                )
                self._cache[word] = result["attractor"]
            else:
                self._cache[word] = frame
        return self._cache[word]

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

SIMILARITY_FNS = {
    "pearson": pearson_similarity,
    "spatial": hybrid_similarity,
    "features": spatial_cosine_similarity,
}


def evaluate_simlex(
    pairs: list[tuple[str, str, float, str]],
    encoder: str,
    mode: str,
    pearson_weight: float = SPATIAL_PEARSON_WEIGHT,
    evolve: bool = True,
    verbose: bool = False,
) -> dict:
    """Evaluate Wheeler similarity against SimLex-999 human judgments.

    Returns dict with spearman_rho, spearman_p, pearson_r, pearson_p,
    n_pairs, per-pair details, and per-POS breakdowns.
    """
    frame_fn = _get_frame_fn(encoder=encoder)
    cache = WordAttractorCache(frame_fn, evolve=evolve)

    sim_fn = SIMILARITY_FNS[mode]
    wheeler_scores = []
    human_scores = []
    pair_details = []

    t0 = time.time()
    for i, (w1, w2, hscore, pos) in enumerate(pairs):
        att_a = cache.get(w1)
        att_b = cache.get(w2)

        if mode == "spatial":
            wscore = sim_fn(att_a, att_b, pearson_weight=pearson_weight)
        else:
            wscore = sim_fn(att_a, att_b)

        wheeler_scores.append(wscore)
        human_scores.append(hscore)
        pair_details.append({
            "word1": w1,
            "word2": w2,
            "human": hscore,
            "wheeler": wscore,
            "pos": pos,
        })

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(pairs)} pairs ({cache.size} unique words, {elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Global correlations
    w = np.array(wheeler_scores)
    h = np.array(human_scores)
    sp_rho, sp_p = stats.spearmanr(w, h)
    pr_r, pr_p = stats.pearsonr(w, h)

    # Per-POS breakdowns
    pos_results = {}
    for pos_tag in ("N", "V", "A"):
        mask = [d["pos"] == pos_tag for d in pair_details]
        if sum(mask) > 2:
            wp = w[mask]
            hp = h[mask]
            rho, _ = stats.spearmanr(wp, hp)
            pos_results[pos_tag] = {"rho": float(rho), "n": sum(mask)}

    return {
        "spearman_rho": float(sp_rho),
        "spearman_p": float(sp_p),
        "pearson_r": float(pr_r),
        "pearson_p": float(pr_p),
        "n_pairs": len(pairs),
        "n_unique_words": cache.size,
        "elapsed_s": elapsed,
        "pos_breakdown": pos_results,
        "pairs": pair_details,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results(result: dict, encoder: str, mode: str, evolve: bool = True) -> None:
    """Print benchmark results in banner format."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  WHEELER SIMLEX-999 BENCHMARK")
    evo_str = "evolved" if evolve else "raw frames"
    print(f"  Encoder: {encoder}   Mode: {mode}   Pairs: {result['n_pairs']}   ({evo_str})")
    print(sep)
    print()
    print(f"  Spearman rho:  {result['spearman_rho']:+.4f}  (p = {result['spearman_p']:.2e})")
    print(f"  Pearson r:     {result['pearson_r']:+.4f}  (p = {result['pearson_p']:.2e})")
    print(f"  Unique words:  {result['n_unique_words']}  in {result['elapsed_s']:.1f}s")

    if result["pos_breakdown"]:
        print()
        print("  Per-POS breakdown:")
        pos_names = {"N": "Nouns", "V": "Verbs", "A": "Adjectives"}
        for pos_tag in ("N", "V", "A"):
            if pos_tag in result["pos_breakdown"]:
                pr = result["pos_breakdown"][pos_tag]
                name = pos_names[pos_tag]
                print(f"    {name:14s} ({pr['n']:3d} pairs):  rho = {pr['rho']:+.4f}")

    print(f"\n{sep}\n")


def print_sweep(sweep_results: list[dict]) -> None:
    """Print sweep comparison table."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ENCODER x MODE SWEEP")
    n = sweep_results[0]["n_pairs"] if sweep_results else 0
    evo = "evolved" if sweep_results[0].get("evolve", True) else "raw"
    print(f"  {n} pairs, {evo}")
    print(sep)
    print()

    # Group by encoder
    encoders_seen = []
    by_encoder: dict[str, dict[str, float]] = {}
    for r in sweep_results:
        enc = r["encoder"]
        if enc not in by_encoder:
            by_encoder[enc] = {}
            encoders_seen.append(enc)
        by_encoder[enc][r["mode"]] = r["spearman_rho"]

    # Header
    print(f"  {'Encoder':<16s}  {'Pearson':>8s}  {'Spatial':>8s}  {'Features':>8s}")
    print(f"  {'-' * 16}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for enc in encoders_seen:
        modes = by_encoder[enc]
        vals = []
        for m in ALL_MODES:
            if m in modes:
                vals.append(f"{modes[m]:+.4f}")
            else:
                vals.append("   --   ")
        print(f"  {enc:<16s}  {vals[0]:>8s}  {vals[1]:>8s}  {vals[2]:>8s}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------


def save_scatter(result: dict, path: str, encoder: str, mode: str) -> None:
    """Save human-vs-Wheeler scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = result["pairs"]
    colors = {"N": "#4477AA", "V": "#EE6677", "A": "#228833"}
    labels = {"N": "Noun", "V": "Verb", "A": "Adjective"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for pos_tag in ("N", "V", "A"):
        xs = [p["human"] for p in pairs if p["pos"] == pos_tag]
        ys = [p["wheeler"] for p in pairs if p["pos"] == pos_tag]
        if xs:
            ax.scatter(xs, ys, c=colors[pos_tag], label=labels[pos_tag], alpha=0.5, s=12)

    ax.set_xlabel("Human similarity (SimLex-999)")
    ax.set_ylabel("Wheeler similarity")
    ax.set_title(
        f"SimLex-999: {encoder} / {mode}\n"
        f"Spearman rho = {result['spearman_rho']:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to {path}")


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------


def save_tsv(result: dict, path: str) -> None:
    """Save per-pair results to TSV."""
    with open(path, "w") as f:
        f.write("word1\tword2\tpos\thuman\twheeler\n")
        for p in result["pairs"]:
            f.write(f"{p['word1']}\t{p['word2']}\t{p['pos']}\t{p['human']}\t{p['wheeler']:.6f}\n")
    print(f"Per-pair results saved to {path}")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


def run_sweep(
    pairs: list[tuple[str, str, float, str]],
    evolve: bool = True,
    pearson_weight: float = SPATIAL_PEARSON_WEIGHT,
) -> list[dict]:
    """Run all encoder x mode combinations and print comparison table."""
    sweep_results = []

    # Check which encoders are available
    available = list(CORE_ENCODERS)
    try:
        _get_frame_fn(encoder="embedding")
        available.append("embedding")
    except (ImportError, Exception):
        print("  (skipping 'embedding' — sentence-transformers not installed)")

    total = len(available) * len(ALL_MODES)
    done = 0

    for enc in available:
        for mode in ALL_MODES:
            done += 1
            print(f"  [{done}/{total}] {enc:16s} / {mode} ...", end="", flush=True)
            t0 = time.time()
            result = evaluate_simlex(
                pairs, enc, mode,
                pearson_weight=pearson_weight,
                evolve=evolve,
            )
            elapsed = time.time() - t0
            print(f"  rho = {result['spearman_rho']:+.4f}  ({elapsed:.1f}s)")
            sweep_results.append({
                "encoder": enc,
                "mode": mode,
                "evolve": evolve,
                "n_pairs": result["n_pairs"],
                "spearman_rho": result["spearman_rho"],
                "pearson_r": result["pearson_r"],
            })

    print_sweep(sweep_results)
    return sweep_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="wheeler-simlex",
        description="Evaluate Wheeler Memory word similarity against SimLex-999.",
    )
    p.add_argument(
        "--encoder",
        choices=ALL_ENCODERS,
        default="hippocampus",
        help="Encoder for text-to-frame (default: hippocampus)",
    )
    p.add_argument(
        "--mode",
        choices=ALL_MODES,
        default="pearson",
        help="Similarity mode: pearson, spatial (hybrid), features (topology only)",
    )
    p.add_argument(
        "--no-evolve",
        action="store_true",
        help="Compare raw seed frames without CA evolution (diagnostic)",
    )
    p.add_argument(
        "--pearson-weight",
        type=float,
        default=None,
        help=f"Pearson weight for spatial mode (default: {SPATIAL_PEARSON_WEIGHT})",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run all encoders x all modes and print comparison table",
    )
    p.add_argument(
        "--pos",
        choices=["N", "V", "A"],
        help="Filter by POS tag (N=nouns, V=verbs, A=adjectives)",
    )
    p.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Limit to first N pairs (for quick testing)",
    )
    p.add_argument(
        "--scatter",
        metavar="FILE",
        help="Save scatter plot (human vs Wheeler similarity) to PNG",
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Save per-pair results to TSV",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Data directory (default: ~/.wheeler_memory)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".wheeler_memory"
    pairs = load_simlex(data_dir)
    print(f"Loaded {len(pairs)} SimLex-999 pairs")

    # Optional POS filter
    if args.pos:
        pairs = [(w1, w2, s, p) for w1, w2, s, p in pairs if p == args.pos]
        print(f"Filtered to {len(pairs)} {args.pos} pairs")

    # Optional sample limit
    if args.samples:
        pairs = pairs[: args.samples]
        print(f"Limited to {len(pairs)} pairs")

    if not pairs:
        print("No pairs to evaluate.", file=sys.stderr)
        sys.exit(1)

    evolve = not args.no_evolve
    pw = args.pearson_weight if args.pearson_weight is not None else SPATIAL_PEARSON_WEIGHT

    if args.sweep:
        run_sweep(pairs, evolve=evolve, pearson_weight=pw)
        return

    # Single evaluation
    result = evaluate_simlex(
        pairs,
        args.encoder,
        args.mode,
        pearson_weight=pw,
        evolve=evolve,
        verbose=args.verbose,
    )

    print_results(result, args.encoder, args.mode, evolve=evolve)

    if args.output:
        save_tsv(result, args.output)
    if args.scatter:
        save_scatter(result, args.scatter, args.encoder, args.mode)


if __name__ == "__main__":
    main()

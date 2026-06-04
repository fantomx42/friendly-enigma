"""CA vs raw-MiniLM passage re-ranking harness (TREC DL 2019 / MS MARCO).

Measures whether the three-grid CA recall path adds value over raw MiniLM
cosine-NN on the gold-standard MS MARCO passage re-ranking task, where
published all-MiniLM-L6-v2 nDCG@10 ≈ 0.65-0.70 — clearly sub-ceiling.

This bench is the successor to wheeler-baseline. The sciq result was at-ceiling
(mean floor top-1 = 0.965, two seeds at 1.00), so the CA delta was bounded by
1−floor ≈ 0 by construction. TREC DL 2019 gives the encoder genuine headroom.

Two arms, both restricted to per-query top-100 BM25 candidates:

    floor : raw 384-dim MiniLM cosine-NN over candidate subset
    ca    : recall_memory(encoder="embedding", readonly=True), filter to candidates

Reconstruction arm is intentionally absent — provably inert across 200 sciq
calls (recon == ca on all metrics). Conclusion captured; no further data needed.

Usage
-----
    pip install -e ".[embed,rerank]"
    wheeler-rerank --queries 3 --top-k 100 --no-save     # smoke (~1 min cache-warm)
    wheeler-rerank --queries 43 --top-k 100              # canonical, all judged
    wheeler-rerank --queries 43 --top-k 100 --output rerank_per_query.csv

Output
------
Appends one row to rerank.tsv at repo root. Columns: iteration, timestamp,
n_queries, top_k_candidates, n_unique_passages, then {floor,ca}_{mrr10,ndcg10,
recall100}, delta_{mrr10,ndcg10,recall100} (ca − floor), {floor,ca}_mean_s,
populate_s, commit, changed, notes.

Notes
-----
- First run downloads ~3GB MS MARCO collection to ~/.ir_datasets/ (cached
  thereafter). Set IR_DATASETS_HOME to override on read-only mounts.
- Contender corpus populated ONCE across deduplicated candidates (~3-5k unique
  passages from 43×100 with ~30-40% overlap), in a tempdir cleaned in finally.
- chunk="science" forced. Mirrors wheeler-baseline isolation rationale.
- Floor uses embed_text (raw 384-dim, no projection). CA uses embed_to_frame
  (projected 4096-vec → CA settle). If ~/.wheeler_memory/learned_projection.npy
  exists, CA path picks it up but floor doesn't — that's the correct asymmetry.
- Score values for ir_measures use reciprocal rank (1/(rank+1)) since CA's
  Pearson similarities and floor's cosines aren't on the same scale — only
  rank order is the measurand.
- Candidates CA's top-K return omits are appended at the tail in sorted-docid
  order. Honest: missing-at-tail can only hurt CA's metrics, never help.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).parent.parent
_RERANK_TSV = _REPO_ROOT / "rerank.tsv"
_DATASET_ID = "msmarco-passage/trec-dl-2019/judged"
_CORPUS_ID = "msmarco-passage"
_BM25_FALLBACK = _REPO_ROOT / "runs" / "dl2019.bm25.txt"

_TSV_HEADER = (
    "iteration\ttimestamp\tn_queries\ttop_k_candidates\tn_unique_passages"
    "\tfloor_mrr10\tca_mrr10\tdelta_mrr10"
    "\tfloor_ndcg10\tca_ndcg10\tdelta_ndcg10"
    "\tfloor_recall100\tca_recall100\tdelta_recall100"
    "\tfloor_mean_s\tca_mean_s\tpopulate_s"
    "\tcommit\tchanged\tnotes\n"
)


# --- env / git helpers (mirror wheeler_baseline.py) --------------------------

def _detect_git_head() -> str:
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


# --- gating ------------------------------------------------------------------

def _require_embed() -> None:
    from wheeler_memory.embedding import embed_available

    if not embed_available():
        print(
            "Error: wheeler-rerank requires sentence-transformers.\n"
            "Install with: pip install -e '.[embed,rerank]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_rerank_deps() -> None:
    try:
        import ir_datasets  # noqa: F401
        import ir_measures  # noqa: F401
    except ImportError:
        print(
            "Error: wheeler-rerank requires ir-datasets and ir-measures.\n"
            "Install with: pip install -e '.[embed,rerank]'",
            file=sys.stderr,
        )
        sys.exit(1)


# --- TREC DL 2019 loading ----------------------------------------------------

def _load_dl2019_judged(
    top_k_cands: int,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, int]]]:
    """Return (queries, candidates_by_qid, qrels).

    Candidates sourced via ir_datasets scoreddocs (preferred), or
    runs/dl2019.bm25.txt fallback. Fails loud if neither — silent
    substitution of a smaller candidate pool would make TSV rows incomparable.
    """
    import ir_datasets

    try:
        ds = ir_datasets.load(_DATASET_ID)
    except OSError as e:
        print(
            f"Error loading {_DATASET_ID}: {e}\n"
            "If ~/.ir_datasets/ is on a read-only mount, set IR_DATASETS_HOME.",
            file=sys.stderr,
        )
        sys.exit(1)

    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for rel in ds.qrels_iter():
        qrels[rel.query_id][rel.doc_id] = int(rel.relevance)

    # Drop queries with no positive judgments (standard TREC DL practice)
    judged_positive = {
        qid for qid, docs in qrels.items() if any(r >= 1 for r in docs.values())
    }
    queries = {qid: q for qid, q in queries.items() if qid in judged_positive}
    qrels = {qid: d for qid, d in qrels.items() if qid in judged_positive}

    candidates_by_qid: dict[str, list[str]] = {}

    if ds.has_scoreddocs():
        per_qid: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sd in ds.scoreddocs_iter():
            if sd.query_id in queries:
                per_qid[sd.query_id].append((sd.doc_id, sd.score))
        for qid, scored in per_qid.items():
            scored.sort(key=lambda t: -t[1])
            candidates_by_qid[qid] = [d for d, _ in scored[:top_k_cands]]
    elif _BM25_FALLBACK.exists():
        # TREC run format: qid Q0 docid rank score tag
        per_qid_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with open(_BM25_FALLBACK) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                qid, _, docid, rank, _, _ = parts[:6]
                if qid in queries:
                    per_qid_lines[qid].append((int(rank), docid))
        for qid, ranked in per_qid_lines.items():
            ranked.sort()
            candidates_by_qid[qid] = [d for _, d in ranked[:top_k_cands]]
    else:
        print(
            f"Error: no candidate source available.\n"
            f"  - ir_datasets has_scoreddocs() = False for {_DATASET_ID}\n"
            f"  - fallback {_BM25_FALLBACK} not present\n"
            "Generate a BM25 run via pyserini or download the official MS MARCO\n"
            "leaderboard run file. Do not silently degrade to a smaller candidate pool.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Final filter: any query that lost its candidates (shouldn't happen if
    # scoreddocs covers all judged queries, but defensive)
    valid_qids = [qid for qid in queries if candidates_by_qid.get(qid)]
    queries = {qid: queries[qid] for qid in valid_qids}
    candidates_by_qid = {qid: candidates_by_qid[qid] for qid in valid_qids}
    qrels = {qid: qrels[qid] for qid in valid_qids}

    return queries, candidates_by_qid, qrels


def _resolve_passages(doc_ids: set[str]) -> dict[str, str]:
    """Random-access lookup of passage text via ir_datasets docs_store."""
    import ir_datasets

    print(
        f"  Resolving {len(doc_ids)} unique passages from MS MARCO corpus...\n"
        f"  (first run downloads ~3GB to ~/.ir_datasets/, cached thereafter)"
    )
    corpus = ir_datasets.load(_CORPUS_ID)
    store = corpus.docs_store()
    out: dict[str, str] = {}
    for d in doc_ids:
        doc = store.get(d)
        if doc is not None:
            out[d] = doc.text
    return out


# --- floor arm (raw MiniLM cosine over per-query candidates) -----------------

def _build_floor_matrix(
    doc_ids_ordered: list[str], doc_texts: dict[str, str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode all unique passages once with raw MiniLM, L2-normalize.

    Returns (matrix (N,384), doc_id → row index).
    """
    from wheeler_memory.embedding import embed_text_batch

    texts = [doc_texts[d] for d in doc_ids_ordered]
    mat = embed_text_batch(texts).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    mat_index = {d: i for i, d in enumerate(doc_ids_ordered)}
    return mat, mat_index


def _run_floor(
    qtext: str,
    cand_ids: list[str],
    floor_mat: np.ndarray,
    mat_index: dict[str, int],
) -> list[str]:
    """Cosine-rank candidates for one query. Returns doc_ids ranked high→low."""
    from wheeler_memory.embedding import embed_text

    rows = [mat_index[d] for d in cand_ids if d in mat_index]
    used_ids = [d for d in cand_ids if d in mat_index]
    if not rows:
        return []
    submat = floor_mat[rows]
    q = embed_text(qtext).astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = submat @ q
    order = np.argsort(-scores)
    return [used_ids[i] for i in order]


# --- contender arm (CA recall_memory, filtered to candidates) ----------------

def _populate_contender(
    doc_ids_ordered: list[str], doc_texts: dict[str, str], data_dir: Path
) -> float:
    """Batched embed → batched evolve → batched store, all into chunk='science'.

    Returns populate wall-clock seconds.
    """
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.dynamics import evolve_batch
    from wheeler_memory.embedding import embed_to_frame_batch
    from wheeler_memory.storage import batch_store_memories

    t0 = time.time()
    texts = [doc_texts[d] for d in doc_ids_ordered]
    frames = embed_to_frame_batch(texts)
    results = evolve_batch(frames)
    entries = [
        (text, result, MemoryBrick.from_evolution_result(result), "science")
        for text, result in zip(texts, results)
    ]
    batch_store_memories(entries, data_dir=data_dir)
    return time.time() - t0


def _run_ca(
    qtext: str,
    cand_ids: list[str],
    data_dir: Path,
    n_unique: int,
    text_to_id: dict[str, str],
) -> list[str]:
    """Rank candidates via recall_memory, filtered+ordered by CA's ranking.

    Candidates CA's top-K return omits are appended at the tail in sorted-docid
    order. Honest: missing-at-tail can only hurt CA's metrics, never help.
    """
    from wheeler_memory.storage import recall_memory

    results = recall_memory(
        qtext,
        top_k=n_unique,
        data_dir=data_dir,
        chunk="science",
        encoder="embedding",
        readonly=True,
    )
    cand_set = set(cand_ids)
    ranked: list[str] = []
    seen: set[str] = set()
    for r in results:
        doc_id = text_to_id.get(r["text"])
        if doc_id is not None and doc_id in cand_set and doc_id not in seen:
            ranked.append(doc_id)
            seen.add(doc_id)
    # Append candidates CA didn't return, in sorted-docid order
    missing = sorted(d for d in cand_ids if d not in seen)
    return ranked + missing


# --- metrics -----------------------------------------------------------------

def _compute_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    latencies: list[float],
) -> dict:
    """Wrap ir_measures.calc_aggregate for MRR@10, nDCG@10, R@100."""
    from ir_measures import MRR, R, calc_aggregate, nDCG

    vals = calc_aggregate([MRR @ 10, nDCG @ 10, R @ 100], qrels, run)
    # ir_measures returns Measure -> float mapping; key by repr
    out = {str(k): float(v) for k, v in vals.items()}
    return {
        "mrr10": round(out.get("RR@10", 0.0), 4),
        "ndcg10": round(out.get("nDCG@10", 0.0), 4),
        "recall100": round(out.get("R@100", 0.0), 4),
        "mean_s": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
    }


# --- TSV / table -------------------------------------------------------------

def _append_tsv_row(
    row: dict, commit: str, changed: str, notes: str
) -> None:
    if not _RERANK_TSV.exists():
        _RERANK_TSV.write_text(_TSV_HEADER)
    iteration = _next_iteration(_RERANK_TSV)
    line = (
        "\t".join(
            [
                str(iteration),
                row["timestamp"],
                str(row["n_queries"]),
                str(row["top_k_candidates"]),
                str(row["n_unique_passages"]),
                str(row["floor"]["mrr10"]),
                str(row["ca"]["mrr10"]),
                str(round(row["ca"]["mrr10"] - row["floor"]["mrr10"], 4)),
                str(row["floor"]["ndcg10"]),
                str(row["ca"]["ndcg10"]),
                str(round(row["ca"]["ndcg10"] - row["floor"]["ndcg10"], 4)),
                str(row["floor"]["recall100"]),
                str(row["ca"]["recall100"]),
                str(round(row["ca"]["recall100"] - row["floor"]["recall100"], 4)),
                str(row["floor"]["mean_s"]),
                str(row["ca"]["mean_s"]),
                str(row["populate_s"]),
                commit,
                changed,
                notes,
            ]
        )
        + "\n"
    )
    with open(_RERANK_TSV, "a") as f:
        f.write(line)


def _print_table(row: dict) -> None:
    floor = row["floor"]
    ca = row["ca"]
    print(f"\n{'=' * 72}")
    print(
        f"  WHEELER-RERANK  (queries={row['n_queries']}, "
        f"top_k={row['top_k_candidates']}, "
        f"corpus={row['n_unique_passages']} passages)"
    )
    print(f"{'=' * 72}")
    print(f"  {'metric':<12} {'floor':>10} {'ca':>10} {'delta':>10}")
    print(f"  {'-' * 70}")
    for key, label in (("mrr10", "MRR@10"), ("ndcg10", "nDCG@10"), ("recall100", "R@100")):
        d = ca[key] - floor[key]
        print(
            f"  {label:<12} {floor[key]:>10.4f} {ca[key]:>10.4f} {d:>+10.4f}"
        )
    print(f"  {'-' * 70}")
    print(f"  {'mean_s':<12} {floor['mean_s']:>10.4f} {ca['mean_s']:>10.4f}")
    print(f"  {'populate_s':<12} {'':>10} {row['populate_s']:>10.4f}")
    print(f"{'=' * 72}")
    if floor["ndcg10"] < 0.25 or floor["ndcg10"] > 0.55:
        print(
            f"  Warning: floor nDCG@10 = {floor['ndcg10']:.4f} outside zero-shot\n"
            "           all-MiniLM-L6-v2 bi-encoder cosine range [0.25, 0.55] on\n"
            "           TREC DL 2019. (Cross-encoder MiniLM trained on MS MARCO\n"
            "           scores ~0.69, but this bench uses raw bi-encoder cosine.)\n"
            "           Suspect embedding-config bug before trusting the CA delta."
        )
    if abs(ca["recall100"] - floor["recall100"]) > 1e-6:
        print(
            f"  Warning: Recall@100 differs between arms (floor={floor['recall100']:.4f},\n"
            f"           ca={ca['recall100']:.4f}). Both arms re-rank the same BM25\n"
            "           candidate pool — R@100 should be identical. Suspect a\n"
            "           candidate-filter or lookup bug."
        )


# --- main --------------------------------------------------------------------

def run_rerank(
    n_queries: int,
    top_k: int,
    seed: int,
    output_csv: Path | None,
    verbose: bool,
) -> dict:
    """Run the rerank harness. Returns the row dict for TSV/JSON emission."""
    # ------------------------------------------------------------------
    # This bench measures the system. Do not tune the system to the bench.
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Loading {_DATASET_ID} (queries + qrels + scoreddocs)...")
    queries, candidates_by_qid, qrels = _load_dl2019_judged(top_k_cands=top_k)

    all_qids = sorted(queries)
    if n_queries < len(all_qids):
        rng = random.Random(seed)
        all_qids = rng.sample(all_qids, n_queries)
        queries = {qid: queries[qid] for qid in all_qids}
        candidates_by_qid = {qid: candidates_by_qid[qid] for qid in all_qids}
        qrels = {qid: qrels[qid] for qid in all_qids}
    n_queries_actual = len(queries)

    # Union of candidates across queries (deduplicated)
    all_doc_ids: set[str] = set()
    for cs in candidates_by_qid.values():
        all_doc_ids.update(cs)
    doc_texts = _resolve_passages(all_doc_ids)
    missing = all_doc_ids - set(doc_texts)
    if missing:
        print(
            f"  Warning: {len(missing)} candidate doc_ids missing from corpus "
            "(skipped — should be 0 on a clean MS MARCO cache)"
        )
    doc_ids_ordered = sorted(doc_texts)
    text_to_id = {t: d for d, t in doc_texts.items()}

    if verbose:
        print(f"  Resolved {len(doc_ids_ordered)} unique passages, {n_queries_actual} queries")
        print("  Building floor matrix (raw MiniLM cosine, single batch)...")
    floor_mat, mat_index = _build_floor_matrix(doc_ids_ordered, doc_texts)

    # --- Floor arm ---
    floor_run: dict[str, dict[str, float]] = {}
    floor_lat: list[float] = []
    for qid in all_qids:
        cand_ids = candidates_by_qid[qid]
        t0 = time.time()
        ranked = _run_floor(queries[qid], cand_ids, floor_mat, mat_index)
        floor_lat.append(time.time() - t0)
        floor_run[qid] = {d: 1.0 / (rank + 1) for rank, d in enumerate(ranked)}

    # --- Contender ---
    tmpdir = Path(tempfile.mkdtemp(prefix="wheeler-rerank-"))
    populate_s = 0.0
    try:
        if verbose:
            print(f"  Populating contender corpus ({len(doc_ids_ordered)} passages)...")
        populate_s = _populate_contender(doc_ids_ordered, doc_texts, tmpdir)
        if verbose:
            print(f"  Populate done in {populate_s:.1f}s. Warming up recall path...")
        # Warmup so first-call costs don't land on query #0
        _run_ca(
            queries[all_qids[0]],
            candidates_by_qid[all_qids[0]],
            tmpdir,
            len(doc_ids_ordered),
            text_to_id,
        )
        if verbose:
            print("  Running CA arm...")
        ca_run: dict[str, dict[str, float]] = {}
        ca_lat: list[float] = []
        ca_top1: dict[str, str] = {}
        for qid in all_qids:
            cand_ids = candidates_by_qid[qid]
            t0 = time.time()
            ranked = _run_ca(
                queries[qid], cand_ids, tmpdir, len(doc_ids_ordered), text_to_id
            )
            ca_lat.append(time.time() - t0)
            ca_run[qid] = {d: 1.0 / (rank + 1) for rank, d in enumerate(ranked)}
            ca_top1[qid] = ranked[0] if ranked else ""

        if output_csv is not None:
            with open(output_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "qid",
                        "query",
                        "n_candidates",
                        "n_positives",
                        "floor_top1",
                        "ca_top1",
                        "floor_top1_rel",
                        "ca_top1_rel",
                    ]
                )
                for qid in all_qids:
                    floor_top1_id = next(iter(floor_run[qid]), "")
                    ca_top1_id = ca_top1[qid]
                    w.writerow(
                        [
                            qid,
                            queries[qid],
                            len(candidates_by_qid[qid]),
                            sum(1 for r in qrels[qid].values() if r >= 1),
                            floor_top1_id,
                            ca_top1_id,
                            qrels[qid].get(floor_top1_id, 0),
                            qrels[qid].get(ca_top1_id, 0),
                        ]
                    )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    floor_m = _compute_metrics(qrels, floor_run, floor_lat)
    ca_m = _compute_metrics(qrels, ca_run, ca_lat)

    return {
        "n_queries": n_queries_actual,
        "top_k_candidates": top_k,
        "n_unique_passages": len(doc_ids_ordered),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "populate_s": round(populate_s, 2),
        "floor": floor_m,
        "ca": ca_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CA vs raw-MiniLM passage re-ranking (TREC DL 2019)"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=43,
        help="number of TREC DL 2019 judged queries to run (default 43 = all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="BM25 candidates per query for re-ranking (default 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for query subsampling when --queries < 43 (default 42)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--no-save", action="store_true", help="do not append to rerank.tsv")
    parser.add_argument("--commit", default="", help="git commit hash (auto-detected if empty)")
    parser.add_argument("--changed", default="", help="parameters changed since last run")
    parser.add_argument("--notes", default="", help="free-text notes")
    parser.add_argument("--output", type=Path, default=None, help="per-query CSV path")
    args = parser.parse_args()

    _require_embed()
    _require_rerank_deps()

    verbose = not args.json
    if verbose:
        print(
            f"\n  wheeler-rerank  queries={args.queries}  "
            f"top_k={args.top_k}  seed={args.seed}"
        )

    row = run_rerank(
        n_queries=args.queries,
        top_k=args.top_k,
        seed=args.seed,
        output_csv=args.output,
        verbose=verbose,
    )

    if args.json:
        print(json.dumps(row))
        return

    _print_table(row)

    commit = args.commit or _detect_git_head()
    env_suffix = _collect_env_info()
    note_parts = [args.notes] if args.notes else []
    if args.queries < 43:
        note_parts.append(f"[subsampled queries={args.queries} seed={args.seed}]")
    note_parts.append(env_suffix)
    notes = " ".join(p for p in note_parts if p)

    if not args.no_save:
        _append_tsv_row(row, commit, args.changed, notes)
        print(f"\n  Appended to {_RERANK_TSV.name}")


if __name__ == "__main__":
    main()

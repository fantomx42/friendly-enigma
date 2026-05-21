"""50-passage closed-loop A/B: Pearson vs. frozen-SCM vs. learning-SCM.

Arms
----
pearson      : pure Pearson recall (no SCM involvement)
frozen_scm   : Pearson candidates re-ranked by interference score with
               a fully-open frozen SCM (zeros, never updated)
learning_scm : Pearson candidates re-ranked by interference score with
               an SCM that learns from each recall outcome (closed loop)

Two-phase design
----------------
Phase 1 — Warmup (exact queries, not scored):
  Run the 50 stored passages through learning_scm using their exact Q-part
  queries so kappa_base builds via EMA before the eval phase begins.
  After 50 exact queries at kappa ~1.2, kappa_base ≈ 1.19.
  Frozen SCM and Pearson arm are stateless and unaffected.

Phase 2 — Paraphrase eval (all three arms scored):
  Query with content-word-shuffled paraphrases. The hippocampus encoder is
  character n-gram based, so word-order and boundary changes reduce Pearson
  similarity substantially. With kappa_base ~1.19 and paraphrase kappa ~0.4-0.8,
  advantage = kappa − kappa_base < 0, the homeostasis ceiling is bypassed, and
  the cold-start seeding path activates for learning_scm.

Why paraphrase over known corpus (not lower TOP_K, not new corpus):
  Stored passages unchanged → storage and experiential paths are identical.
  Query-side encoder drift forces recall misses without introducing corpus
  confounds. This is the minimal perturbation that makes the A/B diagnostic.

Paraphrase transform
--------------------
Remove stop words + shuffle word order. No external models; deterministic
given the RNG seed. Character n-gram overlap drops substantially with
word-order changes.

Corpus  : 50 passages from datasets/arc.jsonl, seeded random sample.
Encoder : hippocampus (character n-gram RI).

Metrics (paraphrase eval phase)
--------------------------------
Recall@1, Recall@3, MRR      : standard IR metrics
top1_iscore                  : mean interference score of rank-1 result
correct_iscore               : mean interference score of correct answer
score_gap                    : mean (top1 - top2) score (confidence margin)
scm_openness (learning only) : SCM openness after all updates
warmup_kappa_base            : kappa_base at end of warmup (diagnostic)

Output: results/scm_ab_eval_<timestamp>.jsonl + stdout summary.

Usage
-----
    python scripts/scm_ab_eval.py
    python scripts/scm_ab_eval.py --n 20 --seed 7
    python scripts/scm_ab_eval.py --no-warmup     # skip warmup (debug only)
    python scripts/scm_ab_eval.py --no-experiential
    python scripts/scm_ab_eval.py --datasets-dir /path/to/datasets
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
_DEFAULT_DATASETS = REPO_ROOT.parent.parent / "datasets"

N_PASSAGES = 50
ENCODER = "hippocampus"
SEED = 42
TOP_K = 15  # Pearson candidates before re-ranking

_STOP = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "on", "at", "by", "for", "with", "about", "as", "into", "through",
        "from", "that", "this", "which", "what", "how", "why", "when",
        "where", "who", "q",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q_part(text: str) -> str:
    """Extract the question before ' A:' in Q:...A:... passages."""
    idx = text.find(" A:")
    return text[:idx].strip() if idx != -1 else text.strip()


def _paraphrase(text: str, rng: np.random.Generator) -> str:
    """Content-word shuffle: remove stop words, randomise word order.

    The hippocampus encoder uses character n-grams, so word-order and
    word-boundary changes substantially reduce Pearson similarity while
    preserving content-word identity.
    """
    words = text.split()
    content = [w for w in words if w.lower().strip("?.,!:") not in _STOP]
    if len(content) < 3:
        content = words  # fallback: don't destroy very short queries
    rng.shuffle(content)
    return " ".join(content)


def _find_rank(hits: list[dict], correct_key: str) -> int | None:
    """1-based rank of correct_key in hits, or None if absent."""
    for i, hit in enumerate(hits, 1):
        if hit.get("hex_key") == correct_key:
            return i
    return None


def _find_score(scored: list[tuple], correct_key: str) -> float | None:
    """Score of the correct_key entry, or None if absent."""
    for score, hit, *_ in scored:
        if hit.get("hex_key") == correct_key:
            return float(score)
    return None


def _score_gap(scored: list[tuple]) -> float:
    """Top-1 minus top-2 score; 0.0 if fewer than 2 candidates."""
    if len(scored) < 2:
        return 0.0
    return float(scored[0][0] - scored[1][0])


class _ArmStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.found = 0
        self.r1 = 0
        self.r3 = 0
        self.rr_sum = 0.0
        self.top1_score_sum = 0.0
        self.correct_score_sum = 0.0
        self.correct_score_n = 0
        self.gap_sum = 0.0
        self.n = 0

    def update(
        self,
        rank: int | None,
        top1_score: float,
        correct_score: float | None,
        gap: float,
    ) -> None:
        self.n += 1
        self.top1_score_sum += top1_score
        self.gap_sum += gap
        if rank is not None:
            self.found += 1
            self.rr_sum += 1.0 / rank
            if rank == 1:
                self.r1 += 1
            if rank <= 3:
                self.r3 += 1
        if correct_score is not None:
            self.correct_score_sum += correct_score
            self.correct_score_n += 1

    def summary(self) -> dict:
        n = self.n or 1
        return {
            "arm": self.name,
            "recall@1": round(self.r1 / n, 4),
            "recall@3": round(self.r3 / n, 4),
            "mrr": round(self.rr_sum / n, 4),
            "found": self.found,
            "n": self.n,
            "mean_top1_score": round(self.top1_score_sum / n, 4),
            "mean_correct_score": (
                round(self.correct_score_sum / self.correct_score_n, 4)
                if self.correct_score_n
                else None
            ),
            "mean_score_gap": round(self.gap_sum / n, 4),
        }


# ---------------------------------------------------------------------------
# Experiential storage helper
# ---------------------------------------------------------------------------


def _store_experiential(text: str, data_dir: Path, chunk: str, encoder: str) -> None:
    """Write experiential attractor npy without touching the corpus index.

    store_memory(..., grid='experiential') overwrites the corpus index entry with
    grid='experiential', causing recall_memory to skip it (storage.py:305-306).
    Writing the npy directly sidesteps the index entirely.
    """
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.dynamics import evolve_with_params
    from wheeler_memory.experiential import experiential_dir
    from wheeler_memory.hashing import text_to_hex
    from wheeler_memory.rotation import _get_frame_fn
    from wheeler_memory.storage import get_chunk_dir

    frame_fn = _get_frame_fn(False, encoder=encoder)
    frame = frame_fn(text)
    result = evolve_with_params(frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW)
    if result["state"] != "CONVERGED":
        return
    chunk_dir = get_chunk_dir(data_dir, chunk)
    exp_dir = experiential_dir(chunk_dir)
    hex_key = text_to_hex(text)
    np.save(exp_dir / f"{hex_key}.npy", result["attractor"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SCM closed-loop A/B evaluation")
    parser.add_argument("--n", type=int, default=N_PASSAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--datasets-dir", default=str(_DEFAULT_DATASETS))
    parser.add_argument(
        "--no-experiential",
        action="store_true",
        help="Skip experiential storage (scalar-gate mode)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip exact-query warmup pass (kappa_base stays at 0; debug only)",
    )
    args = parser.parse_args()

    arc_path = Path(args.datasets_dir) / "arc.jsonl"
    if not arc_path.exists():
        print(f"arc.jsonl not found at {arc_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Load passages ----
    all_passages = [json.loads(l)["text"] for l in arc_path.read_text().splitlines() if l]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_passages), size=min(args.n, len(all_passages)), replace=False)
    passages = [all_passages[i] for i in sorted(idx)]
    exact_queries = [_q_part(p) for p in passages]
    paraphrase_queries = [_paraphrase(q, rng) for q in exact_queries]
    print(
        f"Loaded {len(passages)} passages (seed={args.seed}, "
        f"experiential={'off' if args.no_experiential else 'on'}, "
        f"warmup={'off' if args.no_warmup else 'on'})",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="scm_ab_") as _tmpdir:
        data_dir = Path(_tmpdir)

        # ---- Store all passages ----
        from wheeler_memory.chunking import select_chunk
        from wheeler_memory.hashing import text_to_hex
        from wheeler_memory.rotation import _get_frame_fn, store_with_rotation_retry

        print("Storing passages...", end=" ", flush=True, file=sys.stderr)
        stored_keys = []
        stored_chunks = []
        for passage in passages:
            store_with_rotation_retry(passage, data_dir=data_dir, encoder=args.encoder)
            stored_keys.append(text_to_hex(passage))
            chunk = select_chunk(passage)
            stored_chunks.append(chunk)
            if not args.no_experiential:
                _store_experiential(passage, data_dir, chunk, args.encoder)
        print("done", file=sys.stderr)

        # ---- Setup ----
        from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
        from wheeler_memory.dynamics import evolve_and_interpret, evolve_with_params
        from wheeler_memory.interference import interference_score
        from wheeler_memory.recall_api import recognize_top_k
        from wheeler_memory.scm_grid import SCMGrid

        frame_fn = _get_frame_fn(False, encoder=args.encoder)
        learning_scm = SCMGrid.load_or_create(data_dir)
        frozen_grid = np.zeros((64, 64), dtype=np.float32)

        # ---- Phase 1: Warmup — exact queries to settle kappa_base ----
        warmup_kappa_base = 0.0
        if not args.no_warmup:
            print("Warmup (exact queries)...", end=" ", flush=True, file=sys.stderr)
            for i, (passage, query) in enumerate(zip(passages, exact_queries)):
                warmup_seeds = recognize_top_k(
                    query,
                    k=args.top_k,
                    data_dir=data_dir,
                    encoder=args.encoder,
                    threshold=0.0,
                )
                if not warmup_seeds:
                    continue
                q_frame = frame_fn(query)
                q_corpus = evolve_and_interpret(q_frame)["attractor"]
                q_exp = evolve_with_params(
                    q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
                )["attractor"]
                # Score the top warmup hit to get kappa
                top_seed = warmup_seeds[0]
                hk, hc = top_seed.hex_key, top_seed.chunk
                corpus_path = data_dir / "chunks" / hc / "attractors" / f"{hk}.npy"
                exp_path = data_dir / "chunks" / hc / "experiential" / f"{hk}.npy"
                sc = np.load(corpus_path) if corpus_path.exists() else None
                se = np.load(exp_path) if exp_path.exists() else None
                if sc is not None:
                    se_safe = se if se is not None else np.zeros_like(sc)
                    kappa, _, _ = interference_score(q_corpus, q_exp, sc, se_safe, frozen_grid)
                    learning_scm.update_from_recall(sc, se_safe, float(kappa))
            warmup_kappa_base = round(learning_scm.kappa_base, 4)
            print(f"done (kappa_base={warmup_kappa_base})", file=sys.stderr)

        # ---- Phase 2: Paraphrase eval — all three arms scored ----
        arm_pearson = _ArmStats("pearson")
        arm_frozen = _ArmStats("frozen_scm")
        arm_learning = _ArmStats("learning_scm")
        rows: list[dict] = []
        scm_openness_log: list[float] = []

        for i, (passage, query, correct_key) in enumerate(
            zip(passages, paraphrase_queries, stored_keys)
        ):
            print(f"\rEval {i + 1}/{len(passages)}", end="  ", flush=True, file=sys.stderr)

            # Shared Pearson candidates (recognize-only path: no CA engage on query)
            pearson_seeds = recognize_top_k(
                query,
                k=args.top_k,
                data_dir=data_dir,
                encoder=args.encoder,
                threshold=0.0,
            )
            if not pearson_seeds:
                continue
            pearson_hits = [
                {"hex_key": s.hex_key, "chunk": s.chunk, "similarity": s.similarity}
                for s in pearson_seeds
            ]

            # Query attractors for interference arms
            q_frame = frame_fn(query)
            q_corpus = evolve_and_interpret(q_frame)["attractor"]
            q_exp = evolve_with_params(
                q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
            )["attractor"]

            # Preload stored attractors for all candidates
            hit_data: list[tuple] = []
            for hit in pearson_hits:
                hk, hc = hit["hex_key"], hit["chunk"]
                corpus_path = data_dir / "chunks" / hc / "attractors" / f"{hk}.npy"
                exp_path = data_dir / "chunks" / hc / "experiential" / f"{hk}.npy"
                sc = np.load(corpus_path) if corpus_path.exists() else None
                se = np.load(exp_path) if exp_path.exists() else None
                hit_data.append((sc, se))

            # Arm A: Pearson
            pearson_scored = [
                (hit.get("similarity", 0.0), hit, sc, se)
                for hit, (sc, se) in zip(pearson_hits, hit_data)
            ]
            pearson_rank = _find_rank(pearson_hits, correct_key)
            p_correct_score = _find_score(pearson_scored, correct_key)
            p_gap = _score_gap(pearson_scored)

            # Arm B: Frozen SCM
            frozen_scored = []
            for hit, (sc, se) in zip(pearson_hits, hit_data):
                if sc is None:
                    frozen_scored.append((hit.get("similarity", 0.0), hit, sc, se))
                else:
                    score, _, _ = interference_score(q_corpus, q_exp, sc, se, frozen_grid)
                    frozen_scored.append((score, hit, sc, se))
            frozen_scored.sort(key=lambda x: -x[0])
            frozen_rank = _find_rank([h for _, h, _, _ in frozen_scored], correct_key)
            f_correct_score = _find_score(frozen_scored, correct_key)
            f_gap = _score_gap(frozen_scored)

            # Arm C: Learning SCM
            learning_scored = []
            for hit, (sc, se) in zip(pearson_hits, hit_data):
                if sc is None:
                    learning_scored.append((hit.get("similarity", 0.0), hit, sc, se))
                else:
                    score, _, _ = interference_score(
                        q_corpus, q_exp, sc, se, learning_scm.grid
                    )
                    learning_scored.append((score, hit, sc, se))
            learning_scored.sort(key=lambda x: -x[0])
            learning_rank = _find_rank(
                [h for _, h, _, _ in learning_scored], correct_key
            )
            l_correct_score = _find_score(learning_scored, correct_key)
            l_gap = _score_gap(learning_scored)

            # SCM update from top learning result
            top_score, _, top_corpus, top_exp = learning_scored[0]
            if top_corpus is not None:
                top_exp_safe = (
                    top_exp if top_exp is not None else np.zeros_like(top_corpus)
                )
                learning_scm.update_from_recall(top_corpus, top_exp_safe, float(top_score))

            scm_openness_log.append(round(learning_scm.openness(), 4))

            row = {
                "query_idx": i,
                "paraphrase_query": query[:100],
                "exact_query": exact_queries[i][:100],
                "correct_key": correct_key,
                "pearson_rank": pearson_rank,
                "frozen_rank": frozen_rank,
                "learning_rank": learning_rank,
                "pearson_top1_score": round(float(pearson_scored[0][0]), 4),
                "frozen_top1_score": round(float(frozen_scored[0][0]), 4),
                "learning_top1_score": round(float(learning_scored[0][0]), 4),
                "frozen_score_gap": round(f_gap, 4),
                "learning_score_gap": round(l_gap, 4),
                "scm_openness": scm_openness_log[-1],
                "scm_step": learning_scm._step_count,
                "kappa_base": round(learning_scm.kappa_base, 4),
            }
            rows.append(row)

            arm_pearson.update(pearson_rank, pearson_scored[0][0], p_correct_score, p_gap)
            arm_frozen.update(frozen_rank, frozen_scored[0][0], f_correct_score, f_gap)
            arm_learning.update(learning_rank, learning_scored[0][0], l_correct_score, l_gap)

        print(file=sys.stderr)

        # ---- Write JSONL ----
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"scm_ab_eval_{ts}.jsonl"
        meta = {
            "type": "run_meta",
            "n_passages": len(rows),
            "seed": args.seed,
            "encoder": args.encoder,
            "top_k": args.top_k,
            "experiential": not args.no_experiential,
            "warmup": not args.no_warmup,
            "warmup_kappa_base": warmup_kappa_base,
            "timestamp": ts,
            "final_scm_openness": round(learning_scm.openness(), 4),
            "scm_total_steps": learning_scm._step_count,
        }
        with open(out_path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")

        # ---- Summary ----
        summaries = [arm_pearson.summary(), arm_frozen.summary(), arm_learning.summary()]

        print(f"\n{'Arm':<14} {'R@1':>6} {'R@3':>6} {'MRR':>7} {'top1_s':>8} {'corr_s':>8} {'gap':>7}")
        print("-" * 66)
        for s in summaries:
            corr_s = (
                f"{s['mean_correct_score']:.4f}"
                if s["mean_correct_score"] is not None
                else "  N/A "
            )
            print(
                f"{s['arm']:<14} {s['recall@1']:>6.3f} {s['recall@3']:>6.3f} "
                f"{s['mrr']:>7.3f} {s['mean_top1_score']:>8.4f} {corr_s:>8} {s['mean_score_gap']:>7.4f}"
            )

        print(
            f"\nWarmup kappa_base: {warmup_kappa_base:.4f} "
            f"({'skipped' if args.no_warmup else 'settled after 50 exact queries'})"
        )
        print(
            f"SCM openness: start=1.0000 → end={meta['final_scm_openness']:.4f} "
            f"({meta['scm_total_steps']} steps)"
        )
        print(f"Experiential attractors: {'yes' if not args.no_experiential else 'no'}")
        print(f"\nResults → {out_path}")

        n = len(rows)
        if n > 0:
            not_found = sum(1 for r in rows if r["pearson_rank"] is None)
            if not_found:
                print(
                    f"Note: {not_found}/{n} correct answers outside top-{args.top_k} "
                    f"Pearson candidates (counts as not-found for all arms)."
                )
            # Show whether openness moved (confirms seeding fired)
            if scm_openness_log:
                min_open = min(scm_openness_log)
                if min_open < 1.0:
                    print(f"SCM seeded: openness dropped to {min_open:.4f} (gradient active)")
                else:
                    print(
                        "SCM openness stayed at 1.0000 — seeding did not fire. "
                        "Paraphrase kappas may still exceed kappa_base; try --no-warmup "
                        "to inspect or increase paraphrase aggressiveness."
                    )


if __name__ == "__main__":
    main()

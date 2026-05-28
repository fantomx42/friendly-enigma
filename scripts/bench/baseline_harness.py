"""Baseline harness: MiniLM-NN vs. three-grid Wheeler on context-conditioned recall.

The mechanism (three-grid CA + SCM feedback + two-tier recall) is heavily validated
internally, but the value delta over **MiniLM nearest-neighbor alone** has never been
measured end-to-end. SimLex (+0.255) and MMLU (chance, corpus-bound per CANON §8.2)
are the only outward signals so far. This harness produces the missing number: on a
**context-conditioned** task (where reconstruction-not-retrieval is the architecture's
claim), is full three-grid Wheeler measurably better than raw MiniLM cosine NN?

Two phases run inside one execution.

Phase 1 — Topology signal isolation (within-Wheeler diagnostic).
    Hold the Pearson candidate set frozen; re-score under three SCM topologies
    (zero / random / warmed). Report the fraction of queries where SCM topology
    alone shifts top-1. Low fraction ⇒ SCM layer carries little actionable signal.

Phase 2 — Head-to-head on context-conditioned probes.
    Three arms, scored on the same probe set, same temp data dir, same MiniLM encoder:
      Arm 1 (MiniLM-cos):       cosine NN on raw MiniLM embeddings of (context + query).
                                No CA, no Wheeler. The fair floor.
      Arm 2 (Wheeler-recognize): recognize_top_k(context + query, encoder="embedding").
                                Pearson on the CA frame; no experiential, no SCM rerank.
                                Isolates the "MiniLM→CA-frame→Pearson" contribution.
      Arm 3 (Wheeler-full):     prime experiential with context, then
                                recall_with_interference(query, encoder="embedding").
                                Three-grid contender; SCM warmed in Phase 1.

Decision metric printed at end:
    Δ_recall@1 = acc1(Wheeler-full) − max(acc1(MiniLM-cos), acc1(Wheeler-recognize))
    Δ_MRR      = MRR(Wheeler-full)  − max(MRR(MiniLM-cos),  MRR(Wheeler-recognize))

    Δ_recall@1 ≥ +0.10  →  DYNAMICS EARN KEEP
    Δ_recall@1 in (−0.05, 0.10) →  INCONCLUSIVE
    Δ_recall@1 ≤ −0.05  →  DYNAMICS ARE OVERHEAD (flag for KISS audit)

Output: results/baseline_harness_<UTC>.jsonl plus a row appended to
results/baseline_harness.tsv. JSONL has one run_meta line followed by one line per
(probe, context) pair with per-arm ranks, scores, and top-1 keys.

Usage
-----
    wheeler-baseline                              # default n=50, seed=42, warmup on
    wheeler-baseline --n 80 --seed 7
    wheeler-baseline --no-warmup                  # cold SCM (Phase 1 still runs)
    wheeler-baseline --datasets-dir /path/to/datasets
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
_DEFAULT_DATASETS = REPO_ROOT.parent / "datasets"

N_PASSAGES = 50
ENCODER = "embedding"
SEED = 42
TOP_K = 15
COS_BAND_LO = 0.25  # min MiniLM cosine between paired passages
COS_BAND_HI = 0.55  # max — paired passages must overlap, not be near-duplicates
MIN_PROBES = 30  # below this, the dataset can't support the experiment

_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "as",
        "into",
        "through",
        "from",
        "that",
        "this",
        "which",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "q",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "it",
        "its",
        "their",
        "they",
        "we",
        "you",
        "your",
        "our",
        "i",
    }
)


# ---------------------------------------------------------------------------
# Helpers (inlined verbatim from scripts/bench/scm_ab_eval.py — see plan rationale)
# ---------------------------------------------------------------------------


def _q_part(text: str) -> str:
    idx = text.find(" A:")
    return text[:idx].strip() if idx != -1 else text.strip()


def _paraphrase(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    content = [w for w in words if w.lower().strip("?.,!:") not in _STOP]
    if len(content) < 3:
        content = words
    rng.shuffle(content)
    return " ".join(content)


def _content_tokens(text: str) -> list[str]:
    return [
        w.lower().strip("?.,!:;\"'()")
        for w in text.split()
        if w.lower().strip("?.,!:;\"'()")
        and w.lower().strip("?.,!:;\"'()") not in _STOP
    ]


def _find_rank(top1_keys: list[str], correct_key: str) -> int | None:
    for i, k in enumerate(top1_keys, 1):
        if k == correct_key:
            return i
    return None


class _ArmStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.found = 0
        self.r1 = 0
        self.r3 = 0
        self.rr_sum = 0.0
        self.n = 0

    def update(self, rank: int | None) -> None:
        self.n += 1
        if rank is not None:
            self.found += 1
            self.rr_sum += 1.0 / rank
            if rank == 1:
                self.r1 += 1
            if rank <= 3:
                self.r3 += 1

    def summary(self) -> dict:
        n = self.n or 1
        return {
            "arm": self.name,
            "recall@1": round(self.r1 / n, 4),
            "recall@3": round(self.r3 / n, 4),
            "mrr": round(self.rr_sum / n, 4),
            "found": self.found,
            "n": self.n,
        }


# ---------------------------------------------------------------------------
# Experiential write (inlined from scm_ab_eval, with `text` parameter explicit)
# ---------------------------------------------------------------------------


def _store_experiential_for_text(
    text: str, data_dir: Path, chunk: str, encoder: str
) -> str | None:
    """Evolve `text` under experiential CA params and write npy at hex_key(text).

    Returns the hex_key written, or None if evolution didn't converge.
    Mirrors scm_ab_eval._store_experiential but returns the key so callers can
    delete the file between probes (prevents residue from one context bleeding
    into the next probe).
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
        return None
    chunk_dir = get_chunk_dir(data_dir, chunk)
    exp_dir = experiential_dir(chunk_dir)
    hex_key = text_to_hex(text)
    np.save(exp_dir / f"{hex_key}.npy", result["attractor"])
    return hex_key


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def _shared_cue(text_a: str, text_b: str) -> str:
    """Longest common run of content tokens between two passages."""
    a = _content_tokens(text_a)
    b = _content_tokens(text_b)
    if not a or not b:
        return ""
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    match = sm.find_longest_match(0, len(a), 0, len(b))
    cue = a[match.a : match.a + match.size]
    if len(cue) < 2:
        # fall back to set intersection in document order
        common = [t for t in a if t in set(b)]
        cue = common[:4]
    return " ".join(cue)


def _build_probes(
    passages: list[str],
    keys: list[str],
    cos_matrix: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """Build context-conditioned probes from passage pairs in the cosine band.

    For each accepted pair (i, j): two probes (A and B) where the shared cue
    is the query and one passage's question stem is the context. The probe is
    correct on the passage whose context is primed.
    """
    n = len(passages)
    probes: list[dict] = []
    seen_cues: set[str] = set()
    # Iterate in shuffled pair order so the selection isn't biased toward early
    # passages in the ARC file.
    pair_order = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pair_order)

    for i, j in pair_order:
        c = float(cos_matrix[i, j])
        if not (COS_BAND_LO <= c <= COS_BAND_HI):
            continue
        q_a, q_b = _q_part(passages[i]), _q_part(passages[j])
        cue = _shared_cue(q_a, q_b)
        if not cue or cue in seen_cues:
            continue
        # The query is the shuffled cue — it should be ambiguous between A and B.
        query = _paraphrase(cue, rng)
        if not query.strip():
            continue
        seen_cues.add(cue)
        probes.append(
            {
                "pair": (i, j),
                "context": q_a,
                "query": query,
                "correct_key": keys[i],
                "context_label": "A",
                "shared_cue": cue,
            }
        )
        probes.append(
            {
                "pair": (i, j),
                "context": q_b,
                "query": query,
                "correct_key": keys[j],
                "context_label": "B",
                "shared_cue": cue,
            }
        )
    return probes


# ---------------------------------------------------------------------------
# Phase 1 — topology signal isolation
# ---------------------------------------------------------------------------


def _phase1_topology_isolation(
    paraphrase_queries: list[str],
    stored_keys: list[str],
    data_dir: Path,
    encoder: str,
    learning_scm_grid: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Hold Pearson candidates fixed; re-rank under {zero, random, warmed} SCM.

    Top-1 flip rate is the fraction of queries where the warmed-SCM top-1
    differs from the zero-SCM top-1. Low ⇒ SCM topology carries little signal.
    """
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.dynamics import evolve_and_interpret, evolve_with_params
    from wheeler_memory.interference import interference_score
    from wheeler_memory.recall_api import recognize_top_k
    from wheeler_memory.rotation import _get_frame_fn

    frame_fn = _get_frame_fn(False, encoder=encoder)
    zero_grid = np.zeros((64, 64), dtype=np.float32)
    random_grid = np.clip(
        rng.normal(0.0, 0.3, size=(64, 64)).astype(np.float32), -1.0, 1.0
    )

    n_total = 0
    n_flip_zero_vs_warmed = 0
    n_flip_zero_vs_random = 0
    correct_under_zero = 0
    correct_under_warmed = 0

    for q, correct_key in zip(paraphrase_queries, stored_keys):
        seeds = recognize_top_k(
            q, k=TOP_K, data_dir=data_dir, encoder=encoder, threshold=0.0
        )
        if not seeds:
            continue
        q_frame = frame_fn(q)
        q_corpus = evolve_and_interpret(q_frame)["attractor"]
        q_exp = evolve_with_params(
            q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
        )["attractor"]

        # Preload candidate attractors once.
        candidates = []
        for s in seeds:
            corpus_p = data_dir / "chunks" / s.chunk / "attractors" / f"{s.hex_key}.npy"
            exp_p = data_dir / "chunks" / s.chunk / "experiential" / f"{s.hex_key}.npy"
            if not corpus_p.exists():
                continue
            sc = np.load(corpus_p)
            se = np.load(exp_p) if exp_p.exists() else np.zeros_like(sc)
            candidates.append((s.hex_key, sc, se))
        if not candidates:
            continue

        def _rank(grid: np.ndarray) -> list[str]:
            scored = []
            for hk, sc, se in candidates:
                score, _, _ = interference_score(q_corpus, q_exp, sc, se, grid)
                scored.append((score, hk))
            scored.sort(key=lambda t: -t[0])
            return [hk for _, hk in scored]

        order_zero = _rank(zero_grid)
        order_random = _rank(random_grid)
        order_warmed = _rank(learning_scm_grid)

        n_total += 1
        if order_zero[0] != order_warmed[0]:
            n_flip_zero_vs_warmed += 1
        if order_zero[0] != order_random[0]:
            n_flip_zero_vs_random += 1
        if order_zero[0] == correct_key:
            correct_under_zero += 1
        if order_warmed[0] == correct_key:
            correct_under_warmed += 1

    def _safe_div(x: int, y: int) -> float:
        return round(x / y, 4) if y else 0.0

    return {
        "phase": "topology_isolation",
        "n": n_total,
        "top1_flip_rate_warmed_vs_zero": _safe_div(n_flip_zero_vs_warmed, n_total),
        "top1_flip_rate_random_vs_zero": _safe_div(n_flip_zero_vs_random, n_total),
        "correct_at_top1_zero_scm": _safe_div(correct_under_zero, n_total),
        "correct_at_top1_warmed_scm": _safe_div(correct_under_warmed, n_total),
    }


# ---------------------------------------------------------------------------
# Phase 2 — head-to-head arms
# ---------------------------------------------------------------------------


def _arm_minilm_cosine(
    text: str,
    embed_matrix: np.ndarray,
    stored_keys: list[str],
    embed_fn,
    k: int,
) -> list[tuple[float, str]]:
    """Cosine NN over raw MiniLM embeddings — no CA, no Wheeler."""
    vec = embed_fn([text])[0]
    vec = vec / max(np.linalg.norm(vec), 1e-12)
    # embed_matrix rows are already L2-normalized by the caller.
    sims = embed_matrix @ vec
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), stored_keys[i]) for i in idx]


def _arm_wheeler_recognize(
    text: str,
    data_dir: Path,
    encoder: str,
    k: int,
) -> list[tuple[float, str]]:
    """recognize_top_k — Pearson on CA frame; no experiential, no SCM rerank."""
    from wheeler_memory.recall_api import recognize_top_k

    seeds = recognize_top_k(
        text, k=k, data_dir=data_dir, encoder=encoder, threshold=0.0
    )
    return [(float(s.similarity), s.hex_key) for s in seeds]


def _arm_wheeler_full(
    query: str,
    context: str,
    data_dir: Path,
    encoder: str,
    chunk_for_context: str,
    k: int,
) -> tuple[list[tuple[float, str]], str | None]:
    """Three-grid: prime experiential with context, then recall_with_interference.

    Returns (ranked list, context_hex_key_to_cleanup). Caller deletes the
    experiential npy at that key after scoring so the next probe runs clean.
    """
    from wheeler_memory.interference import recall_with_interference

    ctx_key = _store_experiential_for_text(
        context, data_dir, chunk_for_context, encoder
    )
    hits, _, _ = recall_with_interference(
        query, top_k=k, data_dir=data_dir, encoder=encoder, apply_learning=True
    )
    ranked = [
        (float(h.get("interference_score", h.get("similarity", 0.0))), h.get("hex_key"))
        for h in hits
    ]
    return ranked, ctx_key


def _cleanup_experiential(data_dir: Path, chunk: str, hex_key: str | None) -> None:
    if not hex_key:
        return
    p = data_dir / "chunks" / chunk / "experiential" / f"{hex_key}.npy"
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline harness: MiniLM-NN vs. three-grid Wheeler"
    )
    parser.add_argument("--n", type=int, default=N_PASSAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--datasets-dir", default=str(_DEFAULT_DATASETS))
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip Phase-1.5 SCM warmup pass (kappa_base stays at 0; debug only)",
    )
    parser.add_argument(
        "--json-only", action="store_true", help="Suppress the stdout summary table"
    )
    args = parser.parse_args()

    arc_path = Path(args.datasets_dir) / "arc.jsonl"
    if not arc_path.exists():
        print(
            f"arc.jsonl not found at {arc_path}\n"
            f"  Pass --datasets-dir to point at the directory containing arc.jsonl.",
            file=sys.stderr,
        )
        return 2

    # ── Embedding availability check (MiniLM extras must be installed) ──────
    from wheeler_memory.embedding import embed_available

    if not embed_available():
        print(
            "sentence-transformers not installed — install with `pip install -e .[embed]`",
            file=sys.stderr,
        )
        return 2

    # ── Load passages ──────────────────────────────────────────────────────
    all_passages = [
        json.loads(line)["text"]
        for line in arc_path.read_text().splitlines()
        if line.strip()
    ]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(
        len(all_passages), size=min(args.n, len(all_passages)), replace=False
    )
    passages = [all_passages[i] for i in sorted(idx)]
    exact_queries = [_q_part(p) for p in passages]
    paraphrase_queries = [_paraphrase(q, rng) for q in exact_queries]
    print(
        f"Loaded {len(passages)} ARC passages (seed={args.seed}, encoder={args.encoder})",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="baseline_") as _tmpdir:
        data_dir = Path(_tmpdir)

        # ── Store passages ─────────────────────────────────────────────────
        from wheeler_memory.chunking import select_chunk
        from wheeler_memory.hashing import text_to_hex
        from wheeler_memory.rotation import store_with_rotation_retry

        print("Storing passages...", end=" ", flush=True, file=sys.stderr)
        stored_keys: list[str] = []
        stored_chunks: list[str] = []
        for passage in passages:
            store_with_rotation_retry(passage, data_dir=data_dir, encoder=args.encoder)
            stored_keys.append(text_to_hex(passage))
            stored_chunks.append(select_chunk(passage))
        print("done", file=sys.stderr)

        # ── Build MiniLM embedding matrix once (Arm 1 floor + probe construction) ──
        from wheeler_memory.embedding import embed_text_batch

        print(
            "Embedding corpus for MiniLM-cos arm...",
            end=" ",
            flush=True,
            file=sys.stderr,
        )
        raw_embeds = embed_text_batch(passages)  # (N, 384)
        norms = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        embed_matrix = (raw_embeds / norms).astype(np.float32)
        cos_matrix = embed_matrix @ embed_matrix.T
        np.fill_diagonal(cos_matrix, -1.0)
        print("done", file=sys.stderr)

        # ── SCM warmup (mirrors scm_ab_eval) ───────────────────────────────
        from wheeler_memory.constants import (
            EXPERIENTIAL_MAX_PUSH,
            EXPERIENTIAL_SLOPE_FLOW,
        )
        from wheeler_memory.dynamics import evolve_and_interpret, evolve_with_params
        from wheeler_memory.interference import interference_score
        from wheeler_memory.recall_api import recognize_top_k
        from wheeler_memory.rotation import _get_frame_fn
        from wheeler_memory.scm_grid import SCMGrid

        frame_fn = _get_frame_fn(False, encoder=args.encoder)
        learning_scm = SCMGrid.load_or_create(data_dir)
        frozen_grid = np.zeros((64, 64), dtype=np.float32)

        warmup_kappa_base = 0.0
        if not args.no_warmup:
            print("SCM warmup...", end=" ", flush=True, file=sys.stderr)
            for q in exact_queries:
                seeds = recognize_top_k(
                    q,
                    k=args.top_k,
                    data_dir=data_dir,
                    encoder=args.encoder,
                    threshold=0.0,
                )
                if not seeds:
                    continue
                q_frame = frame_fn(q)
                q_corpus = evolve_and_interpret(q_frame)["attractor"]
                q_exp = evolve_with_params(
                    q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
                )["attractor"]
                top = seeds[0]
                corpus_p = (
                    data_dir
                    / "chunks"
                    / top.chunk
                    / "attractors"
                    / f"{top.hex_key}.npy"
                )
                if not corpus_p.exists():
                    continue
                sc = np.load(corpus_p)
                exp_p = (
                    data_dir
                    / "chunks"
                    / top.chunk
                    / "experiential"
                    / f"{top.hex_key}.npy"
                )
                se = np.load(exp_p) if exp_p.exists() else np.zeros_like(sc)
                kappa, _, _ = interference_score(q_corpus, q_exp, sc, se, frozen_grid)
                learning_scm.update_from_recall(sc, se, float(kappa))
            warmup_kappa_base = round(learning_scm.kappa_base, 4)
            print(f"done (kappa_base={warmup_kappa_base})", file=sys.stderr)

        # ── Phase 1: topology signal isolation ─────────────────────────────
        print(
            "Phase 1: topology signal isolation...",
            end=" ",
            flush=True,
            file=sys.stderr,
        )
        phase1 = _phase1_topology_isolation(
            paraphrase_queries,
            stored_keys,
            data_dir,
            args.encoder,
            learning_scm.grid.copy(),
            rng,
        )
        print(
            f"done (top1_flip_warmed_vs_zero={phase1['top1_flip_rate_warmed_vs_zero']})",
            file=sys.stderr,
        )

        # ── Build context-conditioned probes ───────────────────────────────
        probes = _build_probes(passages, stored_keys, cos_matrix, rng)
        if len(probes) < MIN_PROBES:
            print(
                f"INSUFFICIENT PROBES — built {len(probes)} (need {MIN_PROBES}). "
                f"The cosine band [{COS_BAND_LO}, {COS_BAND_HI}] over {len(passages)} ARC "
                f"passages yielded too few viable pairs. Either rerun with --n larger, "
                f"relax the band, or switch datasets. Phase 2 not run.",
                file=sys.stderr,
            )
            phase2_summary = {"status": "INSUFFICIENT_PROBES", "n_probes": len(probes)}
        else:
            print(
                f"Phase 2: {len(probes)} context-conditioned probes...",
                file=sys.stderr,
            )

            arm_minilm = _ArmStats("minilm_cos")
            arm_recognize = _ArmStats("wheeler_recognize")
            arm_full = _ArmStats("wheeler_full")
            rows: list[dict] = []

            for i, probe in enumerate(probes):
                if (i + 1) % 5 == 0 or i + 1 == len(probes):
                    print(
                        f"\r  probe {i + 1}/{len(probes)}",
                        end="",
                        flush=True,
                        file=sys.stderr,
                    )
                ctx_q = (probe["context"] + " " + probe["query"]).strip()
                correct = probe["correct_key"]

                ranked_minilm = _arm_minilm_cosine(
                    ctx_q, embed_matrix, stored_keys, embed_text_batch, args.top_k
                )
                ranked_recognize = _arm_wheeler_recognize(
                    ctx_q, data_dir, args.encoder, args.top_k
                )
                chunk_for_ctx = select_chunk(probe["context"])
                ranked_full, ctx_key = _arm_wheeler_full(
                    probe["query"],
                    probe["context"],
                    data_dir,
                    args.encoder,
                    chunk_for_ctx,
                    args.top_k,
                )
                _cleanup_experiential(data_dir, chunk_for_ctx, ctx_key)

                r_minilm = _find_rank([k for _, k in ranked_minilm], correct)
                r_recognize = _find_rank([k for _, k in ranked_recognize], correct)
                r_full = _find_rank([k for _, k in ranked_full], correct)

                arm_minilm.update(r_minilm)
                arm_recognize.update(r_recognize)
                arm_full.update(r_full)

                rows.append(
                    {
                        "probe_idx": i,
                        "pair": list(probe["pair"]),
                        "context_label": probe["context_label"],
                        "shared_cue": probe["shared_cue"],
                        "correct_key": correct,
                        "rank_minilm_cos": r_minilm,
                        "rank_wheeler_recognize": r_recognize,
                        "rank_wheeler_full": r_full,
                        "top1_minilm_cos": ranked_minilm[0][1]
                        if ranked_minilm
                        else None,
                        "top1_wheeler_recognize": ranked_recognize[0][1]
                        if ranked_recognize
                        else None,
                        "top1_wheeler_full": ranked_full[0][1] if ranked_full else None,
                    }
                )

            print(file=sys.stderr)

            summaries = [
                arm_minilm.summary(),
                arm_recognize.summary(),
                arm_full.summary(),
            ]
            floor_r1 = max(summaries[0]["recall@1"], summaries[1]["recall@1"])
            floor_mrr = max(summaries[0]["mrr"], summaries[1]["mrr"])
            delta_r1 = round(summaries[2]["recall@1"] - floor_r1, 4)
            delta_mrr = round(summaries[2]["mrr"] - floor_mrr, 4)
            if delta_r1 >= 0.10:
                decision = "DYNAMICS EARN KEEP"
            elif delta_r1 <= -0.05:
                decision = "DYNAMICS ARE OVERHEAD"
            else:
                decision = "INCONCLUSIVE"

            phase2_summary = {
                "status": "ran",
                "n_probes": len(probes),
                "arms": summaries,
                "delta_recall@1": delta_r1,
                "delta_mrr": delta_mrr,
                "decision": decision,
            }

        # ── Write JSONL ────────────────────────────────────────────────────
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"baseline_harness_{ts}.jsonl"
        meta = {
            "type": "run_meta",
            "n_passages": len(passages),
            "seed": args.seed,
            "encoder": args.encoder,
            "top_k": args.top_k,
            "warmup": not args.no_warmup,
            "warmup_kappa_base": warmup_kappa_base,
            "timestamp": ts,
            "phase1": phase1,
            "phase2": phase2_summary,
        }
        with open(out_path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            if phase2_summary["status"] == "ran":
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        # ── Append TSV summary row for cross-run comparison ────────────────
        tsv_path = results_dir / "baseline_harness.tsv"
        tsv_header = (
            "timestamp\tn_passages\tn_probes\twarmup_kappa_base"
            "\ttop1_flip_warmed_vs_zero"
            "\tr1_minilm_cos\tr1_wheeler_recognize\tr1_wheeler_full"
            "\tmrr_minilm_cos\tmrr_wheeler_recognize\tmrr_wheeler_full"
            "\tdelta_recall@1\tdelta_mrr\tdecision\n"
        )
        if not tsv_path.exists():
            tsv_path.write_text(tsv_header)
        if phase2_summary["status"] == "ran":
            s = phase2_summary["arms"]
            with open(tsv_path, "a") as f:
                f.write(
                    f"{ts}\t{len(passages)}\t{phase2_summary['n_probes']}"
                    f"\t{warmup_kappa_base}"
                    f"\t{phase1['top1_flip_rate_warmed_vs_zero']}"
                    f"\t{s[0]['recall@1']}\t{s[1]['recall@1']}\t{s[2]['recall@1']}"
                    f"\t{s[0]['mrr']}\t{s[1]['mrr']}\t{s[2]['mrr']}"
                    f"\t{phase2_summary['delta_recall@1']}"
                    f"\t{phase2_summary['delta_mrr']}"
                    f"\t{phase2_summary['decision']}\n"
                )

        # ── Stdout summary ─────────────────────────────────────────────────
        if args.json_only:
            print(json.dumps(meta, indent=2))
            return 0

        print()
        print("─── Phase 1: topology signal isolation ──────────────────────")
        print(f"  n queries scored          : {phase1['n']}")
        print(
            f"  top-1 flip (warmed v zero): {phase1['top1_flip_rate_warmed_vs_zero']:.3f}"
        )
        print(
            f"  top-1 flip (random v zero): {phase1['top1_flip_rate_random_vs_zero']:.3f}"
        )
        print(f"  correct@1 under zero SCM  : {phase1['correct_at_top1_zero_scm']:.3f}")
        print(
            f"  correct@1 under warmed SCM: {phase1['correct_at_top1_warmed_scm']:.3f}"
        )
        print(f"  (low flip rate ⇒ SCM topology carries little signal at recall layer)")

        print()
        print("─── Phase 2: context-conditioned head-to-head ───────────────")
        if phase2_summary["status"] != "ran":
            print(
                f"  {phase2_summary['status']} (n_probes={phase2_summary['n_probes']})"
            )
        else:
            print(
                f"  {'Arm':<22} {'R@1':>6} {'R@3':>6} {'MRR':>7} {'found':>6} {'n':>4}"
            )
            print("  " + "─" * 60)
            for s in phase2_summary["arms"]:
                print(
                    f"  {s['arm']:<22} {s['recall@1']:>6.3f} {s['recall@3']:>6.3f} "
                    f"{s['mrr']:>7.3f} {s['found']:>6} {s['n']:>4}"
                )
            print()
            print(
                f"  Δ recall@1 (full − max(floor)) = {phase2_summary['delta_recall@1']:+.4f}"
            )
            print(
                f"  Δ MRR      (full − max(floor)) = {phase2_summary['delta_mrr']:+.4f}"
            )
            print(f"  ── DECISION: {phase2_summary['decision']} ──")

        print()
        print(f"JSONL → {out_path}")
        print(f"TSV   → {tsv_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

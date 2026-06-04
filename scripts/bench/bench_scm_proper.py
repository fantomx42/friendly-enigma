"""Appropriate SCM test: sculpt the gate -> prove it has teeth -> measure both jobs.

The audit (`audit/scm-gate-and-ablation`) showed the SCM was never actually
measured. Two reasons, both fixed here in the *harness*, not the model:

  1. The only path that hardens the gate toward +-1 is ``self_consistency_check``
     (re-encode output text, re-evolve, close gaps where it diverges from the
     basin). ``update_from_recall`` only floor-seeds. Neither existing SCM bench
     runs the self-consistency sculpting loop, so the gate stays at the 0.001
     floor and ``1 - |M| ~= 1`` -> transparent -> cannot change any score.
  2. The waveguide bench saturates at 100% recall (no headroom) and stores
     corpus-only memories (credit == 0). No discriminating regime.

This bench fixes both: it sculpts the gate via the designed (non-LLM)
self-consistency loop, asserts the gate became non-transparent (Phase A), then
runs two pre-registered, bootstrapped A/Bs against a frozen-open gate:

  Phase A  gate-teeth manipulation check     (PREREQUISITE; gates B & C)
  Phase B  decode-path faithfulness          (integration test)
  Phase C  retrieval under crosstalk          (direct test)

A null in B or C is only interpretable if Phase A passed (gate has teeth) AND
the frozen baseline has headroom. We report whatever the data says.

Usage
-----
    python scripts/bench/bench_scm_proper.py --phase all
    python scripts/bench/bench_scm_proper.py --phase a --pairs 150
    python scripts/bench/bench_scm_proper.py --phase c --pairs 200 --json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora import synthetic_minimal_pairs  # noqa: E402

from wheeler_memory.constants import SCM_GAP_THRESHOLD
from wheeler_memory.scm_grid import SCMGrid

# Pre-registered "teeth" thresholds: a gate that gates must be substantially
# closed somewhere. openness = fraction of cells with |M| < gap threshold.
TEETH_OPENNESS_MAX = 0.80          # <= 80% of cells open (>= 20% meaningfully closed)
TEETH_HARDENED_FRAC_MIN = 0.10     # >= 10% of cells with |M| >= gap threshold (0.3)
# A gate that gates *rankings* must be spatially DIFFERENTIATED, not merely large.
# A uniform |M| (std ~ 0) scales every interference cell equally -> cannot change
# any ranking, i.e. as useless as a transparent gate. Require real spatial spread.
TEETH_ABSM_STD_MIN = 0.02
# openness must also not be fully closed (all cells shut == uniform suppression).
TEETH_OPENNESS_MIN = 0.02


# --------------------------------------------------------------------------
# Storage + experiential (mirrors scm_ab_eval._store_experiential)
# --------------------------------------------------------------------------

def _store_experiential(text: str, data_dir: Path, chunk: str, encoder: str) -> None:
    """Write an experiential attractor npy directly (no corpus-index overwrite)."""
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.dynamics import evolve_with_params
    from wheeler_memory.experiential import experiential_dir
    from wheeler_memory.hashing import text_to_hex
    from wheeler_memory.rotation import _get_frame_fn
    from wheeler_memory.storage import get_chunk_dir

    frame_fn = _get_frame_fn(False, encoder=encoder)
    result = evolve_with_params(frame_fn(text), EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW)
    if result["state"] != "CONVERGED":
        return
    chunk_dir = get_chunk_dir(data_dir, chunk)
    exp_dir = experiential_dir(chunk_dir)
    np.save(exp_dir / f"{text_to_hex(text)}.npy", result["attractor"])


def _build_corpus(kind: str, pairs: int, mmlu_n: int, seed: int) -> list[dict]:
    """Build the test corpus.

    minimal : templated near-duplicate arithmetic (max lexical crosstalk, but
              collinear -> sculpting closes every cell uniformly = no
              differentiation). Crosstalk yes, gate-structure no.
    mmlu    : diverse real MMLU facts (content-varied -> per-memory peaks differ
              -> sculpting can build a *differentiated* gate). Less crosstalk.
    mixed   : both -- diversity for gate structure + collinear pairs for the hard
              retrieval cases. The regime where a differentiated gate could help.
    """
    if kind == "minimal":
        return synthetic_minimal_pairs(pairs, seed=seed)
    from corpora import load_mmlu_pool  # noqa: E402
    if kind == "mmlu":
        return load_mmlu_pool(mmlu_n, seed=seed)
    if kind == "mixed":
        facts = load_mmlu_pool(mmlu_n, seed=seed) + synthetic_minimal_pairs(pairs, seed=seed)
        return facts
    raise ValueError(f"unknown corpus kind: {kind}")


def _store_corpus(
    facts: list[dict], data_dir: Path, encoder: str
) -> list[tuple[str, str, dict]]:
    """Store each fact (corpus + experiential). Return (hex_key, chunk, fact)."""
    from wheeler_memory.chunking import select_chunk
    from wheeler_memory.hashing import text_to_hex
    from wheeler_memory.rotation import store_with_rotation_retry

    stored: list[tuple[str, str, dict]] = []
    for fact in facts:
        text = fact["text"]
        store_with_rotation_retry(text, data_dir=data_dir, encoder=encoder)
        chunk = select_chunk(text)
        _store_experiential(text, data_dir, chunk, encoder)
        stored.append((text_to_hex(text), chunk, fact))
    return stored


def _load_atts(
    hex_key: str, chunk: str, data_dir: Path
) -> tuple[np.ndarray | None, np.ndarray | None]:
    corpus_path = data_dir / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
    exp_path = data_dir / "chunks" / chunk / "experiential" / f"{hex_key}.npy"
    sc = np.load(corpus_path) if corpus_path.exists() else None
    se = np.load(exp_path) if exp_path.exists() else None
    return sc, se


# --------------------------------------------------------------------------
# Phase A — sculpt the gate + teeth check
# --------------------------------------------------------------------------

def _sculpt_gate(
    scm: SCMGrid,
    stored: list[tuple[str, str, dict]],
    data_dir: Path,
    encoder: str,
    epochs: int,
) -> dict:
    """Run the designed self-consistency loop over the stored corpus.

    Each memory's own text is re-encoded, re-evolved, and Pearson-checked against
    its stored basin; consistent -> open gaps (-1), divergent -> close (+1). This
    is the path that actually hardens |M| (via SCMGrid.update + hardening), which
    neither existing SCM bench exercises.
    """
    from wheeler_memory.interference import self_consistency_check

    consistent = inconsistent = 0
    for _ in range(epochs):
        for hex_key, chunk, _fact in stored:
            sc, se = _load_atts(hex_key, chunk, data_dir)
            if sc is None:
                continue
            # The "output" we check is the memory's own text (non-LLM decode proxy).
            cr = self_consistency_check(_fact["text"], sc, se, scm, encoder=encoder)
            if cr.consistent:
                consistent += 1
            else:
                inconsistent += 1
    scm.save()
    return {"sculpt_consistent": consistent, "sculpt_inconsistent": inconsistent}


def _gate_stats(scm: SCMGrid) -> dict:
    absM = np.abs(scm.grid)
    hardened_frac = float((absM >= SCM_GAP_THRESHOLD).mean())
    return {
        "openness": round(float(scm.openness()), 4),
        "hardened_frac": round(hardened_frac, 4),
        "absM_p50": round(float(np.percentile(absM, 50)), 4),
        "absM_p90": round(float(np.percentile(absM, 90)), 4),
        "absM_max": round(float(absM.max()), 4),
        "absM_std": round(float(absM.std()), 4),  # spatial differentiation
        "hardening_nonzero": int((scm.hardening > 0).sum()),
        "hardening_max": int(scm.hardening.max()),
    }


def _teeth_verdict(stats: dict) -> tuple[bool, str]:
    if stats["absM_std"] < TEETH_ABSM_STD_MIN:
        # Large but uniform -> scales all cells equally -> cannot change rankings.
        return False, "UNIFORM_GATE_NO_DIFFERENTIATION"
    if stats["openness"] < TEETH_OPENNESS_MIN:
        return False, "FULLY_CLOSED_UNIFORM"
    if stats["openness"] > TEETH_OPENNESS_MAX or stats["hardened_frac"] < TEETH_HARDENED_FRAC_MIN:
        return False, "TRANSPARENT_GATE"
    return True, "TEETH_OK"


# --------------------------------------------------------------------------
# Retrieval re-ranking arms (shared by Phase C)
# --------------------------------------------------------------------------

def _rank_arms(
    query: str,
    correct_key: str,
    data_dir: Path,
    encoder: str,
    sculpted_grid: np.ndarray,
    frozen_grid: np.ndarray,
    control_grid: np.ndarray,
    top_k: int,
) -> dict | None:
    """Pearson candidates re-ranked under frozen vs sculpted vs control grid.
    Returns per-arm rank of the correct memory (0-indexed) or None if not
    retrieved. The ``control`` arm uses a hand-built maximally-differentiated
    gate (half-open/half-closed): it is the *sensitivity ceiling* — the best a
    perfectly-differentiated gate could ever do — so a null sculpted result is
    only meaningful relative to whether the control itself moves the metric."""
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.dynamics import evolve_and_interpret, evolve_with_params
    from wheeler_memory.interference import interference_score
    from wheeler_memory.recall_api import recognize_top_k
    from wheeler_memory.rotation import _get_frame_fn

    seeds = recognize_top_k(query, k=top_k, data_dir=data_dir, encoder=encoder, threshold=0.0)
    if not seeds:
        return None
    cand = [(s.hex_key, s.chunk, float(s.similarity)) for s in seeds]

    frame_fn = _get_frame_fn(False, encoder=encoder)
    q_frame = frame_fn(query)
    q_corpus = evolve_and_interpret(q_frame)["attractor"]
    q_exp = evolve_with_params(q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW)["attractor"]

    loaded = [(hk, _load_atts(hk, hc, data_dir), sim) for hk, hc, sim in cand]

    def _ranked(grid: np.ndarray | None) -> list[str]:
        scored = []
        for hk, (sc, se), sim in loaded:
            if grid is None or sc is None:
                scored.append((sim, hk))
            else:
                se_safe = se if se is not None else np.zeros_like(sc)
                score, _, _ = interference_score(q_corpus, q_exp, sc, se_safe, grid)
                scored.append((score, hk))
        scored.sort(key=lambda x: -x[0])
        return [hk for _, hk in scored]

    def _rank_of(order: list[str]) -> int | None:
        return order.index(correct_key) if correct_key in order else None

    return {
        "pearson": _rank_of([hk for hk, _, _ in cand]),  # native Pearson order
        "frozen": _rank_of(_ranked(frozen_grid)),
        "sculpted": _rank_of(_ranked(sculpted_grid)),
        "control": _rank_of(_ranked(control_grid)),  # sensitivity ceiling
    }


# --------------------------------------------------------------------------
# Phase B — decode-path faithfulness (self-consistency of the top-1 hit)
# --------------------------------------------------------------------------

def _faithfulness_arms(
    query: str,
    correct_key: str,
    correct_text: str,
    data_dir: Path,
    encoder: str,
    sculpted_grid: np.ndarray,
    frozen_grid: np.ndarray,
    top_k: int,
) -> dict | None:
    """For the top-1 retrieved text under frozen vs sculpted, score self-consistency
    against the CORRECT basin (grid-independent scorer -> unbiased across arms).
    Returns per-arm sim-to-correct-basin and HALLUCINATION flag."""
    from wheeler_memory.dynamics import evolve_and_interpret, evolve_with_params
    from wheeler_memory.constants import EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    from wheeler_memory.interference import interference_score, self_consistency_check
    from wheeler_memory.recall_api import recognize_top_k
    from wheeler_memory.rotation import _get_frame_fn

    seeds = recognize_top_k(query, k=top_k, data_dir=data_dir, encoder=encoder, threshold=0.0)
    if not seeds:
        return None
    cand = [(s.hex_key, s.chunk, float(s.similarity)) for s in seeds]
    by_key = {hk: (hk, hc) for hk, hc, _ in cand}

    frame_fn = _get_frame_fn(False, encoder=encoder)
    q_frame = frame_fn(query)
    q_corpus = evolve_and_interpret(q_frame)["attractor"]
    q_exp = evolve_with_params(q_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW)["attractor"]
    loaded = {hk: (_load_atts(hk, hc, data_dir)) for hk, hc, _ in cand}

    # correct basin (target for faithfulness)
    correct_sc, correct_se = loaded.get(correct_key, (None, None))
    if correct_sc is None:
        return None

    def _top1_text(grid: np.ndarray) -> str:
        scored = []
        for hk, hc, sim in cand:
            sc, se = loaded[hk]
            if sc is None:
                scored.append((sim, hk))
            else:
                se_safe = se if se is not None else np.zeros_like(sc)
                score, _, _ = interference_score(q_corpus, q_exp, sc, se_safe, grid)
                scored.append((score, hk))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]  # top-1 hex_key

    def _score(grid: np.ndarray) -> tuple[float, int]:
        # Faithfulness = basin fidelity of the top-1 hit: Pearson of its corpus
        # attractor vs the CORRECT basin (grid-independent -> unbiased across arms).
        top_key = _top1_text(grid)
        top_sc, _top_se = loaded[top_key]
        sim = float(np.corrcoef(top_sc.ravel(), correct_sc.ravel())[0, 1])
        halluc = 1 if top_key != correct_key else 0
        return (sim if sim == sim else 0.0), halluc

    f_sim, f_hall = _score(frozen_grid)
    s_sim, s_hall = _score(sculpted_grid)
    return {"frozen_sim": f_sim, "sculpted_sim": s_sim,
            "frozen_hall": f_hall, "sculpted_hall": s_hall}


# --------------------------------------------------------------------------
# Phase D — gate-utility ceiling: can ANY trust-weighting help recall at all?
# --------------------------------------------------------------------------
# The strongest possible test of the SCM premise, independent of sculpting and
# retrieval ranking. The premise is: a per-cell trust map that downweights cells
# which "let noise through" improves associative recall. We test it directly as a
# weighted correlation of a perturbed reconstruction to its true basin, under:
#   uniform        — no gate (every cell weight 1)
#   premise_sharp  — keep only "reliable" (sign-stable-under-perturbation) cells
#   wrong_sharp    — keep only "unreliable" cells (the opposite direction)
#   soft_rel       — soft weighting w = per-cell reliability
# If the premise cannot beat uniform even with the gate fit on the eval data
# (an optimistic, conservative-for-refutation setup), the gating operation has no
# benefit on this substrate, regardless of how it is learned.

def _corrupt(text: str, frac: float, rng) -> str:
    words = text.split()
    keep = [w for w in words if rng.random() > frac]
    return " ".join(keep) if keep else words[0]


def _recon_samples(facts, encoder, k, corruption, seed):
    """Return (basins, recons): basin and K perturbed reconstructions per fact,
    as flattened attractors. Reconstruction = re-evolve a corrupted cue (the
    non-LLM decode proxy)."""
    import random as _random
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.rotation import _get_frame_fn
    ff = _get_frame_fn(False, encoder=encoder)
    rng = _random.Random(seed)
    basins, recons = [], []
    for f in facts:
        b = evolve_and_interpret(ff(f["text"]))["attractor"].ravel()
        for _ in range(k):
            o = evolve_and_interpret(ff(_corrupt(f["text"], corruption, rng)))["attractor"].ravel()
            basins.append(b)
            recons.append(o)
    return np.array(basins), np.array(recons)


def _weighted_corr(b, o, w):
    bw = b - np.average(b, weights=w)
    ow = o - np.average(o, weights=w)
    den = np.sqrt(np.sum(w * bw * bw) * np.sum(w * ow * ow))
    return float(np.sum(w * bw * ow) / den) if den > 0 else 0.0


def _phase_d(facts, encoder, k, corruption, seed, bootstrap_n) -> dict:
    basins, recons = _recon_samples(facts, encoder, k, corruption, seed)
    # Per-cell reliability = fraction of reconstructions whose sign matches the basin.
    reliability = (np.sign(recons) == np.sign(basins)).mean(axis=0)  # (4096,)
    thr = np.percentile(reliability, 50)
    gates = {
        "uniform": np.ones(reliability.shape),
        "premise_sharp": (reliability >= thr).astype(float) + 1e-6,
        "wrong_sharp": (reliability < thr).astype(float) + 1e-6,
        "soft_rel": reliability + 1e-6,
    }
    sims = {name: np.array([_weighted_corr(b, o, w) for b, o in zip(basins, recons)])
            for name, w in gates.items()}
    out = {f"D_fidelity_{name}": round(float(v.mean()), 4) for name, v in sims.items()}
    out["D_n_samples"] = len(basins)
    # Pre-registered: SCM premise (downweight unreliable) BEATS uniform.
    gap, lo, hi = _bootstrap_gap_ci(sims["premise_sharp"], sims["uniform"], bootstrap_n)
    out["D_premise_minus_uniform"] = round(gap, 4)
    out["D_premise_ci_lo"] = round(lo, 4)
    out["D_premise_ci_hi"] = round(hi, 4)
    out["D_reliability_std"] = round(float(reliability.std()), 4)
    if lo > 0:
        out["D_verdict"] = "PASS_GATE_HELPS"
    elif hi < 0:
        out["D_verdict"] = "REFUTED_GATE_HURTS"
    else:
        out["D_verdict"] = "NULL_GATE_NEUTRAL"
    return out


# --------------------------------------------------------------------------
# Bootstrap (paired) — copied from bench_scm_ablation._bootstrap_gap_ci
# --------------------------------------------------------------------------

def _bootstrap_gap_ci(a, b, n: int, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(phase: str, corpus: str, pairs: int, mmlu_n: int, seed: int,
        sculpt_epochs: int, top_k: int, bootstrap_n: int, encoder: str) -> dict:
    t0 = time.time()
    facts = _build_corpus(corpus, pairs, mmlu_n, seed)
    out: dict = {
        "phase": phase, "corpus": corpus, "pairs": len(facts), "seed": seed,
        "sculpt_epochs": sculpt_epochs, "top_k": top_k, "encoder": encoder,
    }

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        stored = _store_corpus(facts, d, encoder)

        # ---- Phase A: sculpt + teeth (always run; prerequisite) ----
        scm = SCMGrid.load_or_create(d)
        out.update(_sculpt_gate(scm, stored, d, encoder, sculpt_epochs))
        gate = _gate_stats(scm)
        out.update(gate)
        has_teeth, teeth_verdict = _teeth_verdict(gate)
        out["teeth_verdict"] = teeth_verdict
        out["has_teeth"] = has_teeth

        sculpted_grid = scm.grid.copy()
        frozen_grid = np.zeros((64, 64), dtype=np.float32)
        # Sensitivity ceiling: a maximally-differentiated gate (half open, half
        # fully closed). Controls show even this never flips top-1 on these tasks
        # — so a null sculpted result is interpreted against what ANY gate can do.
        control_grid = np.zeros((64, 64), dtype=np.float32)
        control_grid[:, :32] = 1.0
        queries = [(hk, f["question"], f["text"]) for hk, _c, f in stored]

        # ---- Phase D: gate-utility ceiling (independent of sculpting/ranking) ----
        if phase in ("d", "all"):
            out.update(_phase_d(facts, encoder, k=12, corruption=0.4,
                                seed=seed, bootstrap_n=bootstrap_n))

        if phase in ("a", "d"):
            out["elapsed_seconds"] = round(time.time() - t0, 2)
            out["timestamp"] = datetime.now(timezone.utc).isoformat()
            return out

        if not has_teeth:
            out["note"] = (
                "Gate is transparent after sculpting — B/C are moot (a flat gate "
                "cannot change any score). Reported for the record; not interpreted."
            )

        # ---- Phase C: retrieval under crosstalk ----
        if phase in ("c", "all"):
            arms = ("pearson", "frozen", "sculpted", "control")
            r1 = {a: [] for a in arms}
            r3 = {a: [] for a in arms}
            mrr = {a: [] for a in arms}
            for ck, q, _txt in queries:
                r = _rank_arms(q, ck, d, encoder, sculpted_grid, frozen_grid,
                               control_grid, top_k)
                if r is None:
                    continue
                for a in arms:
                    rk = r[a]
                    r1[a].append(1 if rk == 0 else 0)
                    r3[a].append(1 if (rk is not None and rk < 3) else 0)
                    mrr[a].append(1.0 / (rk + 1) if rk is not None else 0.0)
            n = len(r1["sculpted"])
            out["C_n"] = n
            for a in arms:
                out[f"C_{a}_r1"] = round(float(np.mean(r1[a])), 4) if n else None
                out[f"C_{a}_r3"] = round(float(np.mean(r3[a])), 4) if n else None
                out[f"C_{a}_mrr"] = round(float(np.mean(mrr[a])), 4) if n else None
            gap, lo, hi = _bootstrap_gap_ci(r1["sculpted"], r1["frozen"], bootstrap_n)
            out["C_gap_sculpted_minus_frozen"] = round(gap, 4)
            out["C_gap_ci_lo"] = round(lo, 4)
            out["C_gap_ci_hi"] = round(hi, 4)
            # Sensitivity ceiling: does ANY gate move R@1 vs frozen? If the control
            # can't either, the task itself is gate-insensitive at top-1.
            cgap, clo, chi = _bootstrap_gap_ci(r1["control"], r1["frozen"], bootstrap_n)
            out["C_control_gap_vs_frozen"] = round(cgap, 4)
            out["C_control_can_move_r1"] = bool(clo > 0 or chi < 0)
            out["C_headroom_ok"] = (out["C_frozen_r1"] is not None
                                    and out["C_frozen_r1"] < 0.90)
            out["C_verdict"] = _ab_verdict(lo, hi, out["C_headroom_ok"], has_teeth)

        # ---- Phase B: decode-path faithfulness ----
        if phase in ("b", "all"):
            f_sim, s_sim, f_hall, s_hall = [], [], [], []
            for ck, q, txt in queries:
                r = _faithfulness_arms(q, ck, txt, d, encoder,
                                       sculpted_grid, frozen_grid, top_k)
                if r is None:
                    continue
                f_sim.append(r["frozen_sim"]); s_sim.append(r["sculpted_sim"])
                f_hall.append(r["frozen_hall"]); s_hall.append(r["sculpted_hall"])
            # sculpted should LOWER hallucination -> gap = frozen - sculpted > 0
            hg, hlo, hhi = _bootstrap_gap_ci(f_hall, s_hall, bootstrap_n)
            sg, slo, shi = _bootstrap_gap_ci(s_sim, f_sim, bootstrap_n)
            out["B_n"] = len(s_sim)
            out["B_frozen_hall_rate"] = round(float(np.mean(f_hall)), 4) if f_hall else None
            out["B_sculpted_hall_rate"] = round(float(np.mean(s_hall)), 4) if s_hall else None
            out["B_hall_reduction"] = round(hg, 4)
            out["B_hall_ci_lo"] = round(hlo, 4); out["B_hall_ci_hi"] = round(hhi, 4)
            out["B_fidelity_gain"] = round(sg, 4)
            out["B_fidelity_ci_lo"] = round(slo, 4); out["B_fidelity_ci_hi"] = round(shi, 4)
            out["B_headroom_ok"] = (out["B_frozen_hall_rate"] is not None
                                    and out["B_frozen_hall_rate"] > 0.05)
            out["B_verdict"] = _ab_verdict(hlo, hhi, out["B_headroom_ok"], has_teeth)

    out["elapsed_seconds"] = round(time.time() - t0, 2)
    out["timestamp"] = datetime.now(timezone.utc).isoformat()
    return out


def _ab_verdict(lo: float, hi: float, headroom_ok: bool, has_teeth: bool) -> str:
    if not has_teeth:
        return "MOOT_TRANSPARENT_GATE"
    if not headroom_ok:
        return "NO_HEADROOM"
    if lo > 0:
        return "PASS"
    if hi < 0:
        return "FAIL_NEGATIVE"
    return "FAIL_NULL"


_RESULTS = Path(__file__).resolve().parents[2] / "scm_proper.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description="Appropriate SCM test (sculpt/teeth/B/C)")
    p.add_argument("--phase", choices=["a", "b", "c", "d", "all"], default="all")
    p.add_argument("--corpus", choices=["minimal", "mmlu", "mixed"], default="mixed")
    p.add_argument("--pairs", type=int, default=150, help="confusable minimal pairs stored")
    p.add_argument("--mmlu-n", type=int, default=150, help="diverse MMLU facts (mmlu/mixed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sculpt-epochs", type=int, default=4)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--encoder", default="hippocampus")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    result = run(args.phase, args.corpus, args.pairs, args.mmlu_n, args.seed,
                 args.sculpt_epochs, args.top_k, args.bootstrap_n, args.encoder)

    if args.json:
        print(json.dumps(result))
    else:
        _print_human(result)

    if not args.no_save:
        with open(_RESULTS, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\n  Appended to {_RESULTS.name}")


def _print_human(r: dict) -> None:
    print("\n" + "=" * 70)
    print("  WHEELER MEMORY — APPROPRIATE SCM TEST (sculpt -> teeth -> B/C)")
    print("=" * 70)
    print(f"  corpus={r.get('corpus')}  stored={r['pairs']}  sculpt epochs={r['sculpt_epochs']}  "
          f"seed={r['seed']}")
    print(f"\n  Phase A — gate after sculpting:")
    print(f"    openness={r['openness']}  hardened_frac={r['hardened_frac']}  "
          f"|M| p50/p90/max={r['absM_p50']}/{r['absM_p90']}/{r['absM_max']}  "
          f"std={r['absM_std']}")
    print(f"    hardening cells={r['hardening_nonzero']}  max count={r['hardening_max']}")
    print(f"    sculpt: consistent={r['sculpt_consistent']} inconsistent={r['sculpt_inconsistent']}")
    print(f"    TEETH VERDICT: {r['teeth_verdict']}  (has_teeth={r['has_teeth']})")
    if "D_verdict" in r:
        print(f"\n  Phase D — gate-utility ceiling (n={r['D_n_samples']} reconstructions):")
        print(f"    reconstruction fidelity (weighted corr to true basin):")
        print(f"      uniform (no gate)            : {r['D_fidelity_uniform']}")
        print(f"      premise: keep-reliable cells : {r['D_fidelity_premise_sharp']}  "
              f"(SCM premise — should WIN)")
        print(f"      wrong:   keep-unreliable     : {r['D_fidelity_wrong_sharp']}")
        print(f"      soft w=reliability           : {r['D_fidelity_soft_rel']}")
        print(f"    premise - uniform = {r['D_premise_minus_uniform']:+}  "
              f"95% CI [{r['D_premise_ci_lo']:+}, {r['D_premise_ci_hi']:+}]")
        print(f"    VERDICT: {r['D_verdict']}")
    if "C_verdict" in r:
        print(f"\n  Phase C — retrieval under crosstalk (n={r['C_n']}):")
        print(f"    {'arm':<10}{'R@1':>8}{'R@3':>8}{'MRR':>8}")
        for a in ("pearson", "frozen", "sculpted", "control"):
            print(f"    {a:<10}{r[f'C_{a}_r1']:>8}{r[f'C_{a}_r3']:>8}{r[f'C_{a}_mrr']:>8}")
        print(f"    gap (sculpted-frozen R@1)={r['C_gap_sculpted_minus_frozen']:+}  "
              f"95% CI [{r['C_gap_ci_lo']:+}, {r['C_gap_ci_hi']:+}]  "
              f"headroom_ok={r['C_headroom_ok']}")
        print(f"    sensitivity ceiling: control vs frozen R@1 gap="
              f"{r['C_control_gap_vs_frozen']:+}  control_can_move_r1={r['C_control_can_move_r1']}")
        print(f"    VERDICT: {r['C_verdict']}")
    if "B_verdict" in r:
        print(f"\n  Phase B — decode-path faithfulness (n={r['B_n']}):")
        print(f"    hallucination  frozen={r['B_frozen_hall_rate']}  "
              f"sculpted={r['B_sculpted_hall_rate']}  reduction={r['B_hall_reduction']:+} "
              f"CI [{r['B_hall_ci_lo']:+}, {r['B_hall_ci_hi']:+}]")
        print(f"    fidelity gain (sculpted-frozen)={r['B_fidelity_gain']:+}  "
              f"CI [{r['B_fidelity_ci_lo']:+}, {r['B_fidelity_ci_hi']:+}]  "
              f"headroom_ok={r['B_headroom_ok']}")
        print(f"    VERDICT: {r['B_verdict']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

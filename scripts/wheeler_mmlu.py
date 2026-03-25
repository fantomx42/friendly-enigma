"""Wheeler Memory — MMLU Benchmark Harness

Evaluates Wheeler against the Massive Multitask Language Understanding benchmark
and reports accuracy per subject, comparable to frontier model scores.

Evaluation modes
----------------
semantic (default, Wheeler-native)
    For each choice A/B/C/D, query Wheeler with "question + choice" and measure
    the top attractor similarity. Pick the choice with the highest score.
    No LLM involved — the CA system picks the answer.

decode
    Recall memories for the question, then pass Wheeler's attractor state +
    the four choices to the small model decoder. Model outputs a single letter.

learn
    Full learn→consolidate→test cycle:
    1. Embed all correct Q&A pairs from dev+validation splits.
    2. Store each as a new attractor in the science chunk.
    3. Run sleep consolidation to reinforce and prune.
    4. Rebuild the attractor cache and test on the test split.

recall-text
    Recall top-K facts, extract answer by letter regex or text matching from context.

cortex
    Full cortex pipeline: L1 graph reasoning → L2 settlement → SCM evaluation
    → optional L3 neural network classifier. Retrieves memories into attractor
    graph, runs settlement CA for opinion diffusion, computes coherence layers,
    and picks answer with highest settled opinion (or via L3 classifier if weights provided).

ternary
    Pure ternary scoring: snap each choice's attractor to {-1, 0, +1} via
    topological cell roles (local max/min/slope), then pick the choice with
    the highest net positivity (count(+1) − count(−1)).  No retrieval, no
    graph reasoning — the CA's 3-state output IS the answer.

ternary-retrieval
    Ternary + retrieval: for each choice, compute ternary roles, then measure
    cell-wise overlap against pre-snapped cached attractors (weighted by
    Pearson retrieval similarity).  Uses AttractorCache's pre-computed ternary
    matrix — no re-encoding of retrieved texts.

Usage
-----
    python scripts/wheeler_mmlu.py --subjects abstract_algebra --samples 50
    python scripts/wheeler_mmlu.py --mode decode --model qwen2.5:1.5b --samples 20
    python scripts/wheeler_mmlu.py --mode learn --subjects high_school_physics
    python scripts/wheeler_mmlu.py --mode cortex --subjects abstract_algebra --recall-k 10
    python scripts/wheeler_mmlu.py --mode cortex --classifier-weights path/to/weights.npz --all
    python scripts/wheeler_mmlu.py --mode ternary --subjects sociology --samples 50
    python scripts/wheeler_mmlu.py --mode ternary-retrieval --all --recall-k 10
    python scripts/wheeler_mmlu.py --all --samples 10
    python scripts/wheeler_mmlu.py --list-subjects
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from wheeler_memory.storage import recall_memory

# ---------------------------------------------------------------------------
# MMLU subject catalogue (57 subjects)
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Attractor cache — loaded once at benchmark startup for fast Pearson search
# ---------------------------------------------------------------------------


class AttractorCache:
    """Pre-loads all stored attractors into a single numpy matrix.

    Replaces per-recall disk reads with one vectorized matrix operation.
    For 1428 attractors: disk load ~0.5s once, then <1ms per query.
    """

    def __init__(self, data_dir: Path | None = None):
        from wheeler_memory.storage import DEFAULT_DATA_DIR, list_memories
        from wheeler_memory.dynamics import evolve_and_interpret

        d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        print("  Loading attractor cache into memory...")
        t0 = time.time()

        memories = list_memories(data_dir=d)
        self._texts: list[str] = []
        self._meta: list[dict] = []
        attractor_list = []

        for m in memories:
            chunk = m.get("chunk", "general")
            hex_key = m["hex_key"]
            att_path = d / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
            if not att_path.exists():
                continue
            att = np.load(att_path)
            if att.shape != (64, 64):
                continue
            attractor_list.append(att.flatten().astype(np.float32))
            self._texts.append(m["text"])
            self._meta.append(m)

        if not attractor_list:
            self._matrix = np.zeros((0, 4096), dtype=np.float32)
            self._att_means = np.zeros(0, dtype=np.float32)
            self._att_stds = np.zeros(0, dtype=np.float32)
        else:
            self._matrix = np.stack(attractor_list)  # (N, 4096)
            self._att_means = self._matrix.mean(axis=1)  # (N,)
            self._att_stds = self._matrix.std(axis=1)  # (N,)

        self._evolve = evolve_and_interpret
        elapsed = time.time() - t0
        print(f"  Cached {len(self._texts)} attractors in {elapsed:.2f}s")

    def _build_ternary(self):
        """Pre-snap all cached attractors to ternary {-1, 0, +1}.

        Stores an (N, 4096) int8 matrix for vectorized ternary overlap.
        Called lazily on first ternary search to avoid overhead when unused.
        """
        from wheeler_memory.dynamics import snap_to_ternary

        if len(self._texts) == 0:
            self._ternary_matrix = np.zeros((0, 4096), dtype=np.int8)
            return

        rows = []
        for i in range(len(self._matrix)):
            att_2d = self._matrix[i].reshape(64, 64)
            rows.append(snap_to_ternary(att_2d).flatten())
        self._ternary_matrix = np.stack(rows)  # (N, 4096) int8

    def _pearson_search(
        self, query_frame: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core Pearson search — returns (top_indices, similarities).

        Shared by search() and search_ternary() to avoid duplicating retrieval logic.
        """
        from wheeler_memory.dynamics import evolve_and_interpret
        from wheeler_memory.constants import (
            SALIENCE_MAX_ITERS_MED,
            SALIENCE_THRESHOLD_MED,
        )

        result = evolve_and_interpret(
            query_frame,
            max_iters=SALIENCE_MAX_ITERS_MED,
            stability_threshold=SALIENCE_THRESHOLD_MED,
        )
        q_flat = result["attractor"].flatten().astype(np.float32)
        q_mean = q_flat.mean()
        q_std = q_flat.std()

        if q_std < 1e-10:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Vectorized Pearson: (N, 4096) dot (4096,) → (N,)
        q_centered = q_flat - q_mean
        centered = self._matrix - self._att_means[:, None]
        dots = centered @ q_centered
        valid = self._att_stds > 1e-10
        sims = np.where(valid, dots / (4096 * self._att_stds * q_std), 0.0)

        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return top_idx, sims

    def search(self, query_frame: np.ndarray, top_k: int = 5) -> list[dict]:
        """Vectorized Pearson search against entire attractor matrix."""
        if len(self._texts) == 0:
            return []

        top_idx, sims = self._pearson_search(query_frame, top_k)
        if len(top_idx) == 0:
            return []

        return [{"text": self._texts[i], "similarity": float(sims[i])} for i in top_idx]

    def search_ternary(self, query_frame: np.ndarray, top_k: int = 5) -> list[dict]:
        """Pearson search that also returns pre-computed ternary arrays.

        Returns list of {"text", "similarity", "ternary": (4096,) int8}.
        Avoids re-encoding retrieved texts — uses cached attractors directly.
        """
        if len(self._texts) == 0:
            return []

        if not hasattr(self, "_ternary_matrix"):
            self._build_ternary()

        top_idx, sims = self._pearson_search(query_frame, top_k)
        if len(top_idx) == 0:
            return []

        return [
            {
                "text": self._texts[i],
                "similarity": float(sims[i]),
                "ternary": self._ternary_matrix[i],  # (4096,) int8
            }
            for i in top_idx
        ]

    def _build_spatial(self):
        """Pre-compute spatial topology features for all cached attractors."""
        from wheeler_memory.dynamics import extract_spatial_features

        t0 = time.time()
        feature_vecs = []
        for i in range(len(self._matrix)):
            att_2d = self._matrix[i].reshape(64, 64)
            fv = extract_spatial_features(att_2d)
            feature_vecs.append(fv)

        if feature_vecs:
            self._spatial_matrix = np.stack(feature_vecs)  # (N, 51)
            norms = np.linalg.norm(self._spatial_matrix, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            self._spatial_normed = self._spatial_matrix / norms
        else:
            self._spatial_matrix = np.zeros((0, 51), dtype=np.float32)
            self._spatial_normed = np.zeros((0, 51), dtype=np.float32)

        elapsed = time.time() - t0
        print(f"  Computed spatial features ({self._spatial_matrix.shape[1]}-dim) in {elapsed:.1f}s")

    def _spatial_search(
        self, query_frame: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hybrid Pearson + spatial topology search."""
        from wheeler_memory.dynamics import (
            evolve_and_interpret,
            extract_spatial_features,
        )
        from wheeler_memory.constants import (
            SALIENCE_MAX_ITERS_MED,
            SALIENCE_THRESHOLD_MED,
            SPATIAL_PEARSON_WEIGHT,
        )

        if not hasattr(self, "_spatial_matrix"):
            self._build_spatial()

        if len(self._texts) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Pearson component
        result = evolve_and_interpret(
            query_frame,
            max_iters=SALIENCE_MAX_ITERS_MED,
            stability_threshold=SALIENCE_THRESHOLD_MED,
        )
        q_flat = result["attractor"].flatten().astype(np.float32)
        q_mean = q_flat.mean()
        q_std = q_flat.std()

        if q_std < 1e-10:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        q_centered = q_flat - q_mean
        centered = self._matrix - self._att_means[:, None]
        dots = centered @ q_centered
        valid = self._att_stds > 1e-10
        pearson_sims = np.where(valid, dots / (4096 * self._att_stds * q_std), 0.0)

        # Spatial component
        q_spatial = extract_spatial_features(result["attractor"])
        q_norm = np.linalg.norm(q_spatial)
        if q_norm > 1e-10:
            q_normed = q_spatial / q_norm
        else:
            q_normed = q_spatial
        spatial_sims = self._spatial_normed @ q_normed

        # Combine
        w = SPATIAL_PEARSON_WEIGHT
        combined = w * pearson_sims + (1 - w) * spatial_sims

        k = min(top_k, len(combined))
        top_idx = np.argpartition(combined, -k)[-k:]
        top_idx = top_idx[np.argsort(combined[top_idx])[::-1]]
        return top_idx, combined


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_mmlu(subjects: list[str], split: str = "test", samples: int | None = None):
    """Yield (subject, question, choices, answer_idx) tuples."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' library required. Run: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
        except Exception as exc:
            print(f"  [skip] {subject}: {exc}", file=sys.stderr)
            continue

        if samples:
            ds = ds.select(range(min(samples, len(ds))))

        for item in ds:
            choices = [item["choices"][i] for i in range(4)]
            answer_idx = int(item["answer"])
            yield subject, item["question"], choices, answer_idx


# ---------------------------------------------------------------------------
# Encoder dispatch — select frame_fn and batch_fn by encoder name
# ---------------------------------------------------------------------------


def _get_encoder_fns(encoder: str = "hippocampus"):
    """Return (frame_fn, batch_fn) for the given encoder name."""
    if encoder == "word":
        from wheeler_memory.word_encoder import word_to_frame, word_to_frame_batch

        return word_to_frame, word_to_frame_batch
    elif encoder == "word-blended":
        from wheeler_memory.word_encoder import word_to_frame
        from wheeler_memory.language_wheeler import language_to_frame
        from wheeler_memory.constants import BLEND_ALPHA

        def _blended(text, size=64):
            w = word_to_frame(text, size)
            l = language_to_frame(text, size)
            return np.tanh(BLEND_ALPHA * w + (1 - BLEND_ALPHA) * l).astype(np.float32)

        def _blended_batch(texts, size=64):
            return [_blended(t, size) for t in texts]

        return _blended, _blended_batch
    elif encoder == "hippo-word":
        from wheeler_memory.hippocampus import hippocampus_to_frame
        from wheeler_memory.word_encoder import word_to_frame
        from wheeler_memory.constants import WORD_HIPPO_BLEND

        def _hippo_word(text, size=64):
            h = hippocampus_to_frame(text, size)
            w = word_to_frame(text, size)
            return np.tanh((1 - WORD_HIPPO_BLEND) * h + WORD_HIPPO_BLEND * w).astype(
                np.float32
            )

        def _hippo_word_batch(texts, size=64):
            return [_hippo_word(t, size) for t in texts]

        return _hippo_word, _hippo_word_batch
    else:  # "hippocampus" (default)
        from wheeler_memory.hippocampus import (
            hippocampus_to_frame,
            hippocampus_to_frame_batch,
        )

        return hippocampus_to_frame, hippocampus_to_frame_batch


# ---------------------------------------------------------------------------
# Semantic scoring (Wheeler-native, no LLM)
# ---------------------------------------------------------------------------


def score_semantic(
    question: str,
    choices: list[str],
    recall_k: int = 5,
    cache: "AttractorCache | None" = None,
    precomputed_frames: list | None = None,
) -> tuple[int, list[float]]:
    """Score choices by Pearson correlation against the attractor cache.

    Uses vectorized matrix search (AttractorCache) for speed — avoids all
    per-recall disk I/O. Falls back to hash_to_frame if no cache provided.

    Returns (predicted_index, [score_A, score_B, score_C, score_D]).
    """
    from wheeler_memory.hashing import hash_to_frame

    if precomputed_frames is not None:
        frames = precomputed_frames
    else:
        frames = [hash_to_frame(f"{question} {c}") for c in choices]

    scores = []
    for frame in frames:
        if cache is not None:
            hits = cache.search(frame, top_k=recall_k)
        else:
            hits = []
        top_sim = max((h.get("similarity", 0.0) for h in hits), default=0.0)
        scores.append(top_sim)
    return int(scores.index(max(scores))), scores


def score_spatial(
    question: str,
    choices: list[str],
    recall_k: int = 5,
    cache: "AttractorCache | None" = None,
    precomputed_frames: list | None = None,
) -> tuple[int, list[float]]:
    """Score choices by hybrid Pearson + spatial topology similarity.

    Like score_semantic but uses _spatial_search() which blends Pearson
    correlation with spatial cluster topology features (island positions,
    sizes, boundary length).
    """
    if precomputed_frames is not None:
        frames = precomputed_frames
    else:
        from wheeler_memory.hashing import hash_to_frame
        frames = [hash_to_frame(f"{question} {c}") for c in choices]

    scores = []
    for frame in frames:
        if cache is not None:
            top_idx, sims = cache._spatial_search(frame, top_k=recall_k)
            top_sim = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0
        else:
            top_sim = 0.0
        scores.append(top_sim)
    return int(scores.index(max(scores))), scores


def score_reverse_lookup(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    encoder: str = "hippo-word",
) -> tuple[int, list[float]]:
    """Score by encoding each choice alone, recalling related facts,
    then checking which recalled facts best match the question.

    The human test-taking strategy: "What do I know about this answer?"
    then compare recalled knowledge to the question.
    """
    if cache is None:
        return 0, [0.0] * len(choices)

    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.constants import (
        SALIENCE_MAX_ITERS_MED,
        SALIENCE_THRESHOLD_MED,
    )

    frame_fn, _ = _get_encoder_fns(encoder)

    # Encode the question once and evolve to attractor
    q_frame = frame_fn(question)
    q_result = evolve_and_interpret(
        q_frame,
        max_iters=SALIENCE_MAX_ITERS_MED,
        stability_threshold=SALIENCE_THRESHOLD_MED,
    )
    q_flat = q_result["attractor"].flatten().astype(np.float32)
    q_mean = q_flat.mean()
    q_std = q_flat.std()
    if q_std < 1e-10:
        return 0, [0.0] * len(choices)
    q_centered = q_flat - q_mean

    # Also get question-only similarity as baseline to subtract
    q_hits = cache.search(q_frame, top_k=1)
    q_baseline = q_hits[0]["similarity"] if q_hits else 0.0

    scores = []
    for choice in choices:
        # Strategy 1: encode choice alone, recall related facts
        c_frame = frame_fn(choice)
        top_idx, _ = cache._pearson_search(c_frame, top_k=recall_k)

        # Strategy 2: encode question+choice, get direct similarity
        qc_frame = frame_fn(f"{question} {choice}")
        qc_hits = cache.search(qc_frame, top_k=1)
        qc_sim = qc_hits[0]["similarity"] if qc_hits else 0.0
        # Differential: how much does adding this choice boost similarity?
        diff_score = qc_sim - q_baseline

        # Strategy 1 score: best match of recalled facts to question
        reverse_score = 0.0
        if len(top_idx) > 0:
            recalled = cache._matrix[top_idx]
            recalled_stds = cache._att_stds[top_idx]
            recalled_means = cache._att_means[top_idx]
            valid = recalled_stds > 1e-10
            centered = recalled - recalled_means[:, None]
            dots = centered @ q_centered
            sims = np.where(valid, dots / (4096 * recalled_stds * q_std), 0.0)
            reverse_score = float(sims.max()) if len(sims) > 0 else 0.0

        # Combine: differential + reverse-lookup
        scores.append(diff_score + 0.5 * reverse_score)

    return int(scores.index(max(scores))), scores


def score_recall_text(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    debug: bool = False,
    encoder: str = "hippocampus",
) -> tuple[int, float]:
    """Recall top-K facts; extract answer by letter regex OR text matching.

    Two extraction strategies (both tried per hit):
      1. Letter regex: "A: B." → vote for choice index 1
      2. Text match:  answer substring appears in a choice → vote for that choice

    Returns (predicted_index, confidence) or (-1, 0.0) on failure.
    """
    try:
        frame_fn, _ = _get_encoder_fns(encoder)
        frame = frame_fn(question)
    except Exception:
        from wheeler_memory.hashing import hash_to_frame

        frame = hash_to_frame(question)

    hits = cache.search(frame, top_k=recall_k) if cache is not None else []

    if debug:
        print(f"      [debug] top-{len(hits)} recalls:")
        for h in hits[:5]:
            print(f"        sim={h['similarity']:.3f}  {h['text'][:80]}")

    votes = [0.0, 0.0, 0.0, 0.0]  # indexed by choice position
    choices_lower = [c.lower().strip() for c in choices]

    from wheeler_memory.constants import RECALL_MIN_SIM

    for hit in hits:
        text = hit.get("text", "")
        sim = hit.get("similarity", 0.0)
        if sim < RECALL_MIN_SIM:
            continue
        matched = False

        # Strategy 1: letter regex (MMLU learn-mode format "A: B. choice_text")
        m = re.search(r"A:\s*([A-D])\.", text)
        if m:
            idx = CHOICES.index(m.group(1))
            votes[idx] += sim
            matched = True

        # Strategy 2: text match (SciQ/ARC format "A: answer_text")
        if not matched:
            # Extract answer portion after "A:" or "A: "
            ans_m = re.search(r"A:\s*(.+?)(?:\.|$)", text)
            if ans_m:
                answer = ans_m.group(1).lower().strip()
                if len(answer) >= 2:  # skip trivially short
                    for ci, cl in enumerate(choices_lower):
                        if answer in cl or cl in answer:
                            votes[ci] += sim
                            matched = True
                            break

    best_idx = int(max(range(4), key=lambda i: votes[i]))
    if votes[best_idx] == 0.0:
        return -1, 0.0
    return best_idx, votes[best_idx]


def score_multi_choice(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    debug: bool = False,
    encoder: str = "hippocampus",
) -> tuple[int, float]:
    """Recall via question, then score each choice by n-gram overlap with recalled text.

    Two-stage approach:
    1. Use question-only query to recall relevant stored facts (attractor similarity)
    2. For each recalled fact, score each choice by character n-gram overlap with
       the answer portion — the choice with highest weighted overlap wins.

    Returns (predicted_index, confidence) or (-1, 0.0) on failure.
    """
    try:
        frame_fn, _ = _get_encoder_fns(encoder)
    except Exception:
        from wheeler_memory.hashing import hash_to_frame
        frame_fn = hash_to_frame

    if cache is None:
        return -1, 0.0

    from wheeler_memory.constants import RECALL_MIN_SIM

    # Probe each choice against stored answer-only attractors
    # Stored format: "A: {letter}. {text}" — probe with same format per choice
    choice_sims = []
    letters = ["A", "B", "C", "D"]
    for ci, choice in enumerate(choices):
        probe_text = f"A: {letters[ci]}. {choice}"
        frame = frame_fn(probe_text)
        hits = cache.search(frame, top_k=recall_k)

        max_sim = 0.0
        for hit in hits:
            sim = hit.get("similarity", 0.0)
            if sim >= RECALL_MIN_SIM:
                max_sim = max(max_sim, sim)

        choice_sims.append(max_sim)

        if debug:
            top_text = hits[0]["text"][:60] if hits else ""
            print(f"      [debug] choice {CHOICES[ci]}: max_sim={max_sim:.3f}  top='{top_text}'")

    if max(choice_sims) == 0.0:
        return -1, 0.0

    best_idx = int(max(range(len(choice_sims)), key=lambda i: choice_sims[i]))
    return best_idx, choice_sims[best_idx]


def score_cortex(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    classifier_weights=None,  # ClassifierWeights | None
    traj_cache=None,  # TrajectoryCache | None
    encoder: str = "hippocampus",
) -> tuple[int, float]:
    """Score using full cortex pipeline: L1 graph → L2 settlement → SCM → optional L3.

    When traj_cache is provided, uses hybrid retrieval: attractor similarity
    weighted with trajectory similarity via TRAJECTORY_ALPHA.

    Returns (predicted_index, confidence).
    """
    from wheeler_memory.dynamics import evolve_and_interpret, apply_ca_dynamics
    from wheeler_memory.cortex import cortex_reason
    from wheeler_memory.cortex_scm import (
        compute_scm,
        score_coevolution_convergence,
        score_coevolution_spread,
        score_coevolution_energy,
    )
    from wheeler_memory.constants import TRAJECTORY_ALPHA, SALIENCE_THRESHOLD_MED

    frame_fn, _ = _get_encoder_fns(encoder)

    # 1. Encode question → frame → attractor
    q_frame = frame_fn(question)
    q_result = evolve_and_interpret(q_frame)
    q_attractor = q_result["attractor"].flatten().astype(np.float32)

    # 2. Encode each choice → attractor + co-evolution dynamics
    choice_attractors = []
    coevo_ticks = []
    coevo_energies = []
    for c in choices:
        c_frame = frame_fn(f"{question} {c}")
        c_result = evolve_and_interpret(c_frame)
        choice_attractors.append(c_result["attractor"].flatten().astype(np.float32))

        # Co-evolution: blend question + choice frames, evolve together
        # The analog CA dynamics find structural coherence between Q and A
        blended = np.tanh(0.5 * q_frame + 0.5 * c_frame).astype(np.float32)
        frame = blended.copy()
        initial_delta = None
        final_delta = 0.0
        ticks = 0
        for t in range(200):  # shorter budget for co-evolution
            frame_old = frame
            frame = apply_ca_dynamics(frame)
            delta = float(np.abs(frame - frame_old).mean())
            if initial_delta is None:
                initial_delta = delta
            final_delta = delta
            ticks = t + 1
            if delta < SALIENCE_THRESHOLD_MED:
                break
        coevo_ticks.append(ticks)
        # Energy drop: how much the dynamics dissipated
        energy_drop = (
            (initial_delta - final_delta) / max(initial_delta, 1e-10)
            if initial_delta
            else 0.0
        )
        coevo_energies.append(max(0.0, energy_drop))
    choice_attractors = np.array(choice_attractors)  # (4, 4096)

    # 3. Retrieve top-K from cache (if available)
    #    With hybrid retrieval: get attractor hits, then re-rank by combined score
    retrieved_attractors = []
    retrieved_sims = []
    if cache is not None:
        hits = cache.search(q_frame, top_k=recall_k)

        # Hybrid re-ranking if trajectory cache available
        if traj_cache is not None and len(traj_cache) > 0:
            from wheeler_memory.trajectory import compute_signature

            q_sig = compute_signature(q_frame)
            traj_hits = traj_cache.search(q_sig, top_k=recall_k * 3)
            traj_map = {h["text"]: h["traj_similarity"] for h in traj_hits}

            # Compute combined scores for attractor hits
            alpha = TRAJECTORY_ALPHA
            scored_hits = []
            for h in hits:
                traj_sim = traj_map.get(h["text"], 0.0)
                combined = alpha * h["similarity"] + (1 - alpha) * traj_sim
                scored_hits.append({**h, "combined": combined, "traj_sim": traj_sim})
            # Re-sort by combined score
            scored_hits.sort(key=lambda x: x["combined"], reverse=True)
            hits = scored_hits[:recall_k]

        for h in hits:
            # We need the actual attractor, not just text
            # Re-encode from text (cache stores text + similarity)
            r_frame = frame_fn(h["text"])
            r_result = evolve_and_interpret(r_frame)
            retrieved_attractors.append(
                r_result["attractor"].flatten().astype(np.float32)
            )
            retrieved_sims.append(h.get("combined", h["similarity"]))

    # 4. Build attractor ensemble: retrieved + choices
    all_attractors = []
    all_sims = []
    if retrieved_attractors:
        all_attractors.extend(retrieved_attractors)
        all_sims.extend(retrieved_sims)
    # Always include the 4 choice attractors
    for i, ca in enumerate(choice_attractors):
        all_attractors.append(ca)
        # Choice similarity = Pearson between query and choice attractor
        q_centered = q_attractor - q_attractor.mean()
        c_centered = ca - ca.mean()
        q_std = q_centered.std()
        c_std = c_centered.std()
        if q_std > 1e-10 and c_std > 1e-10:
            sim = float(
                np.dot(q_centered, c_centered) / (len(q_centered) * q_std * c_std)
            )
        else:
            sim = 0.0
        all_sims.append(sim)

    all_attractors = np.array(all_attractors)
    all_sims = np.array(all_sims)

    # 5. Run cortex
    cortex_result = cortex_reason(all_attractors, all_sims)
    graph = cortex_result["graph"]
    settlement = cortex_result["settlement"]

    # 6. Extract choice similarities from settled opinions
    n_retrieved = len(retrieved_attractors)
    choice_settled = settlement["settled"][-4:]  # last 4 entries are choices

    # 7. Compute co-evolution SCM layers
    coevo_ticks_arr = np.array(coevo_ticks, dtype=np.float32)
    coevo_energies_arr = np.array(coevo_energies, dtype=np.float32)
    coevo_conv = score_coevolution_convergence(coevo_ticks_arr, max_iters=200)
    coevo_spr = score_coevolution_spread(coevo_ticks_arr)
    coevo_eng = score_coevolution_energy(coevo_energies_arr)

    # 8. If we have L3 classifier weights, use them
    if classifier_weights is not None:
        from wheeler_memory.cortex_classifier import classify

        choice_sims = all_sims[-4:]  # Get the 4 choice similarities
        scm_result = compute_scm(
            np.array(all_sims[:n_retrieved]) if n_retrieved > 0 else np.array([]),
            graph.adjacency,
            graph.cluster_labels,
            settlement["settled"],
            settlement["settled"],  # prev = initial for first pass
            float(choice_sims.max()) if len(choice_sims) > 0 else 0.0,
            coevo_convergence=coevo_conv,
            coevo_spread=coevo_spr,
            coevo_energy=coevo_eng,
        )
        scm_layers = np.array(
            [
                scm_result.temperature,
                scm_result.salience,
                scm_result.energy,
                scm_result.integration,
                scm_result.polarity,
                scm_result.net_warrant,
                scm_result.explanation_readiness,
            ]
        )
        coevo_layers = np.array(
            [
                scm_result.coevolution_convergence,
                scm_result.coevolution_spread,
                scm_result.coevolution_energy,
            ],
            dtype=np.float32,
        )
        # Pad/truncate settlement to CORTEX_K to match trained classifier input dim
        from wheeler_memory.constants import CORTEX_K

        settled_fixed = np.zeros(CORTEX_K, dtype=np.float32)
        settled_raw = settlement["settled"]
        n = min(len(settled_raw), CORTEX_K)
        settled_fixed[:n] = settled_raw[:n]
        pred_idx, confidence = classify(
            settled_fixed, choice_sims, scm_layers, classifier_weights, coevo_layers
        )
        return pred_idx, confidence

    # 9. Without L3: combine settled opinions with co-evolution speed
    # Faster co-evolution convergence = better structural coherence
    # Normalize co-evolution ticks: lower ticks → higher score
    max_ticks = max(coevo_ticks) if coevo_ticks else 1
    coevo_scores = np.array(
        [1.0 - t / max(max_ticks, 1) for t in coevo_ticks], dtype=np.float32
    )

    # Combine: 70% settlement + 30% co-evolution speed
    combined = 0.7 * choice_settled + 0.3 * coevo_scores
    pred_idx = int(np.argmax(combined))
    confidence = float(combined[pred_idx])
    return pred_idx, confidence


def score_ternary(
    question: str,
    choices: list[str],
    precomputed_frames: list | None = None,
    encoder: str = "hippocampus",
) -> tuple[int, float]:
    """Score choices by net ternary positivity: count(+1) - count(-1).

    Returns (predicted_index, confidence).
    confidence = net_score / total_cells for the winning choice.
    """
    from wheeler_memory.dynamics import evolve_and_interpret, snap_to_ternary

    if precomputed_frames is not None:
        frames = precomputed_frames
    else:
        frame_fn, _ = _get_encoder_fns(encoder)
        frames = [frame_fn(f"{question} {c}") for c in choices]

    scores = []
    for frame in frames:
        result = evolve_and_interpret(frame)
        ternary = snap_to_ternary(result["attractor"])
        net = int(np.sum(ternary == 1)) - int(np.sum(ternary == -1))
        scores.append(net)

    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    total_cells = 64 * 64
    confidence = scores[best] / total_cells
    return best, confidence


def score_ternary_ensemble(
    question: str,
    choices: list[str],
    encoders: list[str] = ("hippocampus", "hippo-word"),
    margin_threshold: int = 30,
) -> tuple[int, float]:
    """Adaptive routing: use word_count to pick encoder, then score with it.

    Diagnostic showed blend wins on longer questions (word_count > threshold).
    Short/technical questions → hippocampus. Long/conceptual → hippo-word.
    """
    from wheeler_memory.dynamics import evolve_and_interpret, snap_to_ternary
    from wheeler_memory.constants import ENSEMBLE_MARGIN_THRESHOLD
    import re as _re

    words = [w for w in _re.split(r"[^a-z0-9]+", question.lower()) if w]
    word_count = len(words)

    # Route based on question length
    if word_count >= ENSEMBLE_MARGIN_THRESHOLD:
        enc = encoders[1]  # hippo-word for long questions
    else:
        enc = encoders[0]  # hippocampus for short questions

    frame_fn, _ = _get_encoder_fns(enc)
    total_cells = 64 * 64

    scores = []
    for c in choices:
        frame = frame_fn(f"{question} {c}")
        result = evolve_and_interpret(frame)
        ternary = snap_to_ternary(result["attractor"])
        net = int(np.sum(ternary == 1)) - int(np.sum(ternary == -1))
        scores.append(net)

    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return best, scores[best] / total_cells


def score_ternary_retrieval(
    question: str,
    choices: list[str],
    recall_k: int = 5,
    cache: "AttractorCache | None" = None,
    precomputed_frames: list | None = None,
    encoder: str = "hippocampus",
) -> tuple[int, float]:
    """Score choices by ternary overlap with retrieved attractors.

    For each choice, encode+evolve → ternary roles.  Retrieve top-K similar
    attractors from the cache (using pre-computed ternary — no re-encoding).
    Compute fraction of matching cells, weighted by retrieval similarity.

    Falls back to net-positivity scoring when no cache/hits available.

    Returns (predicted_index, confidence).
    """
    from wheeler_memory.dynamics import evolve_and_interpret, snap_to_ternary

    frame_fn, _ = _get_encoder_fns(encoder)

    if precomputed_frames is not None:
        frames = precomputed_frames
    else:
        frames = [frame_fn(f"{question} {c}") for c in choices]

    # Retrieve once using question-only frame (shared across all choices)
    hits = []
    if cache is not None:
        q_frame = frame_fn(question)
        hits = cache.search_ternary(q_frame, top_k=recall_k)

    total_cells = 64 * 64

    # Build retrieved ternary matrix (K, 4096) + similarity vector (K,)
    if hits:
        r_matrix = np.stack([h["ternary"] for h in hits])  # (K, 4096) int8
        r_sims = np.array([h["similarity"] for h in hits])  # (K,)
    else:
        r_matrix = None
        r_sims = None

    scores = []
    for frame in frames:
        result = evolve_and_interpret(frame)
        choice_ternary = snap_to_ternary(result["attractor"]).flatten()  # (4096,)

        if r_matrix is None:
            # No retrieval — fall back to net positivity
            net = int(np.sum(choice_ternary == 1)) - int(np.sum(choice_ternary == -1))
            scores.append(net / total_cells)
            continue

        # Vectorized overlap: fraction of matching cells per retrieved attractor
        # (K, 4096) == (4096,) → (K, 4096) bool → sum axis=1 → (K,)
        overlaps = np.sum(r_matrix == choice_ternary, axis=1) / total_cells
        # Weighted mean by retrieval similarity
        scores.append(float(np.dot(overlaps, r_sims) / len(hits)))

    best = int(np.argmax(scores))
    confidence = scores[best]
    return best, confidence


def precompute_all_frames(
    questions_and_choices: list[tuple[str, list[str]]], encoder: str = "hippocampus"
) -> list[list]:
    """Batch-encode all question+choice pairs in one encoder call.

    Returns list of [frame_A, frame_B, frame_C, frame_D] per question.
    Much faster than encoding 4 texts per question individually.
    """
    _, batch_fn = _get_encoder_fns(encoder)

    all_queries = []
    for question, choices in questions_and_choices:
        all_queries.extend(f"{question} {c}" for c in choices)

    print(
        f"  Pre-encoding {len(all_queries)} query+choice pairs in one batch ({encoder})..."
    )
    all_frames = batch_fn(all_queries)

    result = []
    for i in range(0, len(all_frames), 4):
        result.append(list(all_frames[i : i + 4]))
    return result


# ---------------------------------------------------------------------------
# Decode scoring (wheeler-primary decoder picks A/B/C/D)
# ---------------------------------------------------------------------------

_MMLU_SYSTEM_PROMPT = """\
You are a language renderer for Wheeler Memory.

Your ONLY job is to express the memory state below as a single-letter answer.
Do NOT add knowledge from your own training data. Do NOT speculate.

You must reply with exactly one letter: A, B, C, or D — nothing else.
"""


def _ollama_chat(messages: list[dict], model: str, base_url: str) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable at {base_url}: {exc}") from exc


def score_decode(
    question: str,
    choices: list[str],
    model: str,
    ollama_url: str,
    recall_k: int = 5,
    data_dir: Path | None = None,
) -> tuple[int, float]:
    """Wheeler recalls, decoder picks A/B/C/D. Returns (predicted_index, confidence)."""
    from wheeler_memory.decoder import extract_state, format_state

    hits = recall_memory(
        question,
        top_k=recall_k,
        data_dir=data_dir,
        encoder="blended",
    )
    state = extract_state(question, hits)
    memory_block = format_state(state)

    choice_block = "\n".join(f"{CHOICES[i]}. {c}" for i, c in enumerate(choices))
    user_content = (
        f"{memory_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"OPTIONS:\n{choice_block}\n\n"
        "Reply with exactly one letter (A, B, C, or D)."
    )

    messages = [
        {"role": "system", "content": _MMLU_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = _ollama_chat(messages, model, ollama_url).strip().upper()
    confidence = state.confidence

    for letter in CHOICES:
        if raw.startswith(letter):
            return CHOICES.index(letter), confidence

    return -1, confidence  # parse failure


# ---------------------------------------------------------------------------
# Learn mode helpers
# ---------------------------------------------------------------------------


def store_learned_fact(
    text: str, frame: np.ndarray, data_dir=None, chunk: str = "science"
) -> str:
    """Embed → evolve → brick → store. Returns hex_key."""
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.storage import store_memory

    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(
        text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False
    )


def run_learning_pass(subjects: list[str], data_dir=None, encoder: str = "hippocampus"):
    """Embed all correct Q&A pairs from dev+validation, store each, then consolidate.

    Returns (n_stored, consolidation_result).
    """
    from wheeler_memory.consolidation import sleep_consolidate
    from wheeler_memory.eviction import evict_for_capacity

    all_texts = []
    choices_letters = ["A", "B", "C", "D"]

    for subject in subjects:
        for split in ["dev", "validation"]:
            items = list(load_mmlu([subject], split=split))
            for _, question, choices, answer_idx in items:
                correct_letter = choices_letters[answer_idx]
                correct_text = choices[answer_idx]
                all_texts.append(f"Q: {question} A: {correct_letter}. {correct_text}")
                # Answer-only attractor for multi-choice probing
                all_texts.append(f"A: {correct_letter}. {correct_text}")

    if not all_texts:
        return 0, None

    _, batch_fn = _get_encoder_fns(encoder)
    print(f"\n  [learn] Encoding {len(all_texts)} correct Q&A pairs ({encoder})...")
    frames = batch_fn(all_texts)

    stored = 0
    for text, frame in zip(all_texts, frames):
        store_learned_fact(text, frame, data_dir=data_dir, chunk="science")
        stored += 1
    print(f"  [learn] Stored {stored} facts.")

    d = Path(data_dir) if data_dir else Path.home() / ".wheeler_memory"

    # Train word co-occurrence vectors from stored texts (SVD on PMI)
    from wheeler_memory.word_encoder import train_word_vectors, save_word_vectors

    print("  [learn] Training word co-occurrence vectors (SVD on PMI)...")
    vectors, vocab = train_word_vectors(data_dir=str(d))
    save_word_vectors(vectors, vocab, data_dir=str(d))
    print(f"  [learn] Learned {len(vocab)} word vectors ({vectors.shape[1]}-dim)")

    # Reset cached learned vectors so encoder picks up new ones
    import wheeler_memory.word_encoder as _we

    _we._learned_checked = False
    _we._learned_vectors = None
    _we._learned_word2idx = None

    print("  [learn] Running eviction to enforce capacity limits...")
    evict_for_capacity(d)

    print("  [learn] Running sleep consolidation on science chunk...")
    result = sleep_consolidate(d, chunk="science")
    n_consolidated = len(result.memories_consolidated)
    print(
        f"  [learn] Consolidation: {n_consolidated} memories consolidated, "
        f"{result.total_frames_before} → {result.total_frames_after} frames"
    )

    return stored, result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    subjects: list[str],
    mode: str,
    samples: int | None,
    recall_k: int,
    model: str,
    ollama_url: str,
    data_dir: Path | None,
    output: Path | None,
    split: str,
    use_embed: bool = True,
    debug_recall: bool = False,
    classifier_weights_path: str | None = None,
    encoder: str = "hippocampus",
    classify_errors: bool = False,
) -> dict:
    results = {}  # subject → {correct, total, accuracy}
    all_rows = []  # for TSV output
    error_classifications = {}  # classification label → count (SYNTHESIS, NOVEL, HALLUCINATION)

    total_correct = 0
    total_questions = 0

    embed_label = "embed" if use_embed else "hash"
    is_learn_mode = mode == "learn"
    print(f"\n{'=' * 70}")
    print(f"  WHEELER MMLU BENCHMARK")
    if is_learn_mode:
        print(
            f"  Mode: LEARN → SEMANTIC   Subjects: {len(subjects)}   Test split: {split}"
        )
        print(f"  Phase 1: learn dev+validation → consolidate")
        print(f"  Phase 2: semantic test on {split} split")
    else:
        print(
            f"  Mode: {mode.upper()}({embed_label})   Encoder: {encoder}   Subjects: {len(subjects)}   Split: {split}"
        )
    if samples:
        print(f"  Samples per subject: {samples}")
    if classify_errors:
        print(f"  Error Classification: ON (post-hoc hallucination discrimination)")
    print(f"{'=' * 70}")

    # Learn mode: store dev+validation facts, consolidate, then test on test split
    if is_learn_mode:
        n_stored, cons = run_learning_pass(subjects, data_dir, encoder=encoder)
        # Rebuild attractor cache with newly stored memories
        print(f"\n  Rebuilding attractor cache after learning pass...")
        mode = "semantic"

    print()

    # Load classifier weights if provided (for cortex L3)
    classifier_weights = None
    if classifier_weights_path and mode == "cortex":
        from wheeler_memory.cortex_classifier import load_weights

        print(f"  Loading classifier weights from {classifier_weights_path}...")
        classifier_weights = load_weights(classifier_weights_path)

    # Build attractor cache once for the entire benchmark run
    cache = (
        AttractorCache(data_dir)
        if mode in ("semantic", "recall-text", "multi-choice", "cortex", "ternary-retrieval", "reverse-lookup", "spatial")
        else None
    )

    # Build trajectory cache for hybrid retrieval (cortex mode)
    traj_cache = None
    if mode == "cortex":
        try:
            from wheeler_memory.trajectory_cache import TrajectoryCache

            traj_cache = TrajectoryCache(data_dir)
            if len(traj_cache) == 0:
                print(
                    "  No trajectory signatures found — using attractor-only retrieval"
                )
                traj_cache = None
        except Exception as e:
            print(
                f"  Trajectory cache unavailable ({e}) — using attractor-only retrieval"
            )

    for subj in subjects:
        correct = 0
        total = 0
        parse_failures = 0

        print(f"  [{subj}]")

        # Load all questions for this subject upfront so we can pre-encode
        subject_items = list(load_mmlu([subj], split, samples))

        # Pre-encode all question+choice pairs in one batch (semantic mode only)
        precomputed = None
        if (
            mode in ("semantic", "ternary", "ternary-retrieval", "spatial")
            and use_embed
            and subject_items
        ):
            qc_pairs = [(q, ch) for _, q, ch, _ in subject_items]
            precomputed = precompute_all_frames(qc_pairs, encoder=encoder)

        for idx, (subject, question, choices, answer_idx) in enumerate(subject_items):
            t0 = time.time()

            if mode == "semantic":
                frames = precomputed[idx] if precomputed is not None else None
                pred_idx, scores = score_semantic(
                    question, choices, recall_k, cache, frames
                )
                confidence = max(scores)
            elif mode == "ternary":
                frames = precomputed[idx] if precomputed is not None else None
                pred_idx, confidence = score_ternary(
                    question, choices, frames, encoder=encoder
                )
            elif mode == "ternary-retrieval":
                frames = precomputed[idx] if precomputed is not None else None
                pred_idx, confidence = score_ternary_retrieval(
                    question, choices, recall_k, cache, frames, encoder=encoder
                )
            elif mode == "ternary-ensemble":
                pred_idx, confidence = score_ternary_ensemble(question, choices)
            elif mode == "recall-text":
                _dbg = debug_recall and idx < 3
                pred_idx, confidence = score_recall_text(
                    question, choices, recall_k, cache, debug=_dbg, encoder=encoder
                )
            elif mode == "multi-choice":
                _dbg = debug_recall and idx < 3
                pred_idx, confidence = score_multi_choice(
                    question, choices, recall_k, cache, debug=_dbg, encoder=encoder
                )
            elif mode == "reverse-lookup":
                pred_idx, scores = score_reverse_lookup(
                    question, choices, recall_k, cache, encoder=encoder
                )
                confidence = max(scores)
            elif mode == "spatial":
                frames = precomputed[idx] if precomputed is not None else None
                pred_idx, scores = score_spatial(
                    question, choices, recall_k, cache, frames
                )
                confidence = max(scores)
            elif mode == "cortex":
                pred_idx, confidence = score_cortex(
                    question,
                    choices,
                    recall_k,
                    cache,
                    classifier_weights,
                    traj_cache,
                    encoder=encoder,
                )
            else:
                try:
                    pred_idx, confidence = score_decode(
                        question, choices, model, ollama_url, recall_k, data_dir
                    )
                except RuntimeError as exc:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                    sys.exit(1)

            elapsed = time.time() - t0
            is_correct = pred_idx == answer_idx
            if pred_idx == -1:
                parse_failures += 1

            if is_correct:
                correct += 1
            total += 1

            # Post-hoc error classification (hallucination vs synthesis vs novel)
            classification = None
            if classify_errors and not is_correct and 0 <= pred_idx <= 3:
                try:
                    from wheeler_memory.theories.metrics import classify_output

                    wrong_choice = choices[pred_idx]
                    error_text = f"{question} {wrong_choice}"

                    # Use cached attractors if available (semantic/ternary/cortex/ternary-retrieval modes)
                    known_attractors = []
                    if cache is not None:
                        # Extract attractors from cache (stored as (N, 4096) matrix)
                        if len(cache._matrix) > 0:
                            known_attractors = [
                                cache._matrix[i].reshape(64, 64)
                                for i in range(len(cache._matrix))
                            ]

                    classification = classify_output(error_text, known_attractors)
                    error_classifications[classification] = (
                        error_classifications.get(classification, 0) + 1
                    )
                except Exception as e:
                    if debug_recall:
                        print(
                            f"      [Error classification failed: {e}]", file=sys.stderr
                        )

            marker = "✓" if is_correct else "✗"
            pred_letter = CHOICES[pred_idx] if 0 <= pred_idx <= 3 else "?"
            true_letter = CHOICES[answer_idx]
            status_str = f"    {marker} pred={pred_letter} true={true_letter} conf={confidence:.3f} ({elapsed:.1f}s)"
            if classification:
                status_str += f" [{classification}]"
            print(f"{status_str} {question[:55]}...")

            row_data = {
                "subject": subject,
                "question": question[:100],
                "true": true_letter,
                "pred": pred_letter,
                "correct": is_correct,
                "confidence": round(confidence, 4),
                "elapsed_s": round(elapsed, 2),
            }
            if classification:
                row_data["error_classification"] = classification
            all_rows.append(row_data)

        accuracy = correct / total if total else 0.0
        results[subj] = {"correct": correct, "total": total, "accuracy": accuracy}
        total_correct += correct
        total_questions += total

        if parse_failures:
            print(
                f"    → {correct}/{total} correct ({accuracy:.1%})  [{parse_failures} parse failures]"
            )
        else:
            print(f"    → {correct}/{total} correct ({accuracy:.1%})")

    overall = total_correct / total_questions if total_questions else 0.0

    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {total_correct}/{total_questions} = {overall:.1%}")
    print(f"{'=' * 70}")
    print(f"\n  Per-subject accuracy:")
    for subj, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        bar = "█" * int(r["accuracy"] * 20)
        print(f"    {subj:<40} {r['accuracy']:5.1%}  {bar}")

    # Print error classification summary (if --classify-errors was used)
    if classify_errors and error_classifications:
        total_errors = sum(error_classifications.values())
        print(f"\n  Error Classification Summary ({total_errors} wrong answers):")
        print(f"  {'=' * 70}")
        for label in ["HALLUCINATION", "NOVEL", "SYNTHESIS"]:
            count = error_classifications.get(label, 0)
            pct = count / total_errors * 100 if total_errors else 0
            print(f"    {label:<20} {count:4d}  ({pct:5.1f}%)")
        print(f"  {'=' * 70}")

    # Save results
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_rows[0].keys()) if all_rows else []
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Results saved to: {output}")

    return {"subjects": results, "overall": overall, "total": total_questions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wheeler-mmlu",
        description="Run Wheeler Memory against the MMLU benchmark.",
    )
    p.add_argument(
        "--subjects",
        "-s",
        nargs="+",
        metavar="SUBJECT",
        help="MMLU subject name(s) to evaluate.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all 57 MMLU subjects.",
    )
    p.add_argument(
        "--list-subjects",
        action="store_true",
        help="Print all available subjects and exit.",
    )
    p.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Max questions per subject (default: all).",
    )
    p.add_argument(
        "--mode",
        choices=[
            "semantic",
            "decode",
            "learn",
            "recall-text",
            "cortex",
            "ternary",
            "ternary-retrieval",
            "ternary-ensemble",
            "multi-choice",
            "reverse-lookup",
            "spatial",
        ],
        default="semantic",
        help="Scoring mode: 'semantic' (Wheeler-native), 'decode' (small model), 'learn' (store dev+val → consolidate → test), 'recall-text' (extract answer letter from recalled facts), 'cortex' (full cortex pipeline), 'ternary' (net +1/-1 cell count), 'ternary-retrieval' (ternary overlap with retrieved attractors). Default: semantic.",
    )
    p.add_argument(
        "--split",
        choices=["test", "validation", "dev"],
        default="test",
        help="Dataset split to use (default: test).",
    )
    p.add_argument(
        "--recall-k",
        type=int,
        default=None,
        metavar="K",
        help="Memories to recall per query (default: from constants.RECALL_K).",
    )
    p.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        metavar="MODEL",
        help="Ollama model for decode mode (default: qwen2.5:1.5b).",
    )
    p.add_argument(
        "--ollama",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Wheeler Memory data directory.",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Save per-question results to this TSV/CSV file.",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        help="Use hash-based frames instead of sentence embeddings (much faster, lower accuracy).",
    )
    p.add_argument(
        "--debug-recall",
        action="store_true",
        help="Print top-5 recalled texts for first 3 questions per subject (recall-text mode).",
    )
    p.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to trained cortex classifier weights (.npz) for cortex mode L3 classification.",
    )
    p.add_argument(
        "--encoder",
        choices=["hippocampus", "word", "word-blended", "hippo-word"],
        default="hippocampus",
        help="Encoder for frame generation: 'hippocampus' (char n-grams, default), "
        "'word' (word-level random indexing), 'word-blended' (word + language wheeler).",
    )
    p.add_argument(
        "--classify-errors",
        action="store_true",
        help="Post-hoc: classify wrong answers as SYNTHESIS/NOVEL/HALLUCINATION and report summary.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.list_subjects:
        print("\n".join(MMLU_SUBJECTS))
        return

    if args.all:
        subjects = MMLU_SUBJECTS
    elif args.subjects:
        subjects = args.subjects
    else:
        print(
            "ERROR: Specify subjects with --subjects or use --all.\n"
            "       Run --list-subjects to see all options.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate subjects
    unknown = [s for s in subjects if s not in MMLU_SUBJECTS]
    if unknown:
        print(f"ERROR: Unknown subjects: {unknown}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else None

    from wheeler_memory.constants import RECALL_K

    recall_k = args.recall_k if args.recall_k is not None else RECALL_K

    run_benchmark(
        subjects=subjects,
        mode=args.mode,
        samples=args.samples,
        recall_k=recall_k,
        model=args.model,
        ollama_url=args.ollama,
        data_dir=data_dir,
        output=args.output,
        split=args.split,
        use_embed=not args.no_embed,
        debug_recall=args.debug_recall,
        classifier_weights_path=args.classifier_weights,
        encoder=args.encoder,
        classify_errors=args.classify_errors,
    )


if __name__ == "__main__":
    main()

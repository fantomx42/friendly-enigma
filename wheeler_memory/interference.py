"""Three-grid interference engine.

The answer to any query is not stored in either content grid — it exists
only where both content grids peak through gaps in the SCM pressure topology.

    Answer(i,j) = Corpus(i,j) * Experiential(i,j) * (1 - |SCM(i,j)|)

Four interference states per cell:
  GROUNDED       — C:peak  E:peak  SCM:open  — live knowledge, best answers
  ABSORBED       — C:peak  E:none  SCM:open  — know it, don't remember learning
  UNCONSOLIDATED — C:none  E:peak  SCM:open  — remember event, not crystallized
  CONTESTED      — C:peak  E:peak  SCM:closed — both have signal, trust blocks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import (
    CORPUS_MAX_PUSH,
    CORPUS_SLOPE_FLOW,
    EXPERIENTIAL_MAX_PUSH,
    EXPERIENTIAL_SLOPE_FLOW,
    HALLUCINATION_THRESHOLD,
    INTERFERENCE_PEAK_THRESHOLD,
    SCM_GAP_THRESHOLD,
)

# Interference state labels
GROUNDED = "GROUNDED"
ABSORBED = "ABSORBED"
UNCONSOLIDATED = "UNCONSOLIDATED"
CONTESTED = "CONTESTED"
SILENT = "SILENT"  # no peaks in either grid


@dataclass
class InterferenceResult:
    """Result of three-grid interference computation."""

    pattern: np.ndarray  # (64, 64) interference pattern
    state_map: np.ndarray  # (64, 64) uint8 — state codes per cell
    dominant_state: str  # most frequent non-SILENT state
    grounded_fraction: float
    absorbed_fraction: float
    unconsolidated_fraction: float
    contested_fraction: float
    signal_strength: float  # mean |pattern| in gap regions

    # State code mapping
    STATE_CODES = {SILENT: 0, GROUNDED: 1, ABSORBED: 2, UNCONSOLIDATED: 3, CONTESTED: 4}
    CODE_NAMES = {v: k for k, v in STATE_CODES.items()}


def compute_interference(
    corpus_att: np.ndarray,
    experiential_att: np.ndarray | None,
    scm_grid: np.ndarray,
    peak_threshold: float = INTERFERENCE_PEAK_THRESHOLD,
    gap_threshold: float = SCM_GAP_THRESHOLD,
) -> InterferenceResult:
    """Compute the three-grid interference pattern.

    Parameters
    ----------
    corpus_att : (64, 64) float32 — corpus attractor.
    experiential_att : (64, 64) float32 — experiential attractor, or None
        if no experiential counterpart exists (defaults to zeros → ABSORBED).
    scm_grid : (64, 64) float32 — SCM trust topology.

    Returns
    -------
    InterferenceResult with the interference pattern and per-cell state map.
    """
    if experiential_att is None:
        experiential_att = np.zeros_like(corpus_att)

    # The answer equation
    scm_openness = 1.0 - np.abs(scm_grid)
    pattern = corpus_att * experiential_att * scm_openness

    # Per-cell state classification
    c_peak = np.abs(corpus_att) > peak_threshold
    e_peak = np.abs(experiential_att) > peak_threshold
    scm_open = np.abs(scm_grid) < gap_threshold

    state_map = np.zeros(corpus_att.shape, dtype=np.uint8)
    state_map[c_peak & e_peak & scm_open] = InterferenceResult.STATE_CODES[GROUNDED]
    state_map[c_peak & ~e_peak & scm_open] = InterferenceResult.STATE_CODES[ABSORBED]
    state_map[~c_peak & e_peak & scm_open] = InterferenceResult.STATE_CODES[UNCONSOLIDATED]
    state_map[c_peak & e_peak & ~scm_open] = InterferenceResult.STATE_CODES[CONTESTED]

    total = float(corpus_att.size)
    grounded_frac = float((state_map == 1).sum()) / total
    absorbed_frac = float((state_map == 2).sum()) / total
    unconsolidated_frac = float((state_map == 3).sum()) / total
    contested_frac = float((state_map == 4).sum()) / total

    # Dominant non-SILENT state
    state_counts = {
        GROUNDED: grounded_frac,
        ABSORBED: absorbed_frac,
        UNCONSOLIDATED: unconsolidated_frac,
        CONTESTED: contested_frac,
    }
    dominant = max(state_counts, key=state_counts.get)

    # Signal strength: mean |pattern| in open gap regions
    gap_mask = scm_open
    if gap_mask.any():
        signal = float(np.abs(pattern[gap_mask]).mean())
    else:
        signal = 0.0

    return InterferenceResult(
        pattern=pattern,
        state_map=state_map,
        dominant_state=dominant,
        grounded_fraction=grounded_frac,
        absorbed_fraction=absorbed_frac,
        unconsolidated_fraction=unconsolidated_frac,
        contested_fraction=contested_frac,
        signal_strength=signal,
    )


def interference_score(
    query_corpus_att: np.ndarray,
    query_experiential_att: np.ndarray | None,
    stored_corpus_att: np.ndarray,
    stored_experiential_att: np.ndarray | None,
    scm_grid: np.ndarray,
) -> tuple[float, str, InterferenceResult]:
    """Compute interference-based similarity score between query and stored memory.

    Returns (score, dominant_state) where score combines corpus and experiential
    Pearson correlations, gated by SCM openness.
    """
    from scipy.stats import pearsonr

    # Corpus similarity
    c_corr, _ = pearsonr(query_corpus_att.flatten(), stored_corpus_att.flatten())
    c_sim = float(c_corr) if not np.isnan(c_corr) else 0.0

    # Experiential similarity (0 if either side has no experiential counterpart)
    if query_experiential_att is not None and stored_experiential_att is not None:
        e_corr, _ = pearsonr(
            query_experiential_att.flatten(), stored_experiential_att.flatten()
        )
        e_sim = float(e_corr) if not np.isnan(e_corr) else 0.0
    else:
        e_sim = 0.0

    # SCM gating: mean openness across the grid
    mean_openness = float((1.0 - np.abs(scm_grid)).mean())

    # Combined score: both Pearson channels gated by trust
    score = (c_sim + e_sim) * mean_openness

    # Determine dominant state for this memory pair
    ir = compute_interference(
        stored_corpus_att,
        stored_experiential_att,
        scm_grid,
    )

    return score, ir.dominant_state, ir


def recall_with_interference(
    text: str,
    top_k: int = 5,
    data_dir: "str | Path | None" = None,
    *,
    chunk: str | None = None,
    encoder: str | None = None,
    use_embedding: bool = False,
) -> tuple[list[dict], str, float]:
    """Recall memories through the three-grid interference pipeline.

    Wraps the sacred ``recall_memory()`` with three-grid re-ranking:
    1. Corpus recall via Pearson correlation (existing pipeline).
    2. For each corpus hit, look up its experiential counterpart.
    3. Re-score using ``interference_score()`` (corpus + experiential gated by SCM).

    Returns
    -------
    (results, dominant_state, scm_openness) where results are dicts
    in the same format as ``recall_memory()`` with added keys
    ``interference_score`` and ``interference_state``.
    """
    from .chunking import get_chunk_dir, list_existing_chunks, select_recall_chunks
    from .scm_grid import SCMGrid
    from .storage import _get_data_dir, recall_memory

    d = _get_data_dir(data_dir)
    scm = SCMGrid.load_or_create(d)

    # Step 1: corpus recall (sacred, unchanged)
    corpus_results = recall_memory(
        text,
        top_k=top_k * 2,  # over-fetch so re-ranking has room
        data_dir=data_dir,
        chunk=chunk,
        encoder=encoder,
        use_embedding=use_embedding,
    )

    if not corpus_results:
        return [], SILENT, scm.openness()

    # Step 2: build query attractors for interference scoring
    from .dynamics import evolve_and_interpret, evolve_with_params
    from .rotation import _get_frame_fn

    frame_fn = _get_frame_fn(use_embedding, encoder=encoder)
    query_frame = frame_fn(text)

    q_corpus = evolve_and_interpret(query_frame)["attractor"]
    q_experiential = evolve_with_params(
        query_frame, EXPERIENTIAL_MAX_PUSH, EXPERIENTIAL_SLOPE_FLOW
    )["attractor"]

    # Step 3: re-score each corpus hit through interference
    scored = []
    for hit in corpus_results:
        hex_key = hit.get("hex_key", "")
        hit_chunk = hit.get("chunk", "general")
        chunk_dir = d / "chunks" / hit_chunk

        # Load corpus attractor
        corpus_path = chunk_dir / "attractors" / f"{hex_key}.npy"
        if not corpus_path.exists():
            scored.append((hit.get("similarity", 0.0), ABSORBED, hit, None, None))
            continue
        stored_corpus = np.load(corpus_path)

        # Look up experiential counterpart (may not exist)
        exp_path = chunk_dir / "experiential" / f"{hex_key}.npy"
        stored_exp = np.load(exp_path) if exp_path.exists() else None

        score, state, ir = interference_score(
            q_corpus, q_experiential, stored_corpus, stored_exp, scm.grid
        )
        hit["interference_score"] = round(score, 4)
        hit["interference_state"] = state
        hit["grounded_frac"] = round(ir.grounded_fraction, 3)
        hit["absorbed_frac"] = round(ir.absorbed_fraction, 3)
        hit["unconsolidated_frac"] = round(ir.unconsolidated_fraction, 3)
        hit["contested_frac"] = round(ir.contested_fraction, 3)
        hit["signal_strength"] = round(ir.signal_strength, 3)
        scored.append((score, state, hit, stored_corpus, stored_exp))

    # Step 4: re-rank by interference score, take top_k
    scored.sort(key=lambda x: -x[0])
    results = [item[2] for item in scored[:top_k]]

    # Dominant state across top results
    state_counts: dict[str, int] = {}
    for _, st, _, _, _ in scored[:top_k]:
        state_counts[st] = state_counts.get(st, 0) + 1
    dominant = max(state_counts, key=state_counts.get) if state_counts else SILENT

    # Step 5: recall-driven SCM feedback from top result
    if scored:
        top_score, _, _, top_corpus, top_exp = scored[0]
        if top_corpus is not None:
            top_exp_safe = top_exp if top_exp is not None else np.zeros_like(top_corpus)
            scm.update_from_recall(top_corpus, top_exp_safe, top_score)
            scm.save()

    return results, dominant, scm.openness()


@dataclass
class ConsistencyResult:
    """Result of the self-consistency feedback check."""

    consistent: bool
    similarity: float
    classification: str  # "SYNTHESIS", "NOVEL", "HALLUCINATION"
    cells_updated: int


def self_consistency_check(
    output_text: str,
    original_corpus_att: np.ndarray,
    original_experiential_att: np.ndarray | None,
    scm: "SCMGrid",
    encoder: str = "blended",
) -> ConsistencyResult:
    """Re-encode decoder output, re-evolve in corpus grid, check basin convergence.

    If the re-evolved attractor converges near the original corpus basin,
    the output is self-consistent → open SCM gaps (reinforce).
    If it diverges → close SCM gaps (restrict hallucination).

    This is the ONLY function that writes to the SCM grid.
    """
    from scipy.stats import pearsonr

    from .dynamics import evolve_with_params
    from .rotation import _get_frame_fn
    from .scm_grid import SCMGrid

    # 1. Re-encode the decoder output
    frame_fn = _get_frame_fn(encoder=encoder)
    frame = frame_fn(output_text)

    # 2. Re-evolve under corpus rules
    result = evolve_with_params(frame, CORPUS_MAX_PUSH, CORPUS_SLOPE_FLOW)
    re_attractor = result["attractor"]

    # 3. Check convergence to original basin
    corr, _ = pearsonr(re_attractor.flatten(), original_corpus_att.flatten())
    sim = float(corr) if not np.isnan(corr) else 0.0
    consistent = sim >= HALLUCINATION_THRESHOLD

    # 4. Classification
    if result["state"] != "CONVERGED":
        classification = "HALLUCINATION"
        consistent = False
    elif consistent:
        classification = "SYNTHESIS"
    else:
        classification = "NOVEL"

    # 5. Build update mask: cells where both content grids had peaks
    c_peak = np.abs(original_corpus_att) > INTERFERENCE_PEAK_THRESHOLD
    if original_experiential_att is not None:
        e_peak = np.abs(original_experiential_att) > INTERFERENCE_PEAK_THRESHOLD
        peak_mask = c_peak & e_peak
    else:
        # No experiential data — update only at corpus peak locations
        peak_mask = c_peak

    # 6. Update SCM: -1 opens (consistent), +1 closes (inconsistent)
    direction = -1.0 if consistent else 1.0
    scm.update(peak_mask, np.full(scm.grid.shape, direction))

    return ConsistencyResult(
        consistent=consistent,
        similarity=sim,
        classification=classification,
        cells_updated=int(peak_mask.sum()),
    )

"""Characterize the per-recall → cumulative-substrate wiring of recall_with_interference.

These tests probe the state of CANON §3.6.4 propagation as of 2026-05-18:
whether the Who-axis correctness restored in `interference_score`
(commit 39fb8fce, weighted Pearson) actually reaches the two cumulative
substrate channels that compose "the substrate change":

  1. SCM grid mutation (interference.py:343-348 via SCMGrid.update_from_recall)
  2. Per-basin T-clock (recall_api._apply_recall_learning → t_metadata.update_t_stability)

The tests do not modify production code. They document the wiring; any
fix is a separate session.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import store_test_memory
from wheeler_memory.interference import recall_with_interference
from wheeler_memory.scm_grid import SCMGrid
from wheeler_memory.storage import _load_index


def _read_index_metadata(data_dir, chunk: str, hex_key: str) -> dict:
    chunk_dir = data_dir / "chunks" / chunk
    return _load_index(chunk_dir).get(hex_key, {}).get("metadata", {})


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CANON §3.6.4: recall_with_interference writes the SCM substrate "
        "but does not advance per-basin T-clock metadata. T-clock updates "
        "are only triggered via recall_api._apply_recall_learning (i.e. "
        "recognize(..., learning_enabled=True)). Test exists to flag the "
        "gap; remove xfail once T-clock wiring lands."
    ),
)
def test_recall_with_interference_does_not_advance_t_clock(tmp_data_dir):
    """N driving recalls through `recall_with_interference` must advance the
    per-basin T-clock if Who-axis correctness is to propagate to T-gated
    cumulative state. They currently do not.
    """
    key = store_test_memory("alpha bravo charlie", tmp_data_dir)

    meta_before = _read_index_metadata(tmp_data_dir, "general", key)
    t_recall_before = meta_before.get("t_recall_count", 0)
    t_stability_before = meta_before.get("t_stability")

    for _ in range(8):
        recall_with_interference(
            "alpha bravo charlie", top_k=3, data_dir=tmp_data_dir
        )

    meta_after = _read_index_metadata(tmp_data_dir, "general", key)
    t_recall_after = meta_after.get("t_recall_count", 0)
    t_stability_after = meta_after.get("t_stability")

    assert t_recall_after > t_recall_before, (
        "T-clock recall count did not advance under recall_with_interference."
    )
    if t_stability_before is not None and t_stability_after is not None:
        assert t_stability_after != pytest.approx(t_stability_before)


def test_scm_substrate_accumulates_under_interference_recall(tmp_data_dir):
    """Positive control: SCM grid (the only currently-wired substrate channel)
    *does* accumulate change across repeated recall_with_interference calls,
    given a non-trivial initial SCM and stored experiential counterpart.
    """
    key = store_test_memory("alpha bravo charlie delta echo", tmp_data_dir)

    chunk_dir = tmp_data_dir / "chunks" / "general"
    corpus_path = chunk_dir / "attractors" / f"{key}.npy"
    stored_corpus = np.load(corpus_path)

    exp_dir = chunk_dir / "experiential"
    exp_dir.mkdir(parents=True, exist_ok=True)
    np.save(exp_dir / f"{key}.npy", stored_corpus)

    rng = np.random.default_rng(0)
    seed_mask = rng.random((64, 64)) < 0.4
    scm = SCMGrid.load_or_create(tmp_data_dir)
    scm.grid[seed_mask] = 0.5
    scm.save()

    snap_before = scm.grid.copy()

    for _ in range(5):
        recall_with_interference(
            "alpha bravo charlie delta echo", top_k=3, data_dir=tmp_data_dir
        )

    scm_after = SCMGrid.load_or_create(tmp_data_dir).grid
    total_change = float(np.abs(scm_after - snap_before).sum())

    assert total_change > 0.0, (
        "SCM did not change across 5 recall_with_interference calls — "
        "substrate wiring may be broken even on the SCM channel."
    )


def test_who_axis_propagates_to_cumulative_substrate(tmp_path):
    """Two arms, identical stored content, mirrored SCM spatial patterns with
    equal mean openness. If the Who-axis is honored end-to-end, the two arms
    should accumulate *different* magnitudes of SCM change, because the
    weighted Pearson sees different spatial regions and therefore returns
    different κ → different `advantage` → different ΔM under update_from_recall.

    Failure mode this guards against (§3.6.4 root collapse): per-tick Who-axis
    correct, but κ becomes invariant to SCM spatial pattern by the time it
    reaches update_from_recall — both arms accumulate equal magnitude despite
    mirrored topologies. If this assertion fails, convert to xfail with a note
    that *both* substrate channels (SCM + T-clock) suffer scalar collapse.
    """
    arm_a = tmp_path / "arm_a"
    arm_b = tmp_path / "arm_b"

    text = "alpha bravo charlie delta echo foxtrot golf"

    for d in (arm_a, arm_b):
        d.mkdir()
        key = store_test_memory(text, d)
        chunk_dir = d / "chunks" / "general"
        stored_corpus = np.load(chunk_dir / "attractors" / f"{key}.npy")
        exp_dir = chunk_dir / "experiential"
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.save(exp_dir / f"{key}.npy", stored_corpus)

    scm_a = SCMGrid.load_or_create(arm_a)
    scm_a.grid[32:, :] = 1.0
    scm_a.save()
    snap_a = scm_a.grid.copy()

    scm_b = SCMGrid.load_or_create(arm_b)
    scm_b.grid[:32, :] = 1.0
    scm_b.save()
    snap_b = scm_b.grid.copy()

    assert (1.0 - np.abs(snap_a)).mean() == pytest.approx(
        (1.0 - np.abs(snap_b)).mean()
    ), "Initial mirrored SCMs must have identical mean openness."

    for _ in range(8):
        recall_with_interference(text, top_k=3, data_dir=arm_a)
        recall_with_interference(text, top_k=3, data_dir=arm_b)

    scm_a_after = SCMGrid.load_or_create(arm_a).grid
    scm_b_after = SCMGrid.load_or_create(arm_b).grid

    delta_a = float(np.abs(scm_a_after - snap_a).sum())
    delta_b = float(np.abs(scm_b_after - snap_b).sum())

    pattern_diff = float(
        np.abs((scm_a_after - snap_a) - (scm_b_after - snap_b)).sum()
    )

    assert pattern_diff > 0.0, (
        f"Mirrored SCMs produced IDENTICAL spatial-delta patterns "
        f"(pattern_diff={pattern_diff:.6f}). §3.6.4 root collapse: "
        f"Who-axis did not propagate from per-tick interference_score "
        f"to cumulative SCM substrate. delta_a={delta_a:.6f} "
        f"delta_b={delta_b:.6f}."
    )

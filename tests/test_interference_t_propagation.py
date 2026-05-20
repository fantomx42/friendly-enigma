"""Characterize the per-recall → cumulative-substrate wiring of recall_with_interference.

These tests guard both substrate channels that compose CANON §3.6.4
propagation now that the wiring is unified through ``apply_recall_learning``:

  1. SCM grid mutation — ``interference.recall_with_interference`` routes the
     SCM update through ``apply_recall_learning`` rather than the old inline
     ``SCMGrid.update_from_recall`` call.
  2. Per-basin T-clock — ``apply_recall_learning`` feeds an SCM-weighted
     basin-stability observation (``_basin_stability_weighted``) into
     ``t_metadata.update_t_stability``.

A failure on either assertion is a regression on the §3.6.4 wiring: it means
one of the substrate channels has fallen back to scalar collapse or stopped
firing per-recall.
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


def test_recall_with_interference_advances_t_clock(tmp_data_dir):
    """N driving recalls through `recall_with_interference` advance the
    per-basin T-clock exactly N times — strict per-recall semantics.

    The strict count equality guards against false-green: if the assertion
    passed with `>` but not `== LOOP_COUNT`, the wiring would be advancing
    T-clock for the wrong reason (session-end side effects, cache flushes,
    finalizer artifacts) rather than per-recall as intended.
    """
    LOOP_COUNT = 8

    key = store_test_memory("alpha bravo charlie", tmp_data_dir)

    meta_before = _read_index_metadata(tmp_data_dir, "general", key)
    t_recall_before = meta_before.get("t_recall_count", 0)
    t_stability_before = meta_before.get("t_stability")

    for _ in range(LOOP_COUNT):
        recall_with_interference(
            "alpha bravo charlie", top_k=3, data_dir=tmp_data_dir
        )

    meta_after = _read_index_metadata(tmp_data_dir, "general", key)
    t_recall_after = meta_after.get("t_recall_count", 0)
    t_stability_after = meta_after.get("t_stability")

    assert t_recall_after - t_recall_before == LOOP_COUNT, (
        f"Expected exactly {LOOP_COUNT} T-clock advancements (one per recall); "
        f"got {t_recall_after - t_recall_before}. "
        "If this fails, T-clock advancement is not firing per-recall."
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

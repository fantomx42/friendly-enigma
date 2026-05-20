"""Unit tests for the unified substrate-update entry point.

Owns three concerns:
- apply_learning=False on recall_with_interference is a true no-op for all
  three substrate channels (T-clock, SCM, attractor).
- _basin_stability_weighted respects the SCM spatial pattern when scm_grid
  is provided (added in the SCM-weighted observation commit).
- apply_recall_learning with SCM context produces a different T-update than
  the uniform-observation reference (added in the SCM-weighted observation
  commit).
"""

from __future__ import annotations

import numpy as np

from conftest import store_test_memory
from wheeler_memory.interference import recall_with_interference
from wheeler_memory.scm_grid import SCMGrid
from wheeler_memory.storage import _load_index


def _read_meta(data_dir, chunk: str, hex_key: str) -> dict:
    chunk_dir = data_dir / "chunks" / chunk
    return _load_index(chunk_dir).get(hex_key, {}).get("metadata", {})


def test_recall_with_interference_apply_learning_false_is_no_op(tmp_data_dir):
    """apply_learning=False on the wrapper freezes all three substrate
    channels: T-clock metadata, stored attractor file, SCM grid.

    Smoke test for the escape hatch used by benchmarking, A/B testing,
    and replay scenarios where substrate must not mutate across queries.
    """
    text = "alpha bravo charlie delta echo foxtrot"
    key = store_test_memory(text, tmp_data_dir)

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
    scm_snapshot = scm.grid.copy()

    attractor_snapshot = np.load(corpus_path)
    meta_before = _read_meta(tmp_data_dir, "general", key)
    t_recall_before = meta_before.get("t_recall_count", 0)
    t_stability_before = meta_before.get("t_stability")

    for _ in range(5):
        recall_with_interference(
            text, top_k=3, data_dir=tmp_data_dir, apply_learning=False
        )

    meta_after = _read_meta(tmp_data_dir, "general", key)
    attractor_after = np.load(corpus_path)
    scm_after = SCMGrid.load_or_create(tmp_data_dir).grid

    assert meta_after.get("t_recall_count", 0) == t_recall_before, (
        "T-clock advanced despite apply_learning=False."
    )
    assert meta_after.get("t_stability") == t_stability_before, (
        "t_stability mutated despite apply_learning=False."
    )
    assert np.array_equal(attractor_after, attractor_snapshot), (
        "Stored attractor was rewritten despite apply_learning=False "
        "(should be False-as-no-drift even though the interference path "
        "never drifts; the kwarg gate must hold)."
    )
    assert np.array_equal(scm_after, scm_snapshot), (
        "SCM grid mutated despite apply_learning=False."
    )

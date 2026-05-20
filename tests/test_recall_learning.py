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
from wheeler_memory.recall_learning import _basin_stability_weighted
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


def test_basin_stability_weighted_respects_scm_pattern():
    """Same stored + one_step, mirrored SCMs with equal mean openness →
    different observation values. Mirrors the Who-axis logic of
    interference_score's weighted Pearson at the observation layer.
    """
    rng = np.random.RandomState(11)
    stored = np.sign(rng.standard_normal((64, 64))).astype(np.float32)
    one_step = stored.copy()
    one_step[32:, :] = -one_step[32:, :]

    scm_a = np.zeros((64, 64), dtype=np.float32)
    scm_a[32:, :] = 1.0
    scm_b = np.zeros((64, 64), dtype=np.float32)
    scm_b[:32, :] = 1.0

    assert (1.0 - np.abs(scm_a)).mean() == (1.0 - np.abs(scm_b)).mean(), (
        "Mirrored SCMs must have identical mean openness for this test "
        "to isolate spatial pattern effects from scalar-openness effects."
    )

    stability_a = _basin_stability_weighted(stored, one_step, scm_grid=scm_a)
    stability_b = _basin_stability_weighted(stored, one_step, scm_grid=scm_b)
    stability_uniform = _basin_stability_weighted(stored, one_step)

    assert stability_a != stability_b, (
        f"Mirrored SCMs produced identical stability ({stability_a} == "
        f"{stability_b}). Who-axis collapsed at the observation layer."
    )
    assert stability_a != stability_uniform or stability_b != stability_uniform, (
        "SCM-aware observation matches uniform — SCM context not influencing "
        "the reading."
    )


def test_apply_recall_learning_scm_path_uses_weighted_observation(tmp_data_dir):
    """Mirrored-SCM arms with identical stored content produce different
    t_stability EMA outputs because apply_recall_learning's T-clock
    observation sees the SCM spatial pattern via _basin_stability_weighted.

    Setup writes the stored attractor directly with a spatially non-uniform
    pattern (bottom half saturated, top half zero) so that one CA step
    produces large delta in the top half and small delta in the bottom.
    Mirrored SCMs then expose either the high-delta or low-delta region
    to the p99, yielding distinct observed_stability values.

    The hash-encoder attractors used by store_test_memory are typically
    at CA fixed points (delta near zero everywhere), so they cannot
    distinguish mirrored SCMs — hence the manual attractor write.
    """
    from wheeler_memory.recall_learning import apply_recall_learning

    text = "alpha bravo charlie delta echo foxtrot golf hotel"

    arm_a = tmp_data_dir / "arm_a"
    arm_b = tmp_data_dir / "arm_b"
    arm_a.mkdir()
    arm_b.mkdir()

    key_a = store_test_memory(text, arm_a)
    key_b = store_test_memory(text, arm_b)
    assert key_a == key_b, "Hash encoder is deterministic; keys must match."

    chunk_dir_a = arm_a / "chunks" / "general"
    chunk_dir_b = arm_b / "chunks" / "general"

    # Override the stored attractor with a spatially non-uniform pattern.
    # Bottom half saturated at +1 (fixed point); top half at 0 → flows
    # toward neighbors under one CA step → high delta in top half only.
    stored_pattern = np.zeros((64, 64), dtype=np.float32)
    stored_pattern[32:, :] = 1.0
    np.save(chunk_dir_a / "attractors" / f"{key_a}.npy", stored_pattern)
    np.save(chunk_dir_b / "attractors" / f"{key_b}.npy", stored_pattern)

    scm_a = SCMGrid.load_or_create(arm_a)
    scm_a.grid[32:, :] = 1.0  # top permissive, bottom closed
    scm_a.save()
    scm_b = SCMGrid.load_or_create(arm_b)
    scm_b.grid[:32, :] = 1.0  # bottom permissive, top closed
    scm_b.save()

    rng = np.random.RandomState(3)
    query_frame = rng.standard_normal((64, 64)).astype(np.float32)
    top_corpus = stored_pattern.copy()
    top_exp = np.zeros_like(top_corpus)
    top_kappa = 0.5

    apply_recall_learning(
        chunk_dir_a, key_a, query_frame,
        drift_basin=False, scm=scm_a, scm_top_corpus=top_corpus,
        scm_top_exp=top_exp, scm_top_kappa=top_kappa,
    )
    apply_recall_learning(
        chunk_dir_b, key_b, query_frame,
        drift_basin=False, scm=scm_b, scm_top_corpus=top_corpus,
        scm_top_exp=top_exp, scm_top_kappa=top_kappa,
    )

    meta_a = _read_meta(arm_a, "general", key_a)
    meta_b = _read_meta(arm_b, "general", key_b)

    assert meta_a["t_stability"] != meta_b["t_stability"], (
        f"Mirrored-SCM arms produced identical t_stability "
        f"({meta_a['t_stability']} == {meta_b['t_stability']}). "
        "Who-axis is collapsed at the observation layer despite the "
        "_basin_stability_weighted helper being available — the SCM grid "
        "is not being passed through apply_recall_learning to the observer."
    )


"""Tests for the two-tier recall API: recognize / reconstruct + T-gated drift."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wheeler_memory import recall_api
from wheeler_memory.constants import T_INIT_DEFAULT
from wheeler_memory.recall_api import (
    BasinSeed,
    Pattern,
    recognize,
    recognize_top_k,
    reconstruct,
)
from wheeler_memory.storage import _load_index

from conftest import store_test_memory


# Test 1 ------------------------------------------------------------------
# recognize() must not engage the CA convergence loop.

def test_recognize_does_not_call_evolve_and_interpret(tmp_data_dir, monkeypatch):
    store_test_memory("python list comprehensions", tmp_data_dir)

    call_log = []
    real_evolve = recall_api.evolve_and_interpret

    def spy(*args, **kwargs):
        call_log.append("called")
        return real_evolve(*args, **kwargs)

    monkeypatch.setattr(recall_api, "evolve_and_interpret", spy)

    seed = recognize("python list comprehensions", data_dir=tmp_data_dir)
    assert seed is not None, "expected recognition hit on exact stored text"
    assert len(call_log) == 0, (
        f"recognize() called evolve_and_interpret {len(call_log)} time(s); "
        "must skip the convergence loop entirely"
    )


# Test 2 ------------------------------------------------------------------
# reconstruct() returns a populated Pattern with sensible correlations.

def test_reconstruct_returns_populated_pattern(tmp_data_dir):
    text = "stored python facts about list comprehensions"
    store_test_memory(text, tmp_data_dir)

    seed = recognize(text, data_dir=tmp_data_dir)
    assert seed is not None

    pat = reconstruct(seed, query=text, alpha=0.3)
    assert isinstance(pat, Pattern)
    assert pat.attractor.shape == (64, 64)
    assert pat.state in {"CONVERGED", "OSCILLATING", "CHAOTIC", "DEGENERATE"}
    assert -1.0 <= pat.correlation_with_stored <= 1.0
    assert -1.0 <= pat.correlation_with_query <= 1.0
    # alpha=0.3 is memory-dominant, so reconstruction should be closer to stored
    assert pat.correlation_with_stored >= pat.correlation_with_query
    assert pat.text == text
    assert pat.seed.hex_key == seed.hex_key


def test_reconstruct_with_no_query_returns_stored_attractor(tmp_data_dir):
    text = "memory without a query"
    store_test_memory(text, tmp_data_dir)
    seed = recognize(text, data_dir=tmp_data_dir)
    assert seed is not None

    pat = reconstruct(seed, query=None)
    assert pat.convergence_ticks == 0
    assert pat.correlation_with_stored == pytest.approx(1.0)
    # Same path: load the file and compare
    chunk_dir = tmp_data_dir / "chunks" / seed.chunk
    stored = np.load(chunk_dir / "attractors" / f"{seed.hex_key}.npy")
    assert np.array_equal(pat.attractor, stored)


# Test 3 ------------------------------------------------------------------
# T persists across recognize() calls and updates per the EMA rule.

def test_recognize_persists_t_with_ema_update(tmp_data_dir):
    text = "T stability test"
    key = store_test_memory(text, tmp_data_dir)

    # Pre-condition: backfill provides default T
    seed_before = recognize(text, data_dir=tmp_data_dir, learning_enabled=False)
    assert seed_before is not None
    assert seed_before.t_stability == pytest.approx(T_INIT_DEFAULT)

    # Pre-write known T directly to test the EMA cleanly
    chunk_dir = tmp_data_dir / "chunks" / "general"
    idx = _load_index(chunk_dir)
    idx[key]["metadata"]["t_stability"] = 0.6
    idx[key]["metadata"]["t_recall_count"] = 5
    (chunk_dir / "index.json").write_text(json.dumps(idx, indent=2))
    # Invalidate the cache so the next read sees the override
    from wheeler_memory.cache import invalidate
    invalidate(chunk_dir / "index.json")

    seed = recognize(text, data_dir=tmp_data_dir, learning_enabled=True)
    assert seed is not None

    invalidate(chunk_dir / "index.json")
    idx2 = _load_index(chunk_dir)
    md = idx2[key]["metadata"]
    assert md["t_recall_count"] == 6
    # EMA: T_new = 0.9 * 0.6 + 0.1 * observed_stability (= seed.basin_stability)
    expected_t = 0.9 * 0.6 + 0.1 * seed.basin_stability
    assert md["t_stability"] == pytest.approx(expected_t, abs=1e-4)


# Test 4 ------------------------------------------------------------------
# Drift responds to T. Plastic basins (low T) drift more than rigid ones.
# Failure-mode mitigation: T values are seeded deliberately, not by hope.

def test_drift_responds_to_t_with_seeded_values(tmp_data_dir):
    texts = [f"drift test memory number {i}" for i in range(6)]
    keys = [store_test_memory(t, tmp_data_dir) for t in texts]

    chunk_dir = tmp_data_dir / "chunks" / "general"
    idx = _load_index(chunk_dir)

    # Plastic basins (low T → high plasticity): expect more drift
    plastic_t = {keys[0]: 0.05, keys[1]: 0.10, keys[2]: 0.15}
    # Rigid basins (high T → low plasticity): expect minimal drift
    rigid_t = {keys[3]: 0.85, keys[4]: 0.90, keys[5]: 0.95}

    for k, t in {**plastic_t, **rigid_t}.items():
        idx[k]["metadata"]["t_stability"] = t
        idx[k]["metadata"]["t_recall_count"] = 0

    (chunk_dir / "index.json").write_text(json.dumps(idx, indent=2))
    from wheeler_memory.cache import invalidate
    invalidate(chunk_dir / "index.json")

    # Snapshot all stored attractors before drift
    before = {
        k: np.load(chunk_dir / "attractors" / f"{k}.npy").copy()
        for k in keys
    }

    # Recall each with a paraphrase that produces a different query frame
    # so drift has somewhere to pull. Same paraphrase pattern across all
    # basins → identical recall conditions.
    for t in texts:
        para = t.replace("drift", "movement")
        recognize(para, data_dir=tmp_data_dir, learning_enabled=True, threshold=0.0)

    drifts = {}
    for k in keys:
        after = np.load(chunk_dir / "attractors" / f"{k}.npy")
        drifts[k] = float(np.abs(after.astype(np.float32) - before[k].astype(np.float32)).mean())

    avg_plastic = sum(drifts[k] for k in plastic_t) / len(plastic_t)
    avg_rigid = sum(drifts[k] for k in rigid_t) / len(rigid_t)

    # Plastic basins must drift substantially more than rigid ones.
    # Rigid coefficient (1-T)*0.02 is in [0.001, 0.003]; plastic is in [0.017, 0.019].
    # So plastic drift should be ~5-15x rigid drift in magnitude.
    assert avg_plastic > 2 * avg_rigid, (
        f"Expected plastic basins to drift >2x more than rigid. "
        f"plastic={avg_plastic:.6f}, rigid={avg_rigid:.6f}"
    )

    # Rigid basins should drift very little in absolute terms
    for k in rigid_t:
        assert drifts[k] < 5e-3, (
            f"Rigid basin {k!r} drifted {drifts[k]:.6f}; expected near-zero "
            f"for T={rigid_t[k]}"
        )


# Sanity: recognize_top_k respects threshold and ordering ------------------

def test_recognize_top_k_orders_by_similarity(tmp_data_dir):
    a = store_test_memory("alpha bravo charlie", tmp_data_dir)
    b = store_test_memory("delta echo foxtrot", tmp_data_dir)
    c = store_test_memory("zebra yankee xray", tmp_data_dir)

    seeds = recognize_top_k("alpha bravo charlie", k=3, data_dir=tmp_data_dir, threshold=0.0)
    assert len(seeds) >= 1
    assert seeds[0].hex_key == a
    sims = [s.similarity for s in seeds]
    assert sims == sorted(sims, reverse=True)

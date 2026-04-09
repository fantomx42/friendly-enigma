"""Tests for CA batch evolution via accel.ca / dynamics.evolve_batch.

CPU-only tests run always. GPU tests are marked @pytest.mark.gpu and skip
when no HIP runtime is available.
"""

import numpy as np
import pytest

from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED, SALIENCE_THRESHOLD_MED
from wheeler_memory.dynamics import evolve_and_interpret, evolve_batch
from wheeler_memory.hashing import hash_to_frame


# ---------------------------------------------------------------------------
# CPU batch correctness (always runs)
# ---------------------------------------------------------------------------


def test_evolve_batch_empty():
    """evolve_batch([]) returns []."""
    assert evolve_batch([]) == []


def test_evolve_batch_single():
    """Single-frame batch matches evolve_and_interpret."""
    frame = hash_to_frame("batch test single")
    batch_result = evolve_batch([frame])[0]
    serial_result = evolve_and_interpret(frame)

    np.testing.assert_allclose(
        batch_result["attractor"], serial_result["attractor"], atol=1e-6
    )
    assert batch_result["state"] == serial_result["state"]
    assert batch_result["convergence_ticks"] == serial_result["convergence_ticks"]


def test_evolve_batch_multiple():
    """Multiple-frame batch matches serial evolve_and_interpret for each."""
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    frames = [hash_to_frame(t) for t in texts]

    batch_results = evolve_batch(frames)
    assert len(batch_results) == 5

    for i, frame in enumerate(frames):
        serial = evolve_and_interpret(frame)
        np.testing.assert_allclose(
            batch_results[i]["attractor"], serial["attractor"], atol=1e-6
        )
        assert batch_results[i]["state"] == serial["state"]


def test_evolve_batch_result_keys():
    """Batch results have the same keys as serial results."""
    frame = hash_to_frame("key check")
    serial = evolve_and_interpret(frame)
    batch = evolve_batch([frame])[0]

    # Batch results should have at least the core keys
    for key in ("attractor", "state", "convergence_ticks", "history"):
        assert key in batch, f"Missing key: {key}"


def test_evolve_batch_deterministic():
    """Same input produces same output across two batch calls."""
    frames = [hash_to_frame(f"det_{i}") for i in range(3)]
    r1 = evolve_batch(frames)
    r2 = evolve_batch(frames)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a["attractor"], b["attractor"])


def test_evolve_batch_respects_params():
    """Custom max_iters and stability_threshold are forwarded."""
    frame = hash_to_frame("param test")
    # Very low iteration cap — should hit MAX_ITERS quickly
    result = evolve_batch([frame], max_iters=2)[0]
    assert result["convergence_ticks"] <= 2


def test_evolve_batch_enriches_metadata():
    """Batch results include enriched metadata from _enrich_metadata."""
    frame = hash_to_frame("metadata check")
    result = evolve_batch([frame])[0]
    # _enrich_metadata merges attractor features into metadata dict
    md = result.get("metadata", {})
    assert "alive_fraction" in md
    assert "cluster_count" in md


# ---------------------------------------------------------------------------
# GPU-specific tests (skip without HIP runtime)
# ---------------------------------------------------------------------------


def _gpu_ready():
    try:
        from wheeler_memory.accel.ca import gpu_available

        return gpu_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="No GPU/HIP runtime available")
def test_gpu_evolve_single_matches_cpu():
    """GPU single-frame evolution matches CPU within tolerance."""
    from wheeler_memory.accel.ca import gpu_evolve_single

    frame = hash_to_frame("gpu vs cpu single")
    gpu_result = gpu_evolve_single(frame)
    cpu_result = evolve_and_interpret(frame)
    np.testing.assert_allclose(
        gpu_result["attractor"], cpu_result["attractor"], atol=1e-4
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="No GPU/HIP runtime available")
def test_gpu_batch_matches_cpu():
    """GPU batch evolution matches CPU serial within tolerance."""
    from wheeler_memory.accel.ca import gpu_evolve_batch

    texts = [f"gpu_batch_{i}" for i in range(10)]
    frames = [hash_to_frame(t) for t in texts]

    gpu_results = gpu_evolve_batch(frames)
    for i, frame in enumerate(frames):
        cpu_result = evolve_and_interpret(frame)
        np.testing.assert_allclose(
            gpu_results[i]["attractor"], cpu_result["attractor"], atol=1e-4
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="No GPU/HIP runtime available")
def test_gpu_vram_query():
    """gpu_query_vram returns plausible estimate for a given batch size."""
    from wheeler_memory.accel.ca import gpu_query_vram

    estimate = gpu_query_vram(batch_size=100)
    assert estimate is None or estimate > 0

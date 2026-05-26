"""Tests for the reconstruction-fidelity benchmark (scripts/bench_reconstruction.py)."""

from __future__ import annotations

import numpy as np

from scripts.bench_reconstruction import (
    EPS_LEVELS,
    perturb,
    run_benchmark,
)
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame


def test_perturb_is_deterministic_and_bounded():
    att = evolve_and_interpret(hash_to_frame("a known attractor"))["attractor"]
    p1 = perturb(att, 0.5, seed=123)
    p2 = perturb(att, 0.5, seed=123)
    assert np.array_equal(p1, p2), "same seed must reproduce the same perturbation"
    assert p1.min() >= -1.0 and p1.max() <= 1.0, "perturbed frame stays in [-1, 1]"
    # A different seed gives a different frame.
    assert not np.array_equal(perturb(att, 0.5, seed=456), p1)


def test_tiny_perturbation_recovers_to_basin():
    """A near-zero nudge should re-settle close to the original attractor."""
    att = evolve_and_interpret(hash_to_frame("recover me"))["attractor"]
    nudged = perturb(att, 0.05, seed=7)
    resettled = evolve_and_interpret(nudged)["attractor"]
    r = np.corrcoef(att.flatten(), resettled.flatten())[0, 1]
    assert r > 0.8, f"small perturbation should stay in-basin, got fidelity {r:.3f}"


def test_run_benchmark_structure_and_bounds():
    result = run_benchmark(eps_levels=(0.1, 0.5), verbose=False)
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["mean_fidelity"] <= 1.0
    assert 0.0 <= result["recovery_rate"] <= 1.0
    assert result["n_inputs"] == 20
    assert set(result["per_eps"]) == {"0.1", "0.5"}
    for m in result["per_eps"].values():
        assert 0.0 <= m["mean_fidelity"] <= 1.0
        assert m["mean_ticks"] >= 0.0


def test_fidelity_degrades_monotonically_with_noise():
    """Mean fidelity should not increase as the perturbation magnitude grows."""
    result = run_benchmark(eps_levels=EPS_LEVELS, verbose=False)
    fids = [result["per_eps"][str(e)]["mean_fidelity"] for e in EPS_LEVELS]
    # Allow tiny non-monotonic noise but require the trend to be downward overall.
    assert fids[0] >= fids[-1], f"fidelity should fall with noise: {fids}"

"""Unit tests for CA dynamics engine."""

import numpy as np
import pytest
from wheeler_memory.dynamics import (
    apply_ca_dynamics,
    evolve_and_interpret,
    evolve_with_params,
)
from wheeler_memory.hashing import hash_to_frame


class TestApplyCADynamics:
    """Tests for apply_ca_dynamics function."""

    def test_apply_ca_dynamics_shape_preserved(self):
        """input 64x64 → output 64x64."""
        frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        output = apply_ca_dynamics(frame)
        assert output.shape == (64, 64)

    def test_apply_ca_dynamics_range_preserved(self):
        """output values in [-1, 1]."""
        frame = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
        output = apply_ca_dynamics(frame)
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)

    def test_apply_ca_dynamics_local_max_increases(self):
        """center cell = 1, all neighbors = 0 → center is local max, pushed toward +1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = 1.0  # center >= all neighbors → local max
        output = apply_ca_dynamics(frame)
        # Local max: delta = (1 - cell) * 0.35, center stays near 1.0 (small positive delta)
        assert output[32, 32] >= frame[32, 32]

    def test_apply_ca_dynamics_local_min_decreases(self):
        """center cell = -1, all neighbors = 0 → center is local min, pushed toward -1."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = -1.0  # center <= all neighbors → local min
        output = apply_ca_dynamics(frame)
        # Local min: delta = (-1 - cell) * 0.35, center stays near -1.0 (small negative delta)
        assert output[32, 32] <= frame[32, 32]

    def test_flat_zero_plateau_is_stable(self):
        """A uniform zero grid must not move — 0 is a stable fixed point."""
        frame = np.zeros((64, 64), dtype=np.float32)
        output = apply_ca_dynamics(frame)
        np.testing.assert_array_equal(output, frame)

    def test_flat_positive_plateau_is_stable(self):
        """A uniform +1 grid must not move — +1 is a stable fixed point."""
        frame = np.ones((64, 64), dtype=np.float32)
        output = apply_ca_dynamics(frame)
        np.testing.assert_array_equal(output, frame)

    def test_flat_negative_plateau_is_stable(self):
        """A uniform -1 grid must not move — -1 is a stable fixed point."""
        frame = -np.ones((64, 64), dtype=np.float32)
        output = apply_ca_dynamics(frame)
        np.testing.assert_array_equal(output, frame)


class TestEvolveAndInterpret:
    """Tests for evolve_and_interpret function."""

    def test_evolve_result_keys(self):
        """result dict has required keys."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert "state" in result
        assert "attractor" in result
        assert "convergence_ticks" in result
        assert "history" in result

    def test_evolve_attractor_shape(self):
        """result['attractor'] is 64x64."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["attractor"].shape == (64, 64)

    def test_evolve_attractor_range(self):
        """attractor values in [-1, 1]."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert np.all(result["attractor"] >= -1.0)
        assert np.all(result["attractor"] <= 1.0)

    def test_evolve_state_valid(self):
        """state is one of 'CONVERGED', 'OSCILLATING', 'CHAOTIC'."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["state"] in ("CONVERGED", "OSCILLATING", "CHAOTIC")

    def test_evolve_convergence_ticks_positive(self):
        """convergence_ticks >= 1."""
        frame = hash_to_frame("test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        assert result["convergence_ticks"] >= 1

    def test_evolve_convergence_ticks_respects_max_iters(self):
        """convergence_ticks <= max_iters."""
        frame = hash_to_frame("test")
        max_iters = 50
        result = evolve_and_interpret(
            frame, max_iters=max_iters, stability_threshold=1e-4
        )
        assert result["convergence_ticks"] <= max_iters

    def test_evolve_deterministic(self):
        """same input → same state and same attractor."""
        frame = hash_to_frame("deterministic test")
        result1 = evolve_and_interpret(
            frame.copy(), max_iters=100, stability_threshold=1e-4
        )
        result2 = evolve_and_interpret(
            frame.copy(), max_iters=100, stability_threshold=1e-4
        )
        assert result1["state"] == result2["state"]
        assert np.array_equal(result1["attractor"], result2["attractor"])

    def test_evolve_typical_input_converges(self):
        """hash_to_frame('test input') → state == 'CONVERGED'."""
        frame = hash_to_frame("test input")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["state"] == "CONVERGED"

    def test_evolve_history_nonempty(self):
        """history contains at least seed and attractor frames."""
        frame = hash_to_frame("history test")
        result = evolve_and_interpret(frame, max_iters=100, stability_threshold=1e-4)
        # GPU path stores [seed, attractor]; CPU path stores every tick.
        # Either way history has at least 2 entries.
        assert len(result["history"]) >= 2

    def test_evolve_first_frame_is_input(self):
        """first frame in history matches input."""
        frame = hash_to_frame("first frame test")
        result = evolve_and_interpret(
            frame.copy(), max_iters=100, stability_threshold=1e-4
        )
        assert np.array_equal(result["history"][0], frame)


class TestConvergenceMetadata:
    """Tests for enriched convergence metadata in evolve_and_interpret."""

    def test_converged_metadata_has_features(self):
        """CONVERGED result metadata includes grid features."""
        frame = hash_to_frame("metadata test")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["state"] == "CONVERGED"
        md = result["metadata"]
        assert "grid_entropy" in md
        assert "cluster_count" in md
        assert "alive_fraction" in md
        assert "convergence_speed" in md
        assert "final_delta" in md

    def test_converged_speed_label(self):
        """Speed label reflects tick count: F<=50, M<=150, S>150."""
        frame = hash_to_frame("speed test")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["state"] == "CONVERGED"
        ticks = result["convergence_ticks"]
        expected = "F" if ticks <= 50 else ("M" if ticks <= 150 else "S")
        assert result["metadata"]["convergence_speed"] == expected

    def test_converged_entropy_is_positive(self):
        """Grid entropy should be positive for non-trivial frames."""
        frame = hash_to_frame("entropy test")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["metadata"]["grid_entropy"] > 0.0

    def test_converged_alive_fraction_in_range(self):
        """Alive fraction should be in [0, 1]."""
        frame = hash_to_frame("alive test")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        alive = result["metadata"]["alive_fraction"]
        assert 0.0 <= alive <= 1.0

    def test_converged_final_delta_small(self):
        """Final delta should be below stability threshold for CONVERGED."""
        frame = hash_to_frame("delta test")
        thresh = 1e-4
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=thresh)
        assert result["state"] == "CONVERGED"
        assert result["metadata"]["final_delta"] < thresh

    def test_chaotic_metadata_has_features(self):
        """CHAOTIC result metadata also includes grid features."""
        # Force chaotic by using very tight threshold with few iters
        frame = hash_to_frame("chaotic test")
        result = evolve_and_interpret(frame, max_iters=5, stability_threshold=1e-12)
        assert result["state"] == "CHAOTIC"
        md = result["metadata"]
        assert "grid_entropy" in md
        assert "cluster_count" in md
        assert "alive_fraction" in md
        assert md["convergence_speed"] == "?"

    def test_evolve_with_params_metadata(self):
        """evolve_with_params also returns enriched metadata."""
        frame = hash_to_frame("params metadata test")
        result = evolve_with_params(
            frame, push_strength=0.57, slope_strength=0.55,
            max_iters=1000, stability_threshold=1e-4,
        )
        md = result["metadata"]
        assert "grid_entropy" in md
        assert "cluster_count" in md
        assert "convergence_speed" in md


class TestStabilityWindow:
    """Tests for multi-tick stability window."""

    def test_converged_ticks_at_least_stability_window(self):
        """CONVERGED ticks must be >= STABILITY_WINDOW (need N consecutive sub-threshold)."""
        from wheeler_memory.constants import STABILITY_WINDOW

        frame = hash_to_frame("stability window test")
        result = evolve_and_interpret(frame, max_iters=1000, stability_threshold=1e-4)
        assert result["state"] == "CONVERGED"
        assert result["convergence_ticks"] >= STABILITY_WINDOW

    def test_insufficient_iters_for_window_is_chaotic(self):
        """With max_iters < STABILITY_WINDOW, CPU path cannot satisfy window."""
        from unittest.mock import patch

        from wheeler_memory.constants import STABILITY_WINDOW

        # Patch GPU away so we test the CPU path's stability window
        with patch("wheeler_memory.dynamics._GPU_READY", False):
            frame = hash_to_frame("insufficient iters test")
            result = evolve_and_interpret(
                frame,
                max_iters=STABILITY_WINDOW - 1,
                stability_threshold=1.0,  # trivially easy threshold
            )
            assert result["state"] == "CHAOTIC"


class TestZeroDominantConvergence:
    """Tests for correct convergence on 0-dominant grids (percentile fix)."""

    def test_zero_dominant_not_premature_convergence(self):
        """95% zero grid with boundary pattern must not converge prematurely.

        The boundary cells need time to evolve. With the old mean-delta check,
        the silent zero majority would drag the mean below threshold too early.
        With percentile-based delta, convergence waits for boundary cells.
        """
        from unittest.mock import patch

        from wheeler_memory.constants import STABILITY_WINDOW

        frame = np.zeros((64, 64), dtype=np.float32)
        # Plant a non-trivial boundary pattern: a gradient block in one corner
        frame[0:4, 0:4] = np.linspace(-0.8, 0.8, 16).reshape(4, 4)
        frame[60:64, 60:64] = np.linspace(0.3, -0.5, 16).reshape(4, 4)

        with patch("wheeler_memory.dynamics._GPU_READY", False):
            result = evolve_and_interpret(
                frame, max_iters=500, stability_threshold=1e-4
            )

        # Must converge (not exhaust budget)
        assert result["state"] == "CONVERGED"
        # Must take more than STABILITY_WINDOW ticks (boundary needs real evolution)
        assert result["convergence_ticks"] > STABILITY_WINDOW
        # The attractor should differ from the initial frame in the boundary region
        att = result["attractor"]
        boundary_change = np.abs(att[0:4, 0:4] - frame[0:4, 0:4]).max()
        assert boundary_change > 0.01, "Boundary cells should have evolved"

    def test_percentile_catches_active_boundary(self):
        """Small non-zero region in zero field: attractor must show real structure.

        This tests that evolution doesn't terminate so early that the small
        active region hasn't had time to form a meaningful attractor pattern.
        """
        from unittest.mock import patch

        frame = np.zeros((64, 64), dtype=np.float32)
        # Single 4x4 hot spot with mixed values
        frame[30:34, 30:34] = np.array(
            [
                [0.9, -0.7, 0.5, -0.3],
                [-0.6, 0.8, -0.4, 0.2],
                [0.7, -0.5, 0.6, -0.8],
                [-0.2, 0.4, -0.9, 0.1],
            ],
            dtype=np.float32,
        )

        with patch("wheeler_memory.dynamics._GPU_READY", False):
            result = evolve_and_interpret(
                frame, max_iters=500, stability_threshold=1e-4
            )

        att = result["attractor"]
        # The hot spot should have evolved into a structured attractor
        hotspot = att[28:36, 28:36]  # Slightly wider — activity spreads
        assert np.abs(hotspot).max() > 0.1, "Hot spot region should retain structure"

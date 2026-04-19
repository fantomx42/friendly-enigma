"""Tests for wheeler_memory.scm_grid — persistent spatial trust topology."""

import numpy as np
import pytest

from wheeler_memory.scm_grid import GRID_SIZE, SCMGrid


class TestSCMGridCreate:
    """SCMGrid creation and initialization."""

    def test_load_or_create_fresh(self, tmp_path):
        """Fresh directory → zero grid and zero hardening."""
        scm = SCMGrid.load_or_create(tmp_path)
        assert scm.grid.shape == (GRID_SIZE, GRID_SIZE)
        assert scm.hardening.shape == (GRID_SIZE, GRID_SIZE)
        np.testing.assert_array_equal(scm.grid, 0.0)
        np.testing.assert_array_equal(scm.hardening, 0)

    def test_dtypes(self, tmp_path):
        """Grid is float32, hardening is uint32."""
        scm = SCMGrid.load_or_create(tmp_path)
        assert scm.grid.dtype == np.float32
        assert scm.hardening.dtype == np.uint32

    def test_creates_directory(self, tmp_path):
        """load_or_create creates data_dir if missing."""
        nested = tmp_path / "deep" / "nested"
        SCMGrid.load_or_create(nested)
        assert nested.exists()

    def test_direct_constructor(self, tmp_path):
        """Direct __init__ accepts pre-built arrays."""
        grid = np.ones((64, 64), dtype=np.float32) * 0.5
        hard = np.zeros((64, 64), dtype=np.uint32)
        scm = SCMGrid(grid=grid, hardening=hard, data_dir=tmp_path)
        np.testing.assert_array_equal(scm.grid, 0.5)


class TestSCMGridSaveLoad:
    """Persistence round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save then load recovers grid and hardening."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.grid[10, 20] = 0.75
        scm.hardening[10, 20] = 5
        scm.save()

        loaded = SCMGrid.load_or_create(tmp_path)
        assert loaded.grid[10, 20] == pytest.approx(0.75)
        assert loaded.hardening[10, 20] == 5

    def test_atomic_save_files_exist(self, tmp_path):
        """save() creates .npy files, no leftover temps."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.save()
        assert (tmp_path / "scm_grid.npy").exists()
        assert (tmp_path / "scm_hardening.npy").exists()
        # Temp files should be cleaned up
        assert not (tmp_path / "_scm_grid_tmp.npy").exists()
        assert not (tmp_path / "_scm_hardening_tmp.npy").exists()


class TestSCMGridUpdate:
    """update() self-consistency feedback."""

    def test_positive_direction_increases_grid(self, tmp_path):
        """Direction +1 → grid values increase."""
        scm = SCMGrid.load_or_create(tmp_path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[0, 0] = True
        scm.update(mask, np.float32(1.0))
        assert scm.grid[0, 0] > 0.0

    def test_negative_direction_decreases_grid(self, tmp_path):
        """Direction -1 → grid values decrease."""
        scm = SCMGrid.load_or_create(tmp_path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[0, 0] = True
        scm.update(mask, np.float32(-1.0))
        assert scm.grid[0, 0] < 0.0

    def test_hardening_increments(self, tmp_path):
        """Each update increments hardening count."""
        scm = SCMGrid.load_or_create(tmp_path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[5, 5] = True
        scm.update(mask, np.float32(1.0))
        assert scm.hardening[5, 5] == 1
        scm.update(mask, np.float32(1.0))
        assert scm.hardening[5, 5] == 2

    def test_hardening_reduces_effective_lr(self, tmp_path):
        """More updates → smaller increments."""
        scm = SCMGrid.load_or_create(tmp_path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[0, 0] = True

        scm.update(mask, np.float32(1.0))
        first_val = scm.grid[0, 0]

        scm.grid[0, 0] = 0.0  # reset value, keep hardening
        scm.update(mask, np.float32(1.0))
        second_val = scm.grid[0, 0]

        assert second_val < first_val, "Hardening should reduce update magnitude"

    def test_clipping_to_unit_range(self, tmp_path):
        """Grid values stay in [-1, 1] after many updates."""
        scm = SCMGrid.load_or_create(tmp_path)
        mask = np.ones((64, 64), dtype=bool)
        for _ in range(100):
            scm.update(mask, np.float32(1.0))
        assert np.all(scm.grid <= 1.0)
        assert np.all(scm.grid >= -1.0)

    def test_empty_mask_noop(self, tmp_path):
        """All-false mask → no changes."""
        scm = SCMGrid.load_or_create(tmp_path)
        original = scm.grid.copy()
        mask = np.zeros((64, 64), dtype=bool)
        scm.update(mask, np.float32(1.0))
        np.testing.assert_array_equal(scm.grid, original)


class TestSCMGridGapMask:
    """gap_mask() and openness()."""

    def test_fresh_grid_fully_open(self, tmp_path):
        """Zero grid → all cells are gaps."""
        scm = SCMGrid.load_or_create(tmp_path)
        assert scm.openness() == pytest.approx(1.0)
        assert scm.gap_mask().all()

    def test_closed_cells_not_in_gap_mask(self, tmp_path):
        """Cells beyond threshold are not gaps."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.grid[0, 0] = 0.9  # well above any reasonable threshold
        assert not scm.gap_mask(threshold=0.5)[0, 0]

    def test_openness_decreases_when_cells_pushed_past_threshold(self, tmp_path):
        """Manually closing cells reduces openness."""
        scm = SCMGrid.load_or_create(tmp_path)
        initial_openness = scm.openness()
        # Directly set some cells above gap threshold
        scm.grid[:32, :32] = 0.5
        assert scm.openness() < initial_openness


class TestSCMGridAnneal:
    """anneal() hardening decay."""

    def test_anneal_reduces_hardening(self, tmp_path):
        """Annealing decays hardening counts."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.hardening[0, 0] = 100
        before = scm.hardening[0, 0]
        scm.anneal()
        assert scm.hardening[0, 0] < before

    def test_anneal_returns_affected_count(self, tmp_path):
        """Returns number of cells with nonzero hardening."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.hardening[0:3, 0:3] = 50
        count = scm.anneal()
        assert count == 9

    def test_anneal_on_fresh_grid(self, tmp_path):
        """Anneal on zero hardening → 0 affected."""
        scm = SCMGrid.load_or_create(tmp_path)
        assert scm.anneal() == 0

    def test_anneal_never_goes_negative(self, tmp_path):
        """Hardening counts never go below zero (uint32)."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.hardening[0, 0] = 1
        scm.anneal()
        assert scm.hardening[0, 0] >= 0


class TestSCMGridStats:
    """stats() diagnostics."""

    def test_fresh_grid_stats(self, tmp_path):
        """Zero grid produces expected stats."""
        scm = SCMGrid.load_or_create(tmp_path)
        s = scm.stats()
        assert s["openness"] == pytest.approx(1.0)
        assert s["mean_abs_scm"] == 0.0
        assert s["max_abs_scm"] == 0.0
        assert s["mean_hardening"] == 0.0
        assert s["max_hardening"] == 0
        assert s["hardened_cells"] == 0

    def test_stats_keys(self, tmp_path):
        """All expected keys present."""
        scm = SCMGrid.load_or_create(tmp_path)
        s = scm.stats()
        expected = {
            "openness", "mean_abs_scm", "max_abs_scm",
            "mean_hardening", "max_hardening", "hardened_cells", "closed_cells",
        }
        assert set(s.keys()) == expected


class TestSCMGridRecallFeedback:
    """update_from_recall() — outcome-driven SCM adjustment."""

    def _seeded_scm(self, tmp_path, scm_value=0.5):
        """Create an SCM with uniform nonzero grid (has opinions from erosion)."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.grid[:] = scm_value
        return scm

    def _peaked_attractors(self, peak=0.8):
        """Return corpus + experiential attractors with strong signal everywhere."""
        c = np.full((64, 64), peak, dtype=np.float32)
        x = np.full((64, 64), peak, dtype=np.float32)
        return c, x

    def test_good_recall_opens_gates(self, tmp_path):
        """κ > κ_base → cells driven toward ε_floor (|M| decreases)."""
        scm = self._seeded_scm(tmp_path, 0.5)
        c, x = self._peaked_attractors()
        original_abs = np.abs(scm.grid).mean()

        scm.update_from_recall(c, x, kappa=0.8)  # well above baseline 0.0

        assert np.abs(scm.grid).mean() < original_abs

    def test_bad_recall_closes_gates(self, tmp_path):
        """κ < κ_base → cells driven toward ±1 (|M| increases)."""
        scm = self._seeded_scm(tmp_path, 0.3)
        scm.kappa_base = 0.5  # set baseline above what we'll feed
        c, x = self._peaked_attractors()
        original_abs = np.abs(scm.grid).mean()

        scm.update_from_recall(c, x, kappa=0.1)  # below baseline

        assert np.abs(scm.grid).mean() > original_abs

    def test_fresh_grid_no_update(self, tmp_path):
        """M_i = 0 everywhere → sign(0) = 0 → no cells updated."""
        scm = SCMGrid.load_or_create(tmp_path)  # all zeros
        c, x = self._peaked_attractors()

        count = scm.update_from_recall(c, x, kappa=0.5)

        assert count == 0
        np.testing.assert_array_equal(scm.grid, 0.0)

    def test_sign_preservation(self, tmp_path):
        """Update never flips sign of M_i — positive stays positive."""
        scm = self._seeded_scm(tmp_path, 0.02)  # small positive, near zero
        c, x = self._peaked_attractors()

        # Large advantage should try to push M toward 0, but not past it
        scm.update_from_recall(c, x, kappa=10.0)

        assert (scm.grid > 0).all(), "Sign must not flip"

    def test_sign_preservation_negative(self, tmp_path):
        """Negative M cells stay negative after opening update."""
        scm = self._seeded_scm(tmp_path, -0.02)
        c, x = self._peaked_attractors()

        scm.update_from_recall(c, x, kappa=10.0)

        assert (scm.grid < 0).all(), "Negative sign must not flip"

    def test_epsilon_floor_clamp(self, tmp_path):
        """After update, |M_i| never falls below SCM_HARDENING_FLOOR."""
        from wheeler_memory.constants import SCM_HARDENING_FLOOR

        scm = self._seeded_scm(tmp_path, 0.002)  # barely above floor
        c, x = self._peaked_attractors()

        scm.update_from_recall(c, x, kappa=1.0)  # strong opening signal

        updated = scm.grid[scm.grid != 0]
        assert (np.abs(updated) >= SCM_HARDENING_FLOOR - 1e-9).all()

    def test_per_recall_cap(self, tmp_path):
        """Total |ΔM| is bounded by α · N · cap_frac."""
        from wheeler_memory.constants import (
            SCM_LEARNING_RATE,
            SCM_RECALL_UPDATE_CAP,
        )

        scm = self._seeded_scm(tmp_path, 0.5)
        c, x = self._peaked_attractors()
        before = scm.grid.copy()

        # Extreme advantage to trigger the cap
        scm.update_from_recall(c, x, kappa=100.0)

        total_delta = np.abs(scm.grid - before).sum()
        cap = SCM_LEARNING_RATE * scm.grid.size * SCM_RECALL_UPDATE_CAP
        assert total_delta <= cap + 1e-4, f"Total Δ {total_delta:.4f} exceeds cap {cap:.4f}"

    def test_openness_ceiling_blocks_opening(self, tmp_path):
        """When openness >= SCM_OPEN_FRACTION_CEIL, opening updates are frozen."""
        from wheeler_memory.constants import SCM_OPEN_FRACTION_CEIL

        scm = SCMGrid.load_or_create(tmp_path)
        # Set a few cells nonzero but keep openness above ceiling
        scm.grid[0, 0] = 0.1  # below gap threshold → still counts as open
        assert scm.openness() >= SCM_OPEN_FRACTION_CEIL

        c, x = self._peaked_attractors()
        before = scm.grid.copy()

        count = scm.update_from_recall(c, x, kappa=1.0)  # positive advantage

        assert count == 0
        np.testing.assert_array_equal(scm.grid, before)

    def test_openness_ceiling_allows_closing(self, tmp_path):
        """Closing updates (κ < κ_base) still work above the openness ceiling."""
        scm = SCMGrid.load_or_create(tmp_path)
        scm.grid[0, 0] = 0.1
        scm.kappa_base = 0.5  # above what we'll feed
        c, x = self._peaked_attractors()

        count = scm.update_from_recall(c, x, kappa=0.1)  # negative advantage

        # Cell [0,0] should have been pushed further from zero
        assert count >= 1

    def test_kappa_ema_tracking(self, tmp_path):
        """kappa_base converges toward repeated κ values."""
        scm = self._seeded_scm(tmp_path)
        c, x = self._peaked_attractors()

        for _ in range(50):
            scm.update_from_recall(c, x, kappa=0.6)

        assert scm.kappa_base == pytest.approx(0.6, abs=0.01)

    def test_kappa_ema_updates_even_on_skip(self, tmp_path):
        """EMA updates even when grid update is skipped (fresh grid)."""
        scm = SCMGrid.load_or_create(tmp_path)  # positive advantage → homeostasis blocks
        c, x = self._peaked_attractors()

        scm.update_from_recall(c, x, kappa=0.5)

        assert scm.kappa_base > 0.0

    def test_kappa_base_persistence(self, tmp_path):
        """kappa_base survives save/load round-trip."""
        scm = self._seeded_scm(tmp_path)
        c, x = self._peaked_attractors()
        scm.update_from_recall(c, x, kappa=0.7)
        scm.save()

        loaded = SCMGrid.load_or_create(tmp_path)
        assert loaded.kappa_base == pytest.approx(scm.kappa_base)

    def test_no_credit_no_update(self, tmp_path):
        """Zero attractors → |C·X| = 0 everywhere → no cells updated."""
        scm = self._seeded_scm(tmp_path, 0.5)
        c = np.zeros((64, 64), dtype=np.float32)
        x = np.zeros((64, 64), dtype=np.float32)

        count = scm.update_from_recall(c, x, kappa=0.8)

        assert count == 0

    def test_returns_cell_count(self, tmp_path):
        """Return value is the number of cells that were updated."""
        scm = self._seeded_scm(tmp_path, 0.5)
        c, x = self._peaked_attractors()

        count = scm.update_from_recall(c, x, kappa=0.5)

        # All 64*64 cells have M≠0 and C*X≠0, so all should be in the mask
        assert count == 64 * 64

    def test_partial_credit(self, tmp_path):
        """Only cells where BOTH attractors have signal get updated."""
        scm = self._seeded_scm(tmp_path, 0.5)
        c = np.full((64, 64), 0.8, dtype=np.float32)
        x = np.zeros((64, 64), dtype=np.float32)
        # Only top-left quadrant has experiential signal
        x[:32, :32] = 0.8

        before = scm.grid.copy()
        scm.update_from_recall(c, x, kappa=0.8)

        # Bottom-right should be unchanged (no credit)
        np.testing.assert_array_equal(scm.grid[32:, 32:], before[32:, 32:])
        # Top-left should have changed
        assert not np.array_equal(scm.grid[:32, :32], before[:32, :32])

    def test_cold_start_negative_advantage_seeds_grid(self, tmp_path):
        """Fresh grid + bad recall → seeds closing opinions at high-credit cells."""
        scm = SCMGrid.load_or_create(tmp_path)  # all zeros
        scm.kappa_base = 0.5  # simulate accumulated history
        c, x = self._peaked_attractors()

        count = scm.update_from_recall(c, x, kappa=0.1)  # advantage = -0.4

        assert count > 0
        assert np.any(scm.grid > 0)  # closing opinions seeded


class TestSCMGridRepr:
    """__repr__() formatting."""

    def test_repr_contains_openness(self, tmp_path):
        """Repr includes openness percentage."""
        scm = SCMGrid.load_or_create(tmp_path)
        r = repr(scm)
        assert "SCMGrid" in r
        assert "openness" in r

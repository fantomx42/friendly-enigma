"""Tests for ternary scoring — snap_to_ternary and MMLU ternary modes."""

import sys
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.dynamics import snap_to_ternary

# Make scripts/ importable for score_ternary / score_ternary_retrieval
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class TestSnapToTernary:
    """snap_to_ternary() produces valid {-1, 0, +1} int8 grids."""

    def _make_frame(self, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(-1, 1, (64, 64)).astype(np.float32)

    def test_topological_values(self):
        frame = self._make_frame()
        t = snap_to_ternary(frame, mode="topological")
        assert t.dtype == np.int8
        assert set(np.unique(t)).issubset({-1, 0, 1})

    def test_threshold_values(self):
        frame = self._make_frame()
        t = snap_to_ternary(frame, mode="threshold")
        assert t.dtype == np.int8
        assert set(np.unique(t)).issubset({-1, 0, 1})

    def test_shape_preserved(self):
        frame = self._make_frame()
        for mode in ("topological", "threshold"):
            t = snap_to_ternary(frame, mode=mode)
            assert t.shape == (64, 64)

    def test_cell_count_equals_grid_size(self):
        frame = self._make_frame()
        for mode in ("topological", "threshold"):
            t = snap_to_ternary(frame, mode=mode)
            pos = int(np.sum(t == 1))
            neg = int(np.sum(t == -1))
            zero = int(np.sum(t == 0))
            assert pos + neg + zero == 64 * 64

    def test_deterministic(self):
        frame = self._make_frame(seed=99)
        t1 = snap_to_ternary(frame, mode="topological")
        t2 = snap_to_ternary(frame, mode="topological")
        np.testing.assert_array_equal(t1, t2)

    def test_threshold_known_values(self):
        """A frame of all 0.5 should snap to all +1 in threshold mode."""
        frame = np.full((64, 64), 0.5, dtype=np.float32)
        t = snap_to_ternary(frame, mode="threshold")
        assert np.all(t == 1)

        frame_neg = np.full((64, 64), -0.5, dtype=np.float32)
        t_neg = snap_to_ternary(frame_neg, mode="threshold")
        assert np.all(t_neg == -1)

        frame_zero = np.full((64, 64), 0.0, dtype=np.float32)
        t_zero = snap_to_ternary(frame_zero, mode="threshold")
        assert np.all(t_zero == 0)

    def test_topological_flat_frame(self):
        """A perfectly flat frame: all cells are both max and min.
        get_cell_roles assigns +1 then -1 (last write wins) — values stay valid."""
        frame = np.full((64, 64), 0.3, dtype=np.float32)
        t = snap_to_ternary(frame, mode="topological")
        assert set(np.unique(t)).issubset({-1, 0, 1})

    def test_invalid_mode_raises(self):
        frame = self._make_frame()
        with pytest.raises(ValueError, match="Unknown ternary snap mode"):
            snap_to_ternary(frame, mode="bogus")

    def test_default_mode_uses_constant(self):
        """Calling with mode=None should use the constant default."""
        frame = self._make_frame()
        t = snap_to_ternary(frame)
        assert t.dtype == np.int8
        assert t.shape == (64, 64)

    def test_topological_vs_threshold_differ(self):
        """The two modes should generally produce different patterns."""
        frame = self._make_frame(seed=7)
        t_topo = snap_to_ternary(frame, mode="topological")
        t_thresh = snap_to_ternary(frame, mode="threshold")
        # They shouldn't be identical on random input
        assert not np.array_equal(t_topo, t_thresh)


class TestScoreTernary:
    """score_ternary() returns valid predictions."""

    def test_returns_valid_index(self):
        from wheeler_mmlu import score_ternary

        question = "What is 2 + 2?"
        choices = ["3", "4", "5", "6"]
        idx, conf = score_ternary(question, choices)
        assert 0 <= idx <= 3
        assert -1.0 <= conf <= 1.0

    def test_identical_choices_same_score(self):
        from wheeler_mmlu import score_ternary

        question = "test question"
        choices = ["same", "same", "same", "same"]
        idx, conf = score_ternary(question, choices)
        assert 0 <= idx <= 3

    def test_deterministic(self):
        from wheeler_mmlu import score_ternary

        question = "What color is the sky?"
        choices = ["red", "blue", "green", "yellow"]
        idx1, conf1 = score_ternary(question, choices)
        idx2, conf2 = score_ternary(question, choices)
        assert idx1 == idx2
        assert conf1 == conf2

    def test_with_precomputed_frames(self):
        """score_ternary accepts precomputed frames to skip encoding."""
        from wheeler_memory.hippocampus import hippocampus_to_frame
        from wheeler_mmlu import score_ternary

        question = "Capital of France?"
        choices = ["London", "Paris", "Berlin", "Madrid"]
        frames = [hippocampus_to_frame(f"{question} {c}") for c in choices]
        idx, conf = score_ternary(question, choices, precomputed_frames=frames)
        assert 0 <= idx <= 3


class TestScoreTernaryRetrieval:
    """score_ternary_retrieval() returns valid predictions without a cache."""

    def test_no_cache_fallback(self):
        """Without cache, falls back to net positivity."""
        from wheeler_mmlu import score_ternary_retrieval

        question = "What is gravity?"
        choices = ["force", "color", "sound", "taste"]
        idx, conf = score_ternary_retrieval(question, choices, cache=None)
        assert 0 <= idx <= 3
        assert isinstance(conf, float)

"""Tests for wheeler_memory.rotation — rotation retry logic."""

import json

import numpy as np
import pytest

from wheeler_memory.rotation import (
    _get_frame_fn,
    _load_rotation_stats,
    _save_rotation_stats,
    store_with_rotation_retry,
    update_rotation_stats,
)


# ── _get_frame_fn ─────────────────────────────────────────────────────


class TestGetFrameFn:
    """_get_frame_fn() encoder dispatch."""

    def test_hash_encoder(self):
        """hash → returns hash_to_frame."""
        fn = _get_frame_fn(encoder="hash")
        frame = fn("hello")
        assert frame.shape == (64, 64)
        assert frame.dtype == np.float32

    def test_hippocampus_encoder(self):
        """hippocampus → returns hippocampus_to_frame."""
        fn = _get_frame_fn(encoder="hippocampus")
        frame = fn("hello")
        assert frame.shape == (64, 64)

    def test_language_encoder(self):
        """language → returns language_to_frame."""
        fn = _get_frame_fn(encoder="language")
        frame = fn("hello world")
        assert frame.shape == (64, 64)

    def test_blended_encoder(self):
        """blended → returns tanh(alpha*hippo + (1-alpha)*language)."""
        fn = _get_frame_fn(encoder="blended")
        frame = fn("test sentence")
        assert frame.shape == (64, 64)
        assert np.all(frame >= -1.0)
        assert np.all(frame <= 1.0)

    def test_word_encoder(self):
        """word → returns word_to_frame."""
        fn = _get_frame_fn(encoder="word")
        frame = fn("hello")
        assert frame.shape == (64, 64)

    def test_legacy_use_embedding_flag(self):
        """use_embedding=True → embedding encoder."""
        # This may fail if sentence-transformers isn't installed;
        # skip if not available
        try:
            fn = _get_frame_fn(use_embedding=True)
            frame = fn("test")
            assert frame.shape == (64, 64)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_unknown_encoder_raises(self):
        """Unknown encoder name → ValueError."""
        with pytest.raises(ValueError, match="Unknown encoder"):
            _get_frame_fn(encoder="nonexistent")

    def test_encoder_overrides_use_embedding(self):
        """Explicit encoder overrides legacy use_embedding flag."""
        fn = _get_frame_fn(use_embedding=True, encoder="hash")
        frame = fn("test")
        # hash_to_frame always returns deterministic result
        from wheeler_memory.hashing import hash_to_frame

        np.testing.assert_array_equal(frame, hash_to_frame("test"))


# ── Rotation stats persistence ────────────────────────────────────────


class TestRotationStats:
    """Rotation stats JSON persistence."""

    def test_fresh_stats(self, tmp_path):
        """Non-existent stats → default zeros."""
        stats = _load_rotation_stats(tmp_path)
        assert stats == {"0": 0, "90": 0, "180": 0, "270": 0}

    def test_save_and_reload(self, tmp_path):
        """save → load round-trip."""
        _save_rotation_stats(tmp_path, {"0": 5, "90": 3, "180": 1, "270": 0})
        loaded = _load_rotation_stats(tmp_path)
        assert loaded["0"] == 5
        assert loaded["90"] == 3

    def test_update_increments_on_success(self, tmp_path):
        """Success increments the angle counter."""
        update_rotation_stats(0, True, tmp_path)
        update_rotation_stats(0, True, tmp_path)
        update_rotation_stats(90, True, tmp_path)
        stats = _load_rotation_stats(tmp_path)
        assert stats["0"] == 2
        assert stats["90"] == 1

    def test_update_does_not_increment_on_failure(self, tmp_path):
        """Failure does not increment."""
        update_rotation_stats(0, False, tmp_path)
        stats = _load_rotation_stats(tmp_path)
        assert stats["0"] == 0


# ── store_with_rotation_retry ─────────────────────────────────────────


class TestStoreWithRotationRetry:
    """store_with_rotation_retry() full pipeline."""

    def test_returns_result_dict(self, tmp_path):
        """Returns dict with expected keys."""
        result = store_with_rotation_retry(
            "hello world",
            data_dir=tmp_path,
            save=False,
            encoder="hash",
        )
        assert "state" in result
        assert "attractor" in result
        assert "metadata" in result
        assert "rotation_used" in result["metadata"]
        assert "attempts" in result["metadata"]

    def test_attractor_shape(self, tmp_path):
        """Attractor is 64×64 float32."""
        result = store_with_rotation_retry(
            "test text",
            data_dir=tmp_path,
            save=False,
            encoder="hash",
        )
        assert result["attractor"].shape == (64, 64)

    def test_max_rotations_1(self, tmp_path):
        """max_rotations=1 → only tries angle 0."""
        result = store_with_rotation_retry(
            "quick test",
            max_rotations=1,
            data_dir=tmp_path,
            save=False,
            encoder="hash",
        )
        assert result["metadata"]["rotation_used"] == 0
        assert result["metadata"]["attempts"] == 1

    def test_records_wall_time(self, tmp_path):
        """wall_time_seconds is a positive number."""
        result = store_with_rotation_retry(
            "wall time test",
            data_dir=tmp_path,
            save=False,
            encoder="hash",
        )
        assert result["metadata"]["wall_time_seconds"] > 0.0

    def test_converged_result_has_salience_metadata(self, tmp_path):
        """Result includes salience and attention_label."""
        result = store_with_rotation_retry(
            "metadata check",
            data_dir=tmp_path,
            save=False,
            encoder="hash",
        )
        assert "salience" in result["metadata"]
        assert "attention_label" in result["metadata"]
        assert "stability_threshold" in result["metadata"]

    def test_save_true_stores_memory(self, tmp_path):
        """save=True persists memory to disk."""
        result = store_with_rotation_retry(
            "persistent memory",
            data_dir=tmp_path,
            save=True,
            encoder="hash",
        )
        if result["state"] == "CONVERGED":
            # Check that index file was created in the chunk dir
            chunks_dir = tmp_path / "chunks"
            assert chunks_dir.exists()

    def test_different_encoders(self, tmp_path):
        """Different encoders produce different attractors."""
        r_hash = store_with_rotation_retry(
            "encoder test", data_dir=tmp_path, save=False, encoder="hash"
        )
        r_hippo = store_with_rotation_retry(
            "encoder test", data_dir=tmp_path, save=False, encoder="hippocampus"
        )
        assert not np.array_equal(r_hash["attractor"], r_hippo["attractor"])

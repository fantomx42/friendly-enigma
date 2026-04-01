"""Tests for wheeler_memory.hippocampus — n-gram random indexing encoder."""

import numpy as np
import pytest

from wheeler_memory.hippocampus import (
    EMBED_DIM,
    FRAME_SIZE,
    hippocampus_to_frame,
    hippocampus_to_frame_batch,
)


class TestHippocampusToFrame:
    """hippocampus_to_frame() text → 64×64 CA frame."""

    def test_output_shape(self):
        """Output is 64×64."""
        frame = hippocampus_to_frame("hello world")
        assert frame.shape == (FRAME_SIZE, FRAME_SIZE)

    def test_output_dtype(self):
        """Output is float32."""
        frame = hippocampus_to_frame("hello world")
        assert frame.dtype == np.float32

    def test_values_in_range(self):
        """All values in (-1, 1) due to tanh."""
        frame = hippocampus_to_frame("The quick brown fox jumps over the lazy dog")
        assert np.all(frame >= -1.0)
        assert np.all(frame <= 1.0)

    def test_deterministic(self):
        """Same text → identical frame."""
        f1 = hippocampus_to_frame("determinism test")
        f2 = hippocampus_to_frame("determinism test")
        np.testing.assert_array_equal(f1, f2)

    def test_different_text_different_frame(self):
        """Different text → different frame."""
        f1 = hippocampus_to_frame("cat")
        f2 = hippocampus_to_frame("dog")
        assert not np.array_equal(f1, f2)

    def test_similar_text_higher_correlation(self):
        """Similar texts → more correlated frames than dissimilar texts."""
        f_cat = hippocampus_to_frame("the cat sat on the mat")
        f_cat2 = hippocampus_to_frame("the cat sat on the rug")
        f_physics = hippocampus_to_frame("quantum electrodynamics describes photons")

        corr_similar = np.corrcoef(f_cat.flat, f_cat2.flat)[0, 1]
        corr_distant = np.corrcoef(f_cat.flat, f_physics.flat)[0, 1]
        assert corr_similar > corr_distant

    def test_case_insensitive(self):
        """Lowered internally → same frame for different cases."""
        f1 = hippocampus_to_frame("Hello World")
        f2 = hippocampus_to_frame("hello world")
        np.testing.assert_array_equal(f1, f2)

    def test_empty_string(self):
        """Empty string → zero frame (no n-grams)."""
        frame = hippocampus_to_frame("")
        np.testing.assert_array_equal(frame, np.zeros((64, 64), dtype=np.float32))

    def test_short_string(self):
        """Two-char string → only one 3-gram possible (none), so near-zero."""
        frame = hippocampus_to_frame("ab")
        assert frame.shape == (64, 64)

    def test_nonzero_values(self):
        """Non-trivial text produces non-zero frame."""
        frame = hippocampus_to_frame("hello world this is a test")
        assert np.abs(frame).sum() > 0


class TestHippocampusBatch:
    """hippocampus_to_frame_batch() batch encoding."""

    def test_batch_matches_individual(self):
        """Batch output matches individual calls."""
        texts = ["hello world", "foo bar baz", "quantum computing"]
        batch = hippocampus_to_frame_batch(texts)
        for i, text in enumerate(texts):
            individual = hippocampus_to_frame(text)
            np.testing.assert_array_equal(batch[i], individual)

    def test_batch_length(self):
        """Batch returns same number of frames as inputs."""
        texts = ["a", "b", "c"]
        batch = hippocampus_to_frame_batch(texts)
        assert len(batch) == 3

    def test_each_frame_is_64x64(self):
        """Each frame in batch has correct shape."""
        batch = hippocampus_to_frame_batch(["one", "two"])
        for frame in batch:
            assert frame.shape == (64, 64)
            assert frame.dtype == np.float32

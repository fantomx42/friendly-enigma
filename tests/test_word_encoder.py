"""Tests for the word-level random indexing encoder."""

import numpy as np
import pytest

from wheeler_memory.word_encoder import (
    _tokenize,
    _word_hash,
    _extract_words,
    _text_to_vector,
    word_to_frame,
    word_to_frame_batch,
)


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert _tokenize("Hello, World!") == ["hello", "world"]

    def test_numbers_kept(self):
        assert _tokenize("test123 foo") == ["test123", "foo"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestWordHash:
    def test_deterministic(self):
        assert _word_hash("paris") == _word_hash("paris")

    def test_different_words_different_hash(self):
        assert _word_hash("paris") != _word_hash("france")

    def test_range(self):
        from wheeler_memory.constants import WORD_HASH_SPACE

        for w in ["hello", "world", "test", "xyz"]:
            h = _word_hash(w)
            assert 0 <= h < WORD_HASH_SPACE


class TestExtractWords:
    def test_basic_counts(self):
        result = _extract_words("the cat sat on the mat")
        # "the" appears twice → log1p(2) ≈ 1.099
        # Others appear once → log1p(1) ≈ 0.693
        the_bucket = _word_hash("the")
        assert result[the_bucket] == pytest.approx(np.log1p(2), abs=1e-5)

    def test_empty(self):
        assert _extract_words("") == {}


class TestTextToVector:
    def test_shape(self):
        v = _text_to_vector("hello world")
        assert v.shape == (384,)
        assert v.dtype == np.float32

    def test_unit_norm(self):
        v = _text_to_vector("hello world")
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_empty_returns_zero(self):
        v = _text_to_vector("")
        assert np.allclose(v, 0.0)

    def test_deterministic(self):
        v1 = _text_to_vector("test input")
        v2 = _text_to_vector("test input")
        np.testing.assert_array_equal(v1, v2)


class TestWordToFrame:
    def test_shape_and_dtype(self):
        frame = word_to_frame("hello world")
        assert frame.shape == (64, 64)
        assert frame.dtype == np.float32

    def test_values_in_range(self):
        frame = word_to_frame("hello world")
        assert np.all(frame > -1.0)
        assert np.all(frame < 1.0)

    def test_deterministic(self):
        f1 = word_to_frame("test input")
        f2 = word_to_frame("test input")
        np.testing.assert_array_equal(f1, f2)

    def test_bag_of_words_invariance(self):
        """Same words in different order produce nearly identical frames.

        Tiny float diffs from accumulation order in L2 norm — atol=1e-6 is fine.
        """
        f1 = word_to_frame("Paris is the capital of France")
        f2 = word_to_frame("France capital the is Paris of")
        np.testing.assert_allclose(f1, f2, atol=1e-6)

    def test_different_words_different_frames(self):
        f1 = word_to_frame("Paris France capital")
        f2 = word_to_frame("Pizza Pasta Italian")
        assert not np.allclose(f1, f2)

    def test_semantic_similarity_ordering(self):
        """Texts with shared words should be more similar than unrelated texts."""
        f1 = word_to_frame("Paris is the capital of France")
        f2 = word_to_frame("The capital of France is Paris")
        f3 = word_to_frame("Pizza is an Italian food")

        # Pearson correlation
        def pearson(a, b):
            a_flat, b_flat = a.flatten(), b.flatten()
            a_c = a_flat - a_flat.mean()
            b_c = b_flat - b_flat.mean()
            return float(np.dot(a_c, b_c) / (np.linalg.norm(a_c) * np.linalg.norm(b_c)))

        sim_same = pearson(f1, f2)  # identical bags → should be 1.0
        sim_diff = pearson(f1, f3)  # different bags → should be lower

        assert sim_same > sim_diff, (
            f"Same-content similarity ({sim_same:.4f}) should exceed "
            f"cross-content similarity ({sim_diff:.4f})"
        )

    def test_different_from_hippocampus(self):
        """Word encoder should produce different frames than hippocampus."""
        from wheeler_memory.hippocampus import hippocampus_to_frame

        w = word_to_frame("hello world")
        h = hippocampus_to_frame("hello world")
        assert not np.allclose(w, h)


class TestWordToFrameBatch:
    def test_matches_individual(self):
        texts = ["hello world", "foo bar baz", "test input"]
        batch = word_to_frame_batch(texts)
        for i, text in enumerate(texts):
            individual = word_to_frame(text)
            np.testing.assert_array_equal(batch[i], individual)

    def test_length(self):
        texts = ["a", "b", "c"]
        batch = word_to_frame_batch(texts)
        assert len(batch) == 3

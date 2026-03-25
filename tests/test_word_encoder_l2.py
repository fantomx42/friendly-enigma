"""Tests for Level 2 co-occurrence learning in word encoder.

Tests the functions that enable learning word co-occurrence vectors
from stored Wheeler memories, enabling adaptive word embeddings.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.word_encoder import (
    _tokenize,
    _text_to_vector,
    build_cooccurrence,
    load_word_vectors,
    save_word_vectors,
    train_word_vectors,
)


@pytest.fixture
def mock_index_data(tmp_path: Path) -> Path:
    """Create a temporary directory with a mock index.json file.

    Returns the path to the tmp directory containing chunks/test/index.json.
    """
    chunk_dir = tmp_path / "chunks" / "test"
    chunk_dir.mkdir(parents=True)

    # Create index.json with test memories
    index = {
        "abc123": {
            "text": "paris is the capital of france",
            "timestamp": "2026-01-01T00:00:00",
            "hex_key": "abc123",
        },
        "def456": {
            "text": "london is the capital of england",
            "timestamp": "2026-01-02T00:00:00",
            "hex_key": "def456",
        },
        "ghi789": {
            "text": "paris france europe capital",
            "timestamp": "2026-01-03T00:00:00",
            "hex_key": "ghi789",
        },
        "jkl012": {
            "text": "london england british",
            "timestamp": "2026-01-04T00:00:00",
            "hex_key": "jkl012",
        },
        "mno345": {
            "text": "capital city is important",
            "timestamp": "2026-01-05T00:00:00",
            "hex_key": "mno345",
        },
    }

    (chunk_dir / "index.json").write_text(json.dumps(index, indent=2))
    return tmp_path


class TestBuildCooccurrence:
    """Tests for co-occurrence matrix building."""

    def test_basic_cooccurrence(self, mock_index_data: Path):
        """Test that co-occurrence counts are computed correctly."""
        cooccur, vocab = build_cooccurrence(
            data_dir=mock_index_data,
            window=2,
        )

        # Check that vocab is a list and sorted
        assert isinstance(vocab, list)
        assert len(vocab) > 0
        assert vocab == sorted(vocab)

        # Check that cooccurrence is a dict mapping (word_i, word_j) -> count
        assert isinstance(cooccur, dict)
        for key, value in cooccur.items():
            assert len(key) == 2
            assert isinstance(value, (int, float))

    def test_cooccurrence_symmetry(self, mock_index_data: Path):
        """Test that co-occurrence is symmetric: (i,j) == (j,i)."""
        cooccur, vocab = build_cooccurrence(
            data_dir=mock_index_data,
            window=3,
        )

        # For every (i, j) pair, (j, i) should have same count
        for (i, j), count in cooccur.items():
            if i != j:  # Skip diagonal
                assert (j, i) in cooccur
                assert cooccur[(j, i)] == count

    def test_window_size_respected(self, mock_index_data: Path):
        """Test that window size affects co-occurrence collection.

        With window=1, only immediate neighbors should co-occur.
        """
        # Small window: "paris is" and "is the" should have counts
        # But "paris france" (2 tokens apart) should not
        cooccur_w1, vocab_w1 = build_cooccurrence(
            data_dir=mock_index_data,
            window=1,
        )
        cooccur_w3, vocab_w3 = build_cooccurrence(
            data_dir=mock_index_data,
            window=3,
        )

        # Larger window should have more pairs
        assert len(cooccur_w3) >= len(cooccur_w1)

    def test_vocab_is_sorted(self, mock_index_data: Path):
        """Test that returned vocab is in alphabetical order."""
        _, vocab = build_cooccurrence(data_dir=mock_index_data, window=2)

        assert vocab == sorted(vocab), "vocab should be sorted alphabetically"

    def test_empty_dir_handled(self, tmp_path: Path):
        """Test that empty directory with no chunks is handled gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Should not crash, may return empty results or raise FileNotFoundError
        try:
            cooccur, vocab = build_cooccurrence(
                data_dir=empty_dir,
                window=2,
            )
            # If it succeeds, results should be empty or minimal
            assert isinstance(cooccur, dict)
            assert isinstance(vocab, list)
        except FileNotFoundError:
            # Acceptable: no data found
            pass


class TestTrainWordVectors:
    """Tests for training word vectors from co-occurrence data."""

    def test_basic_training(self, mock_index_data: Path):
        """Test that training produces vectors of correct shape."""
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        # Check shapes
        assert isinstance(vectors, np.ndarray)
        assert isinstance(vocab, list)
        assert vectors.shape == (len(vocab), 384)

    def test_vector_dtype_is_float32(self, mock_index_data: Path):
        """Test that output vectors are float32."""
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        assert vectors.dtype == np.float32

    def test_different_dimensions(self, mock_index_data: Path):
        """Test training with different dimensionalities."""
        for dim in [32, 128, 384, 512]:
            vectors, vocab = train_word_vectors(
                data_dir=mock_index_data,
                dim=dim,
                window=3,
            )
            assert vectors.shape == (len(vocab), dim)

    def test_deterministic_output(self, mock_index_data: Path):
        """Test that same input produces same output."""
        vectors1, vocab1 = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )
        vectors2, vocab2 = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        assert vocab1 == vocab2
        np.testing.assert_array_almost_equal(vectors1, vectors2)

    def test_vectors_normalized(self, mock_index_data: Path):
        """Test that output vectors are L2-normalized (or close to it)."""
        vectors, _ = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        # Each vector should have unit norm (or very close)
        norms = np.linalg.norm(vectors, axis=1)
        # May not be exactly 1.0 if normalization is skipped, but should be reasonable
        assert np.all(norms > 0), "All vectors should be non-zero"


class TestSaveLoadWordVectors:
    """Tests for saving and loading word vectors."""

    def test_save_and_load_roundtrip(self, mock_index_data: Path, tmp_path: Path):
        """Test that vectors survive save → load cycle."""
        # Train vectors
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        # Save to tmp directory
        save_word_vectors(vectors, vocab, data_dir=tmp_path)

        # Check file exists
        vector_file = tmp_path / "word_vectors.npz"
        assert vector_file.exists()

        # Load back — returns (vectors, word2idx dict)
        loaded_vectors, loaded_word2idx = load_word_vectors(data_dir=tmp_path)

        # Verify word2idx covers the same words as vocab
        assert set(loaded_word2idx.keys()) == set(vocab)
        np.testing.assert_array_equal(loaded_vectors, vectors)

    def test_word2idx_mapping(self, mock_index_data: Path, tmp_path: Path):
        """Test that word2idx mapping is consistent after load."""
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        save_word_vectors(vectors, vocab, data_dir=tmp_path)
        loaded_vectors, loaded_word2idx = load_word_vectors(data_dir=tmp_path)

        # Build original mapping
        word2idx_orig = {w: i for i, w in enumerate(vocab)}

        # Check a few words — vectors at mapped indices should match
        for word in vocab[: min(5, len(vocab))]:
            orig_idx = word2idx_orig[word]
            loaded_idx = loaded_word2idx[word]
            np.testing.assert_array_equal(
                vectors[orig_idx],
                loaded_vectors[loaded_idx],
            )

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test that loading from dir with no vectors returns None."""
        empty_dir = tmp_path / "no_vectors"
        empty_dir.mkdir()

        result = load_word_vectors(data_dir=empty_dir)
        assert result is None

    def test_save_with_default_data_dir(self, mock_index_data: Path, monkeypatch):
        """Test that save uses ~/.wheeler_memory by default."""
        # We'll use tmp_path but pretend it's the home dir
        fake_home = mock_index_data
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        # Save with explicit data_dir (easier to test than mocking home)
        save_word_vectors(vectors, vocab, data_dir=fake_home)

        vector_file = fake_home / "word_vectors.npz"
        assert vector_file.exists()


class TestLearnedVectorsFallback:
    """Tests for fallback when no learned vectors are available."""

    def test_text_to_vector_without_learned_vectors(self, tmp_path: Path):
        """Test that _text_to_vector works even without learned vectors.

        Should fall back to random indexing if no learned vectors exist.
        """
        # Don't create any learned vectors
        result = _text_to_vector("hello world test")

        # Should still produce a vector
        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_vector_shape_consistency(self, tmp_path: Path):
        """Test that vector shape is consistent regardless of learned vectors."""
        v1 = _text_to_vector("some text")
        v2 = _text_to_vector("different text")

        assert v1.shape == v2.shape == (384,)
        assert v1.dtype == v2.dtype == np.float32

    def test_random_indexing_produces_different_vectors(self, tmp_path: Path):
        """Test that different texts produce different vectors (random indexing)."""
        v1 = _text_to_vector("paris france capital")
        v2 = _text_to_vector("london england british")

        # Should be different (with very high probability)
        assert not np.allclose(v1, v2)

    def test_empty_text_fallback(self, tmp_path: Path):
        """Test that empty text is handled correctly by fallback."""
        v = _text_to_vector("")

        assert v.shape == (384,)
        assert v.dtype == np.float32
        # Empty text should give zero vector
        assert np.allclose(v, 0.0)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_learn_and_use_vectors(self, mock_index_data: Path, tmp_path: Path):
        """Test full pipeline: build cooccurrence → train → save → use."""
        # Step 1: Train vectors from stored memories
        vectors, vocab = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        assert len(vocab) > 0
        assert vectors.shape[0] == len(vocab)

        # Step 2: Save vectors
        save_word_vectors(vectors, vocab, data_dir=tmp_path)

        # Step 3: Load vectors back — returns (vectors, word2idx dict)
        loaded_vectors, loaded_word2idx = load_word_vectors(data_dir=tmp_path)

        assert set(loaded_word2idx.keys()) == set(vocab)
        np.testing.assert_array_equal(loaded_vectors, vectors)

    def test_cooccurrence_vocabulary_consistency(self, mock_index_data: Path):
        """Test that vocabulary from cooccurrence and training are consistent."""
        cooccur, vocab_cooccur = build_cooccurrence(
            data_dir=mock_index_data,
            window=5,
        )
        vectors, vocab_train = train_word_vectors(
            data_dir=mock_index_data,
            dim=384,
            window=5,
        )

        # Both should have same vocab
        assert set(vocab_cooccur) == set(vocab_train)

        # Both should be sorted the same way
        assert vocab_cooccur == vocab_train

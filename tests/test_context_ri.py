"""Tests for context-window random indexing encoder."""

import tempfile
import json
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.hippocampus import FRAME_SIZE
from wheeler_memory.word_encoder import (
    train_context_ri,
    save_context_ri_vectors,
    load_context_ri_vectors,
    context_to_frame,
    context_to_frame_batch,
    _text_to_vector_context_ri,
    _collect_texts,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_corpus(tmp_path):
    """A small JSONL corpus with semantic clusters for testing."""
    corpus_path = tmp_path / "corpus.jsonl"
    entries = [
        # Dog/pet cluster
        "the dog barked at the mailman",
        "the puppy chased its tail in the yard",
        "she walked her dog on a leash every morning",
        "the canine wagged its tail happily",
        "pet owners should walk their dogs daily",
        "the puppy learned to sit and stay",
        "dogs are loyal companions to humans",
        "the bark of the dog scared the cat away",
        # Science cluster
        "the atom contains protons neutrons and electrons",
        "quantum mechanics describes particle behavior",
        "electrons orbit the nucleus of an atom",
        "physics explains the fundamental forces of nature",
        "the molecule consists of bonded atoms",
        "energy is conserved in all physical processes",
        "particles exhibit wave duality at quantum scales",
        "the nucleus holds most of the atomic mass",
    ]
    with open(corpus_path, "w") as f:
        for text in entries:
            f.write(json.dumps({"text": text}) + "\n")
    return str(corpus_path)


@pytest.fixture
def trained_vectors(small_corpus, tmp_path):
    """Train context-RI vectors from the small corpus."""
    vectors, vocab = train_context_ri(
        data_dir=str(tmp_path),
        window=3,
        corpus_path=small_corpus,
        subsample_t=0,
        remove_top_k=0,
    )
    save_context_ri_vectors(vectors, vocab, data_dir=str(tmp_path))
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vectors, word2idx, vocab


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrainContextRI:
    def test_returns_vectors_and_vocab(self, small_corpus, tmp_path):
        vectors, vocab = train_context_ri(
            data_dir=str(tmp_path),
            corpus_path=small_corpus,
        )
        assert isinstance(vectors, np.ndarray)
        assert isinstance(vocab, list)
        assert len(vocab) > 0
        assert vectors.shape == (len(vocab), 384)

    def test_vectors_are_normalized(self, small_corpus, tmp_path):
        vectors, vocab = train_context_ri(
            data_dir=str(tmp_path),
            corpus_path=small_corpus,
        )
        norms = np.linalg.norm(vectors, axis=1)
        # All non-zero vectors should be unit length
        nonzero = norms > 0
        np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-5)

    def test_empty_corpus_returns_empty(self, tmp_path):
        vectors, vocab = train_context_ri(data_dir=str(tmp_path))
        assert len(vocab) == 0
        assert vectors.shape == (0, 384)

    def test_vocab_is_sorted(self, small_corpus, tmp_path):
        _, vocab = train_context_ri(
            data_dir=str(tmp_path),
            corpus_path=small_corpus,
        )
        assert vocab == sorted(vocab)

    def test_window_size_affects_vectors(self, small_corpus, tmp_path):
        v1, _ = train_context_ri(
            data_dir=str(tmp_path), corpus_path=small_corpus, window=2
        )
        v2, _ = train_context_ri(
            data_dir=str(tmp_path), corpus_path=small_corpus, window=5
        )
        # Different windows should produce different vectors
        assert not np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoadContextRI:
    def test_roundtrip(self, trained_vectors, tmp_path):
        vectors, word2idx, vocab = trained_vectors
        result = load_context_ri_vectors(data_dir=str(tmp_path))
        assert result is not None
        loaded_vectors, loaded_word2idx = result
        np.testing.assert_array_equal(loaded_vectors, vectors)
        assert set(loaded_word2idx.keys()) == set(word2idx.keys())

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_context_ri_vectors(data_dir=str(tmp_path / "nonexistent"))
        assert result is None


# ---------------------------------------------------------------------------
# Encoding (frame generation)
# ---------------------------------------------------------------------------


class TestContextToFrame:
    def test_output_shape(self):
        frame = context_to_frame("hello world")
        assert frame.shape == (FRAME_SIZE, FRAME_SIZE)

    def test_output_dtype(self):
        frame = context_to_frame("hello world")
        assert frame.dtype == np.float32

    def test_values_in_range(self):
        frame = context_to_frame("the quick brown fox")
        assert np.all(frame >= -1.0)
        assert np.all(frame <= 1.0)

    def test_deterministic(self):
        f1 = context_to_frame("determinism test")
        f2 = context_to_frame("determinism test")
        np.testing.assert_array_equal(f1, f2)

    def test_different_text_different_frame(self):
        f1 = context_to_frame("cat")
        f2 = context_to_frame("quantum physics")
        assert not np.array_equal(f1, f2)

    def test_empty_string_zero_frame(self):
        frame = context_to_frame("")
        np.testing.assert_array_equal(frame, np.zeros((64, 64), dtype=np.float32))

    def test_batch_matches_individual(self):
        texts = ["hello world", "foo bar baz", "quantum computing"]
        batch = context_to_frame_batch(texts)
        for i, text in enumerate(texts):
            individual = context_to_frame(text)
            np.testing.assert_array_equal(batch[i], individual)


# ---------------------------------------------------------------------------
# Vector blending with learned context
# ---------------------------------------------------------------------------


class TestContextRIBlending:
    def test_fallback_for_unseen_words(self, trained_vectors):
        """Words not in trained vocab should fall back to hippocampus."""
        from wheeler_memory.hippocampus import _text_to_vector as hippo_vector

        vectors, word2idx, _ = trained_vectors
        # Use a word that definitely won't be in the small corpus
        unseen = "xylophone"
        assert unseen not in word2idx

        # With blend=1.0, unseen words should produce pure hippocampus
        result = _text_to_vector_context_ri(unseen, vectors, word2idx, blend=1.0)
        expected = hippo_vector(unseen)
        norm = np.linalg.norm(expected)
        if norm > 0:
            expected /= norm
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_blend_zero_is_pure_hippocampus(self, trained_vectors):
        """blend=0.0 should give pure hippocampus vectors."""
        from wheeler_memory.hippocampus import _text_to_vector as hippo_vector

        vectors, word2idx, vocab = trained_vectors
        # Pick a word that IS in vocab
        word = vocab[0]
        result = _text_to_vector_context_ri(word, vectors, word2idx, blend=0.0)
        expected = hippo_vector(word)
        norm = np.linalg.norm(expected)
        if norm > 0:
            expected /= norm
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Semantic signal (the whole point)
# ---------------------------------------------------------------------------


class TestSemanticSignal:
    def test_same_domain_words_more_similar(self, trained_vectors):
        """Words from the same domain cluster should have higher cosine similarity."""
        vectors, word2idx, _ = trained_vectors

        def _cos_sim(w1, w2):
            v1 = _text_to_vector_context_ri(w1, vectors, word2idx)
            v2 = _text_to_vector_context_ri(w2, vectors, word2idx)
            dot = np.dot(v1, v2)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return 0.0
            return dot / (n1 * n2)

        # "dog" and "puppy" share context (pet cluster)
        # "dog" and "atom" are cross-domain
        sim_same = _cos_sim("dog", "puppy")
        sim_cross = _cos_sim("dog", "atom")

        assert sim_same > sim_cross, (
            f"Expected dog-puppy ({sim_same:.4f}) > dog-atom ({sim_cross:.4f})"
        )


# ---------------------------------------------------------------------------
# Text collection
# ---------------------------------------------------------------------------


class TestCollectTexts:
    def test_from_corpus_jsonl(self, small_corpus, tmp_path):
        texts = _collect_texts(data_dir=str(tmp_path), corpus_path=small_corpus)
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_from_index_json(self, tmp_path):
        """Collect from stored memories format."""
        chunks_dir = tmp_path / "chunks" / "general"
        chunks_dir.mkdir(parents=True)
        index = {
            "abc123": {"text": "hello world", "hash": "abc123"},
            "def456": {"text": "foo bar baz", "hash": "def456"},
        }
        with open(chunks_dir / "index.json", "w") as f:
            json.dump(index, f)

        texts = _collect_texts(data_dir=str(tmp_path))
        assert len(texts) == 2
        assert "hello world" in texts

    def test_empty_dir_returns_empty(self, tmp_path):
        texts = _collect_texts(data_dir=str(tmp_path / "nonexistent"))
        assert texts == []

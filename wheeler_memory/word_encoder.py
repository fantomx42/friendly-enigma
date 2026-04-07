"""Word-level random indexing encoder: text → 64×64 CA frame via whole-word hashing.

Replaces character n-gram tokenization (hippocampus) with whole-word units.
Same architecture: 384-dim intermediate → JL projection → 64×64 frame.

Pipeline:
  1. Text → tokenize to words (lowercase, split on non-alphanumeric)
  2. Each word → SHA-1 hash → bucket index in fixed random matrix
  3. Accumulate weighted rows (log1p sublinear weighting) → 384-dim vector
  4. L2 normalize → project 384 → 4096 via same JL matrix as hippocampus
  5. tanh(x * 3.0) → reshape 64×64 float32

Key difference from hippocampus: "Paris" and "France" share zero signal
(good — they shouldn't), but "Paris is the capital" and "the capital is Paris"
produce identical frames (bag-of-words invariance).
"""

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    WORD_PROJECTION_SEED,
    WORD_HASH_SPACE,
    BLEND_ALPHA,
    CONTEXT_RI_WINDOW,
    CONTEXT_RI_BLEND,
)
from .hippocampus import EMBED_DIM, FRAME_SIZE, FRAME_CELLS, _get_frame_matrix

# Cached word→384 matrix (lazy-initialized)
_word_matrix: np.ndarray | None = None


def _get_word_matrix() -> np.ndarray:
    """Fixed random matrix mapping word hash buckets to 384-dim vectors.

    Shape: (WORD_HASH_SPACE, EMBED_DIM) = (65536, 384)
    Seeded differently from hippocampus n-gram matrix so word and n-gram
    encodings occupy distinct subspaces.
    """
    global _word_matrix
    if _word_matrix is None:
        rng = np.random.Generator(np.random.PCG64(WORD_PROJECTION_SEED))
        raw = rng.standard_normal((WORD_HASH_SPACE, EMBED_DIM)).astype(np.float32)
        raw /= np.sqrt(EMBED_DIM)
        _word_matrix = raw
    return _word_matrix


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, strip empty tokens."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _word_hash(word: str) -> int:
    """Deterministic hash of a word string to a bucket index."""
    digest = hashlib.sha1(word.encode("utf-8", errors="replace")).digest()
    return struct.unpack("<I", digest[:4])[0] % WORD_HASH_SPACE


def _extract_words(text: str) -> dict[int, float]:
    """Extract weighted word buckets from text.

    Returns dict mapping hash bucket → log1p(count) weight.
    """
    words = _tokenize(text)
    counts: dict[int, int] = {}
    for w in words:
        bucket = _word_hash(w)
        counts[bucket] = counts.get(bucket, 0) + 1

    return {bucket: np.log1p(count) for bucket, count in counts.items()}


def _text_to_vector(text: str) -> np.ndarray:
    """Convert text to a 384-dim vector via word-level random indexing.

    Each word maps to a row in the fixed projection matrix.
    Rows are accumulated with sublinear frequency weights, then L2 normalized.
    Tries learned co-occurrence vectors first, falls back to random indexing.
    """
    learned = _get_learned_vectors()
    if learned is not None:
        return _text_to_vector_learned(text, learned[0], learned[1])

    # Fall back to random indexing
    word_weights = _extract_words(text)
    if not word_weights:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    matrix = _get_word_matrix()

    buckets = np.array(list(word_weights.keys()), dtype=np.int64)
    weights = np.array(list(word_weights.values()), dtype=np.float32)

    # Vectorized: sum of (weight_i * matrix[bucket_i])
    vector = (weights[:, None] * matrix[buckets]).sum(axis=0)

    # L2 normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


def word_to_frame(text: str, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert text to a 64×64 CA frame via word-level random indexing.

    Uses the same JL projection matrix as hippocampus (_get_frame_matrix)
    so the projection space is compatible with existing stored attractors.
    """
    vector = _text_to_vector(text)
    proj = _get_frame_matrix()

    frame_flat = vector @ proj
    frame_flat = np.tanh(frame_flat * 3.0)

    return frame_flat.reshape(size, size).astype(np.float32)


def word_to_frame_batch(texts: list[str], size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Batch-convert texts to CA frames via word-level random indexing."""
    vectors = np.stack([_text_to_vector(t) for t in texts])  # (N, 384)
    proj = _get_frame_matrix()  # (384, 4096)

    frames_flat = vectors @ proj  # (N, 4096)
    frames_flat = np.tanh(frames_flat * 3.0)

    return [
        frames_flat[i].reshape(size, size).astype(np.float32) for i in range(len(texts))
    ]


# Level 2: Co-occurrence vector learning
_learned_vectors: np.ndarray | None = None
_learned_word2idx: dict[str, int] | None = None
_learned_checked: bool = False


def build_cooccurrence(
    data_dir: str | None = None, window: int = 5
) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Scan stored texts from index.json files and count word co-occurrences.

    Returns (cooccurrence_dict, vocab_list) where cooccurrence_dict maps
    (word_a, word_b) tuples to co-occurrence counts (both orders stored).
    """
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")
    else:
        data_dir = str(data_dir)

    data_path = Path(data_dir)
    cooccurrence: dict[tuple[str, str], int] = {}
    vocab_set: set[str] = set()

    # Walk all chunks/*/index.json
    chunks_dir = data_path / "chunks"
    if not chunks_dir.exists():
        return cooccurrence, sorted(vocab_set)

    for chunk_dir in chunks_dir.iterdir():
        if not chunk_dir.is_dir():
            continue
        index_file = chunk_dir / "index.json"
        if not index_file.exists():
            continue

        try:
            with open(index_file, "r") as f:
                index_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # index_data is a dict of objects with "text" field
        for item in index_data.values():
            if not isinstance(item, dict) or "text" not in item:
                continue
            text = item["text"]
            words = _tokenize(text)

            # Sliding window of co-occurrences
            for i in range(len(words)):
                for j in range(i + 1, min(i + window, len(words))):
                    w1, w2 = words[i], words[j]
                    vocab_set.add(w1)
                    vocab_set.add(w2)

                    # Store both directions (symmetric)
                    key = (w1, w2)
                    cooccurrence[key] = cooccurrence.get(key, 0) + 1
                    if w1 != w2:
                        key_rev = (w2, w1)
                        cooccurrence[key_rev] = cooccurrence.get(key_rev, 0) + 1

    vocab = sorted(vocab_set)
    return cooccurrence, vocab


def _build_pmi_matrix(
    cooccurrence: dict[tuple[str, str], int], vocab: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Build PMI matrix from co-occurrence counts.

    Cap vocab to 50000 most frequent words. Returns (pmi_matrix, word2idx).
    """
    # Cap vocab size
    if len(vocab) > 50000:
        # Count total occurrences of each word
        word_counts: dict[str, int] = {}
        for (w1, w2), count in cooccurrence.items():
            word_counts[w1] = word_counts.get(w1, 0) + count
        # Keep top 50000 by count
        top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:50000]
        vocab = sorted(top_words)

    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # Build PMI matrix
    pmi_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    # Compute marginal counts
    margin_a: dict[int, int] = {}
    margin_b: dict[int, int] = {}
    total_count = 0

    for (w1, w2), count in cooccurrence.items():
        if w1 not in word2idx or w2 not in word2idx:
            continue
        i, j = word2idx[w1], word2idx[w2]
        margin_a[i] = margin_a.get(i, 0) + count
        margin_b[j] = margin_b.get(j, 0) + count
        total_count += count

    if total_count == 0:
        return pmi_matrix, word2idx

    total_float = float(total_count)

    # Fill PMI matrix: PMI = log(p(a,b) / (p(a) * p(b)))
    for (w1, w2), count in cooccurrence.items():
        if w1 not in word2idx or w2 not in word2idx:
            continue
        i, j = word2idx[w1], word2idx[w2]
        p_ab = count / total_float
        p_a = margin_a.get(i, 0) / total_float
        p_b = margin_b.get(j, 0) / total_float

        if p_a > 0 and p_b > 0:
            pmi = np.log(p_ab / (p_a * p_b))
            pmi_matrix[i, j] = max(0.0, pmi)  # Clamp to >= 0

    return pmi_matrix, word2idx


def train_word_vectors(
    data_dir: str | None = None, dim: int = 384, window: int = 5
) -> tuple[np.ndarray, list[str]]:
    """Train word vectors via SVD on PMI matrix.

    Returns (vectors, vocab) where vectors is (vocab_size, dim) and vocab is sorted word list.
    """
    cooccurrence, vocab = build_cooccurrence(data_dir, window)
    pmi_matrix, word2idx = _build_pmi_matrix(cooccurrence, vocab)

    # Try scipy.sparse.linalg.svds first for memory efficiency
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
    except ImportError:
        svds = None

    if svds is not None and pmi_matrix.size > 1_000_000:
        # Use sparse SVD for large matrices
        sparse_pmi = csr_matrix(pmi_matrix)
        U, S, Vt = svds(sparse_pmi, k=min(dim, min(pmi_matrix.shape) - 1))
    else:
        # Use dense SVD
        U, S, Vt = np.linalg.svd(pmi_matrix, full_matrices=False)

    # Learned vectors = U[:, :dim] * sqrt(S[:dim])
    k = min(dim, len(S))
    vectors = U[:, :k] * np.sqrt(S[:k])

    # Pad to dim if needed
    if vectors.shape[1] < dim:
        pad = np.zeros((vectors.shape[0], dim - vectors.shape[1]), dtype=np.float32)
        vectors = np.hstack([vectors, pad])

    return vectors[:, :dim].astype(np.float32), [v for v in word2idx.keys()]


def save_word_vectors(
    vectors: np.ndarray, vocab: list[str], data_dir: str | None = None
) -> None:
    """Save learned word vectors to data_dir / word_vectors.npz."""
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")
    else:
        data_dir = str(data_dir)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    npz_file = data_path / "word_vectors.npz"
    np.savez(npz_file, vectors=vectors, vocab=np.array(vocab, dtype=object))


def load_word_vectors(
    data_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Load learned word vectors from data_dir / word_vectors.npz.

    Returns (vectors, word2idx_dict) or None if file doesn't exist.
    """
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")
    else:
        data_dir = str(data_dir)

    npz_file = Path(data_dir) / "word_vectors.npz"
    if not npz_file.exists():
        return None

    try:
        data = np.load(npz_file, allow_pickle=True)
        vectors = data["vectors"].astype(np.float32)
        vocab = data["vocab"]
        word2idx = {w: i for i, w in enumerate(vocab)}
        return vectors, word2idx
    except (KeyError, IOError, ValueError):
        return None


def _get_learned_vectors(
    data_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Lazy-load learned vectors, caching result even if None."""
    global _learned_vectors, _learned_word2idx, _learned_checked

    if _learned_checked:
        if _learned_vectors is not None:
            return (_learned_vectors, _learned_word2idx)
        return None

    result = load_word_vectors(data_dir)
    _learned_checked = True

    if result is not None:
        _learned_vectors, _learned_word2idx = result
        return (_learned_vectors, _learned_word2idx)

    return None


def _text_to_vector_learned(
    text: str, vectors: np.ndarray, word2idx: dict[str, int]
) -> np.ndarray:
    """Convert text to 384-dim vector using learned co-occurrence vectors.

    Falls back to random indexing for unseen words.
    """
    words = _tokenize(text)
    if not words:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    # Accumulate learned vectors + fallback random indexing for unseen words
    matrix = _get_word_matrix()
    vector = np.zeros(EMBED_DIM, dtype=np.float32)

    for word, count in word_counts.items():
        weight = np.log1p(count)

        if word in word2idx:
            # Use learned vector
            idx = word2idx[word]
            vector += weight * vectors[idx]
        else:
            # Fall back to random indexing
            bucket = _word_hash(word)
            vector += weight * matrix[bucket]

    # L2 normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


# =========================================================================
# Context-window random indexing (distributional semantics via RI)
# =========================================================================

_context_ri_vectors: np.ndarray | None = None
_context_ri_word2idx: dict[str, int] | None = None
_context_ri_checked: bool = False


def _collect_texts(
    data_dir: str | None = None, corpus_path: str | None = None
) -> list[str]:
    """Collect texts from stored memories and/or a JSONL corpus file.

    Sources (both optional, at least one should provide data):
      1. chunks/*/index.json — stored Wheeler memories
      2. corpus_path — JSONL file with {"text": "..."} entries
    """
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")

    texts: list[str] = []

    # Source 1: stored memories (same iteration as build_cooccurrence)
    chunks_dir = Path(data_dir) / "chunks"
    if chunks_dir.exists():
        for chunk_dir in chunks_dir.iterdir():
            if not chunk_dir.is_dir():
                continue
            index_file = chunk_dir / "index.json"
            if not index_file.exists():
                continue
            try:
                with open(index_file, "r") as f:
                    index_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            for item in index_data.values():
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])

    # Source 2: JSONL corpus file
    if corpus_path is not None:
        corpus_file = Path(corpus_path)
        if corpus_file.exists():
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "text" in obj:
                            texts.append(obj["text"])
                    except json.JSONDecodeError:
                        continue

    return texts


def train_context_ri(
    data_dir: str | None = None,
    window: int = CONTEXT_RI_WINDOW,
    corpus_path: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Train word context vectors via Random Indexing with hippocampus n-gram index vectors.

    For each word in the corpus, accumulates the hippocampus n-gram vectors
    of its context-window neighbors.  Words appearing in similar contexts
    (e.g., "dog" and "canine" both near "bark", "leash", "pet")
    accumulate similar context vectors.

    Returns (vectors, vocab) where vectors is (vocab_size, 384) and
    vocab is a sorted list of words.
    """
    from .hippocampus import _text_to_vector as hippo_vector

    texts = _collect_texts(data_dir, corpus_path)
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []

    # Cache: word string → hippocampus 384-dim index vector
    index_cache: dict[str, np.ndarray] = {}

    def _get_index_vec(word: str) -> np.ndarray:
        if word not in index_cache:
            index_cache[word] = hippo_vector(word)
        return index_cache[word]

    # Accumulate context vectors
    context_accum: dict[str, np.ndarray] = {}

    for text in texts:
        words = _tokenize(text)
        if len(words) < 2:
            continue

        for i, target in enumerate(words):
            start = max(0, i - window)
            end = min(len(words), i + window + 1)

            if target not in context_accum:
                context_accum[target] = np.zeros(EMBED_DIM, dtype=np.float32)

            for j in range(start, end):
                if j != i:
                    context_accum[target] += _get_index_vec(words[j])

    # Build sorted output arrays
    vocab = sorted(context_accum.keys())
    vectors = np.zeros((len(vocab), EMBED_DIM), dtype=np.float32)

    for idx, word in enumerate(vocab):
        v = context_accum[word]
        norm = np.linalg.norm(v)
        if norm > 0:
            vectors[idx] = v / norm

    return vectors, vocab


def save_context_ri_vectors(
    vectors: np.ndarray, vocab: list[str], data_dir: str | None = None
) -> None:
    """Save learned context-RI vectors to data_dir / context_ri_vectors.npz."""
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    npz_file = data_path / "context_ri_vectors.npz"
    np.savez(npz_file, vectors=vectors, vocab=np.array(vocab, dtype=object))


def load_context_ri_vectors(
    data_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Load context-RI vectors from data_dir / context_ri_vectors.npz.

    Returns (vectors, word2idx_dict) or None if file doesn't exist.
    """
    if data_dir is None:
        data_dir = str(Path.home() / ".wheeler_memory")

    npz_file = Path(data_dir) / "context_ri_vectors.npz"
    if not npz_file.exists():
        return None

    try:
        data = np.load(npz_file, allow_pickle=True)
        vectors = data["vectors"].astype(np.float32)
        vocab = data["vocab"]
        word2idx = {w: i for i, w in enumerate(vocab)}
        return vectors, word2idx
    except (KeyError, IOError, ValueError):
        return None


def _get_context_ri_vectors(
    data_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Lazy-load context-RI vectors, caching result even if None."""
    global _context_ri_vectors, _context_ri_word2idx, _context_ri_checked

    if _context_ri_checked:
        if _context_ri_vectors is not None:
            return (_context_ri_vectors, _context_ri_word2idx)
        return None

    result = load_context_ri_vectors(data_dir)
    _context_ri_checked = True

    if result is not None:
        _context_ri_vectors, _context_ri_word2idx = result
        return (_context_ri_vectors, _context_ri_word2idx)

    return None


def _text_to_vector_context_ri(
    text: str,
    vectors: np.ndarray,
    word2idx: dict[str, int],
    blend: float = CONTEXT_RI_BLEND,
) -> np.ndarray:
    """Convert text to 384-dim vector using context-RI vectors.

    For each word: if a learned context vector exists, blend it with the
    hippocampus n-gram vector.  For unseen words, use pure hippocampus.
    """
    from .hippocampus import _text_to_vector as hippo_vector

    words = _tokenize(text)
    if not words:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    vector = np.zeros(EMBED_DIM, dtype=np.float32)

    for word, count in word_counts.items():
        weight = np.log1p(count)
        hippo_vec = hippo_vector(word)

        if word in word2idx:
            ctx_vec = vectors[word2idx[word]]
            blended = blend * ctx_vec + (1 - blend) * hippo_vec
        else:
            blended = hippo_vec

        vector += weight * blended

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


def _text_to_vector_context(text: str) -> np.ndarray:
    """Convert text to 384-dim vector via context-RI (with hippocampus fallback).

    If no trained context vectors are available, falls back entirely
    to hippocampus n-gram encoding.
    """
    from .hippocampus import _text_to_vector as hippo_vector

    learned = _get_context_ri_vectors()
    if learned is not None:
        return _text_to_vector_context_ri(text, learned[0], learned[1])
    return hippo_vector(text)


def context_to_frame(text: str, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert text to a 64x64 CA frame via context-window random indexing.

    Uses learned distributional context vectors when available,
    blended with hippocampus n-gram vectors.  Falls back to pure
    hippocampus if no context vectors have been trained.
    """
    vector = _text_to_vector_context(text)
    proj = _get_frame_matrix()

    frame_flat = vector @ proj
    frame_flat = np.tanh(frame_flat * 3.0)

    return frame_flat.reshape(size, size).astype(np.float32)


def context_to_frame_batch(
    texts: list[str], size: int = FRAME_SIZE
) -> list[np.ndarray]:
    """Batch-convert texts to CA frames via context-window random indexing."""
    vectors = np.stack([_text_to_vector_context(t) for t in texts])
    proj = _get_frame_matrix()

    frames_flat = vectors @ proj
    frames_flat = np.tanh(frames_flat * 3.0)

    return [
        frames_flat[i].reshape(size, size).astype(np.float32)
        for i in range(len(texts))
    ]

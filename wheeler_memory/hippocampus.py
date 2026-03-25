"""Hippocampus encoder: zero-dependency text-to-frame via character n-gram random indexing.

Replaces MiniLM (sentence-transformers) with a Wheeler-native encoder that
preserves lexical similarity without any external pre-trained model.

Pipeline:
  1. Text → character 3-grams and 4-grams
  2. Each n-gram → deterministic hash → row index into fixed projection matrix
  3. Accumulate weighted rows → 384-dim vector
  4. Project 384 → 4096 via fixed Gaussian random matrix (JL lemma)
  5. tanh(x * 3.0) → reshape to 64×64 float32

Similar text (shared words, morphological variants, overlapping phrases) produces
similar frames. True semantic similarity ("cat" vs "feline") is NOT captured —
that emerges from attractor dynamics and cortex layers, not the encoder.
"""

import hashlib
import struct

import numpy as np

# Projection dimensions — match embedding.py for compatibility
EMBED_DIM = 384
FRAME_SIZE = 64
FRAME_CELLS = FRAME_SIZE * FRAME_SIZE  # 4096

# Seeds for deterministic projection matrices
NGRAM_PROJECTION_SEED = 0xCAFE_BEEF
FRAME_PROJECTION_SEED = 0xDEAD_BEEF  # same as embedding.py JL matrix

# Hash space for n-gram → row mapping
NGRAM_HASH_SPACE = 2**16  # 65536 buckets — plenty for character n-grams

# Cached matrices (lazy-initialized)
_ngram_matrix: np.ndarray | None = None
_frame_matrix: np.ndarray | None = None


def _get_ngram_matrix() -> np.ndarray:
    """Fixed random matrix mapping n-gram hash buckets to 384-dim vectors.

    Shape: (NGRAM_HASH_SPACE, EMBED_DIM) = (65536, 384)
    Each row is a sparse-ish random direction seeded deterministically.
    """
    global _ngram_matrix
    if _ngram_matrix is None:
        rng = np.random.Generator(np.random.PCG64(NGRAM_PROJECTION_SEED))
        # Sparse random indexing: most entries near zero, some strong signals
        # This gives better discrimination than dense Gaussian
        raw = rng.standard_normal((NGRAM_HASH_SPACE, EMBED_DIM)).astype(np.float32)
        raw /= np.sqrt(EMBED_DIM)
        _ngram_matrix = raw
    return _ngram_matrix


def _get_frame_matrix() -> np.ndarray:
    """Fixed random matrix projecting 384-dim → 4096-dim (JL lemma).

    Same seed as embedding.py so the projection space is compatible.
    """
    global _frame_matrix
    if _frame_matrix is None:
        rng = np.random.Generator(np.random.PCG64(FRAME_PROJECTION_SEED))
        _frame_matrix = rng.standard_normal((EMBED_DIM, FRAME_CELLS)).astype(
            np.float32
        ) / np.sqrt(FRAME_CELLS)
    return _frame_matrix


def _ngram_hash(ngram: str) -> int:
    """Deterministic hash of an n-gram to a bucket index."""
    # Use first 4 bytes of SHA-1 for speed + uniformity
    digest = hashlib.sha1(ngram.encode("utf-8", errors="replace")).digest()
    return struct.unpack("<I", digest[:4])[0] % NGRAM_HASH_SPACE


def _extract_ngrams(text: str) -> dict[int, float]:
    """Extract weighted character 3-grams and 4-grams from text.

    Returns dict mapping hash bucket → accumulated weight.
    Weight = log(1 + count) for sublinear frequency scaling.
    """
    text = text.lower().strip()
    counts: dict[int, int] = {}

    # Character 3-grams
    for i in range(len(text) - 2):
        bucket = _ngram_hash(text[i : i + 3])
        counts[bucket] = counts.get(bucket, 0) + 1

    # Character 4-grams
    for i in range(len(text) - 3):
        bucket = _ngram_hash(text[i : i + 4])
        counts[bucket] = counts.get(bucket, 0) + 1

    # Sublinear weighting to prevent common n-grams from dominating
    return {bucket: np.log1p(count) for bucket, count in counts.items()}


def _text_to_vector(text: str) -> np.ndarray:
    """Convert text to a 384-dim vector via n-gram random indexing.

    Each n-gram maps to a row in the fixed projection matrix.
    Rows are accumulated with sublinear frequency weights.
    """
    ngram_weights = _extract_ngrams(text)
    if not ngram_weights:
        # Empty or very short text — return zero vector
        return np.zeros(EMBED_DIM, dtype=np.float32)

    matrix = _get_ngram_matrix()

    # Accumulate weighted rows
    buckets = np.array(list(ngram_weights.keys()), dtype=np.int64)
    weights = np.array(list(ngram_weights.values()), dtype=np.float32)

    # Vectorized: sum of (weight_i * matrix[bucket_i])
    vector = (weights[:, None] * matrix[buckets]).sum(axis=0)

    # L2 normalize to unit sphere for consistent projection behavior
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


def hippocampus_to_frame(text: str, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert text to a 64×64 CA frame via character n-gram random indexing.

    1. Text → 384-dim n-gram vector (deterministic, no external model)
    2. Project 384 → 4096 via fixed Gaussian random matrix
    3. tanh(x * 3.0) to map to (-1, 1) range
    4. Reshape to 64×64

    Similar text produces similar frames via shared n-gram overlap.
    """
    vector = _text_to_vector(text)
    proj = _get_frame_matrix()

    # Project: (384,) @ (384, 4096) → (4096,)
    frame_flat = vector @ proj

    # Scale so tanh spreads values across (-1, 1)
    frame_flat = np.tanh(frame_flat * 3.0)

    return frame_flat.reshape(size, size).astype(np.float32)


def hippocampus_to_frame_batch(
    texts: list[str], size: int = FRAME_SIZE
) -> list[np.ndarray]:
    """Batch-convert texts to CA frames via n-gram random indexing."""
    vectors = np.stack([_text_to_vector(t) for t in texts])  # (N, 384)
    proj = _get_frame_matrix()  # (384, 4096)

    frames_flat = vectors @ proj  # (N, 4096)
    frames_flat = np.tanh(frames_flat * 3.0)

    return [
        frames_flat[i].reshape(size, size).astype(np.float32) for i in range(len(texts))
    ]

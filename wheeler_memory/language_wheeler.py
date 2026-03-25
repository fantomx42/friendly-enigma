"""Language Wheeler: grammatical-structure encoder for deterministic 64×64 frames.

Syntactic complement to the Hippocampus (semantic).  Encodes POS tag sequences
as weighted sums of deterministic ternary spatial signatures so that sentences
with the same grammatical structure produce similar initial CA frames.

POS tagging is fully algorithmic — suffix heuristics + a baked-in closed-class
word list.  No dictionary file, no external data.
"""

import hashlib
import re

import numpy as np

# ---------------------------------------------------------------------------
# POS tag set
# ---------------------------------------------------------------------------
# Closed-class (function) words → POS.  ~80 entries, all lowercase.
_CLOSED_CLASS: dict[str, str] = {}

_CC_LISTS: dict[str, list[str]] = {
    "DET": [
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "some",
        "any",
        "no",
        "every",
        "each",
        "both",
        "all",
        "few",
        "many",
        "much",
        "several",
    ],
    "PREP": [
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "to",
        "of",
        "about",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "across",
        "along",
        "against",
        "among",
        "around",
        "behind",
        "beside",
        "beyond",
        "near",
        "off",
        "onto",
        "toward",
        "upon",
        "within",
        "without",
    ],
    "CONJ": [
        "and",
        "or",
        "but",
        "nor",
        "yet",
        "so",
        "because",
        "although",
        "while",
        "if",
        "when",
        "since",
        "unless",
        "until",
        "whereas",
        "whether",
    ],
    "PRON": [
        "i",
        "me",
        "you",
        "he",
        "him",
        "she",
        "her",
        "it",
        "we",
        "us",
        "they",
        "them",
        "who",
        "whom",
        "what",
        "which",
        "that",
        "this",
        "these",
        "those",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "themselves",
    ],
    "AUX": [
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    ],
}

for _pos, _words in _CC_LISTS.items():
    for _w in _words:
        _CLOSED_CLASS[_w] = _pos

# Suffix heuristics — applied in order, first match wins.
# Each entry: (suffix, POS tag)
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ly", "ADV"),
    ("tion", "NOUN"),
    ("sion", "NOUN"),
    ("ment", "NOUN"),
    ("ness", "NOUN"),
    ("ity", "NOUN"),
    ("ence", "NOUN"),
    ("ance", "NOUN"),
    ("ing", "VERB"),
    ("ed", "VERB"),
    ("ous", "ADJ"),
    ("ive", "ADJ"),
    ("ful", "ADJ"),
    ("less", "ADJ"),
    ("able", "ADJ"),
    ("ible", "ADJ"),
    ("ial", "ADJ"),
    ("al", "ADJ"),
    ("er", "NOUN"),
    ("or", "NOUN"),
    ("est", "ADJ"),
    ("s", "NOUN"),
]

# Punctuation POS tags
_PUNCT_MAP: dict[str, str] = {
    ".": "SENT_END",
    "!": "SENT_END",
    "?": "SENT_END",
    ",": "PAUSE",
    ";": "PAUSE",
    ":": "PAUSE",
    '"': "QUOTE",
    "'": "QUOTE",
}

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
# Splits on whitespace, separates leading/trailing punctuation as own tokens.
_TOKEN_RE = re.compile(r"""[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\s]""")


def tokenize(text: str) -> list[str]:
    """Split *text* into word and punctuation tokens.

    >>> tokenize("Hello, world!")
    ['Hello', ',', 'world', '!']
    """
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# POS classifier
# ---------------------------------------------------------------------------


def classify_pos(token: str) -> str:
    """Classify a single token into a POS tag.

    Resolution order:
      1. Punctuation map
      2. Closed-class word list (case-insensitive)
      3. Suffix heuristics (first match)
      4. Default → NOUN
    """
    # Punctuation
    if token in _PUNCT_MAP:
        return _PUNCT_MAP[token]

    low = token.lower()

    # Closed-class lookup
    cc = _CLOSED_CLASS.get(low)
    if cc is not None:
        return cc

    # Suffix rules — first match wins
    for suffix, pos in _SUFFIX_RULES:
        if low.endswith(suffix) and len(low) > len(suffix):
            return pos

    return "NOUN"


# ---------------------------------------------------------------------------
# POS spatial signatures (ternary, cached)
# ---------------------------------------------------------------------------
_pos_cache: dict[str, np.ndarray] = {}


def _pos_signature(pos: str, size: int = 64) -> np.ndarray:
    """Deterministic ternary *size*×*size* spatial pattern for a POS tag.

    Values are {-1, 0, +1} — matching the CA's three cell roles:
      +1 = local max, -1 = local min, 0 = slope.
    """
    key = f"{pos}:{size}"
    cached = _pos_cache.get(key)
    if cached is not None:
        return cached

    seed = int(hashlib.sha256(pos.encode()).hexdigest()[:8], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    sig = rng.integers(-1, 2, size=(size, size)).astype(np.float32)
    _pos_cache[key] = sig
    return sig


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------
_POSITIONAL_DECAY = 0.15


def language_to_frame(text: str, size: int = 64) -> np.ndarray:
    """Encode grammatical structure of *text* as a deterministic frame.

    Returns a float32 array of shape (*size*, *size*) with values in (-1, 1).
    """
    tokens = tokenize(text)
    frame = np.zeros((size, size), dtype=np.float32)
    for i, tok in enumerate(tokens):
        pos = classify_pos(tok)
        sig = _pos_signature(pos, size)
        weight = 1.0 / (1.0 + i * _POSITIONAL_DECAY)
        frame += sig * weight
    # Normalize to (-1, 1)
    peak = max(abs(float(frame.max())), abs(float(frame.min())))
    if peak > 0:
        frame /= peak
    return frame


def language_to_frame_batch(texts: list[str], size: int = 64) -> list[np.ndarray]:
    """Encode a batch of texts. Convenience wrapper."""
    return [language_to_frame(t, size) for t in texts]

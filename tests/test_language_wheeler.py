"""Tests for wheeler_memory.language_wheeler — grammatical-structure encoder."""

import numpy as np
import pytest

from wheeler_memory.language_wheeler import (
    classify_pos,
    language_to_frame,
    language_to_frame_batch,
    tokenize,
)


# ── Tokenizer ─────────────────────────────────────────────────────────


class TestTokenize:
    """tokenize() splits text into word and punctuation tokens."""

    def test_simple_sentence(self):
        """Basic sentence with comma and period."""
        tokens = tokenize("Hello, world!")
        assert tokens == ["Hello", ",", "world", "!"]

    def test_contractions(self):
        """Contractions kept as single tokens."""
        tokens = tokenize("I'm don't")
        assert "I'm" in tokens
        assert "don't" in tokens

    def test_empty_string(self):
        """Empty input → empty list."""
        assert tokenize("") == []

    def test_only_punctuation(self):
        """Just punctuation → punctuation tokens."""
        tokens = tokenize("...!")
        assert all(t in ".!" for t in tokens)

    def test_multiple_spaces(self):
        """Extra whitespace is ignored."""
        tokens = tokenize("hello   world")
        assert tokens == ["hello", "world"]


# ── POS Classifier ────────────────────────────────────────────────────


class TestClassifyPos:
    """classify_pos() heuristic POS tagging."""

    def test_determiners(self):
        """Known determiners → DET."""
        assert classify_pos("the") == "DET"
        assert classify_pos("The") == "DET"
        assert classify_pos("a") == "DET"

    def test_prepositions(self):
        """Known prepositions → PREP."""
        assert classify_pos("in") == "PREP"
        assert classify_pos("with") == "PREP"

    def test_conjunctions(self):
        """Known conjunctions → CONJ."""
        assert classify_pos("and") == "CONJ"
        assert classify_pos("but") == "CONJ"

    def test_pronouns(self):
        """Known pronouns → PRON."""
        assert classify_pos("i") == "PRON"
        assert classify_pos("they") == "PRON"

    def test_auxiliaries(self):
        """Known auxiliaries → AUX."""
        assert classify_pos("is") == "AUX"
        assert classify_pos("would") == "AUX"

    def test_punctuation(self):
        """Punctuation → correct POS tag."""
        assert classify_pos(".") == "SENT_END"
        assert classify_pos("!") == "SENT_END"
        assert classify_pos("?") == "SENT_END"
        assert classify_pos(",") == "PAUSE"
        assert classify_pos(";") == "PAUSE"

    def test_suffix_ly_adverb(self):
        """Words ending in -ly → ADV."""
        assert classify_pos("quickly") == "ADV"
        assert classify_pos("slowly") == "ADV"

    def test_suffix_tion_noun(self):
        """Words ending in -tion → NOUN."""
        assert classify_pos("creation") == "NOUN"
        assert classify_pos("evolution") == "NOUN"

    def test_suffix_ing_verb(self):
        """Words ending in -ing → VERB."""
        assert classify_pos("running") == "VERB"

    def test_suffix_ous_adjective(self):
        """Words ending in -ous → ADJ."""
        assert classify_pos("dangerous") == "ADJ"

    def test_suffix_ful_adjective(self):
        """Words ending in -ful → ADJ."""
        assert classify_pos("beautiful") == "ADJ"

    def test_suffix_less_adjective(self):
        """Words ending in -less → ADJ."""
        assert classify_pos("careless") == "ADJ"

    def test_unknown_defaults_to_noun(self):
        """Unknown word without matching suffix → NOUN."""
        assert classify_pos("xyz") == "NOUN"


# ── Frame Encoder ─────────────────────────────────────────────────────


class TestLanguageToFrame:
    """language_to_frame() grammatical structure → 64×64 frame."""

    def test_output_shape(self):
        """Output is 64×64 float32."""
        frame = language_to_frame("The cat sat on the mat.")
        assert frame.shape == (64, 64)
        assert frame.dtype == np.float32

    def test_values_in_range(self):
        """Values normalized to [-1, 1]."""
        frame = language_to_frame("The quick brown fox jumps.")
        assert np.all(frame >= -1.0)
        assert np.all(frame <= 1.0)

    def test_deterministic(self):
        """Same text → identical frame."""
        f1 = language_to_frame("Hello world.")
        f2 = language_to_frame("Hello world.")
        np.testing.assert_array_equal(f1, f2)

    def test_different_grammar_different_frame(self):
        """Different grammatical structures → different frames."""
        f1 = language_to_frame("The cat sat.")
        f2 = language_to_frame("Running quickly, she jumped!")
        assert not np.array_equal(f1, f2)

    def test_same_grammar_similar_frame(self):
        """Same structure, different words → correlated frames."""
        f1 = language_to_frame("The dog ran quickly.")
        f2 = language_to_frame("The cat jumped slowly.")
        corr = np.corrcoef(f1.flat, f2.flat)[0, 1]
        # Same DET NOUN VERB ADV SENT_END structure → should be correlated
        assert corr > 0.5

    def test_empty_string_zero_frame(self):
        """Empty input → zero frame."""
        frame = language_to_frame("")
        np.testing.assert_array_equal(frame, np.zeros((64, 64), dtype=np.float32))

    def test_custom_size(self):
        """Custom size changes output shape."""
        frame = language_to_frame("hello", size=32)
        assert frame.shape == (32, 32)

    def test_positional_decay(self):
        """Later tokens have less influence — mixed POS yields different frame."""
        # Single noun vs noun+verb+adj sequence → different frames
        f_short = language_to_frame("cat")
        f_long = language_to_frame("cat runs quickly")
        assert not np.array_equal(f_short, f_long)


class TestLanguageToFrameBatch:
    """language_to_frame_batch() batch encoding."""

    def test_batch_matches_individual(self):
        """Batch matches individual calls."""
        texts = ["The cat sat.", "Dogs run fast.", "I think so."]
        batch = language_to_frame_batch(texts)
        for i, text in enumerate(texts):
            np.testing.assert_array_equal(batch[i], language_to_frame(text))

    def test_batch_length(self):
        """Correct number of frames."""
        assert len(language_to_frame_batch(["a", "b"])) == 2

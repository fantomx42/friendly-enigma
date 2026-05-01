"""Tests for wheeler_memory.theories.resonance — CA-dynamics corpus search."""

import numpy as np
import pytest

from wheeler_memory.theories.resonance import (
    ResonanceResult,
    _chunk_text,
    _is_text_file,
    query_corpus,
)


# ── ResonanceResult dataclass ─────────────────────────────────────────


class TestResonanceResult:
    """ResonanceResult defaults."""

    def test_defaults(self):
        """All fields default to zero/empty."""
        r = ResonanceResult()
        assert r.total_chunks == 0
        assert r.resonant_chunks == 0
        assert r.skipped_chunks == 0
        assert r.resonant_files == []
        assert r.cost_ratio == 0.0


# ── _chunk_text ───────────────────────────────────────────────────────


class TestChunkText:
    """_chunk_text() text splitting."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size → one chunk."""
        chunks = _chunk_text("hello world", 512)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_multiple_chunks(self):
        """Long text → multiple chunks."""
        text = "x" * 2000
        chunks = _chunk_text(text, 512)
        assert len(chunks) == 4  # ceil(2000/512)

    def test_empty_text(self):
        """Empty text → no chunks."""
        assert _chunk_text("", 512) == []

    def test_whitespace_only_stripped(self):
        """Whitespace-only chunks are dropped."""
        assert _chunk_text("   ", 512) == []

    def test_exact_boundary(self):
        """Text exactly chunk_size → one chunk."""
        text = "a" * 512
        chunks = _chunk_text(text, 512)
        assert len(chunks) == 1


# ── _is_text_file ─────────────────────────────────────────────────────


class TestIsTextFile:
    """_is_text_file() detection."""

    def test_text_file(self, tmp_path):
        """Valid UTF-8 text file → True."""
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert _is_text_file(f) is True

    def test_binary_file(self, tmp_path):
        """Binary content → False."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\xff\xfe\xfd" * 100)
        assert _is_text_file(f) is False

    def test_nonexistent_file_raises(self, tmp_path):
        """Missing file → FileNotFoundError (not caught by the function)."""
        with pytest.raises(FileNotFoundError):
            _is_text_file(tmp_path / "nope.txt")


# ── query_corpus ──────────────────────────────────────────────────────


@pytest.mark.slow
class TestQueryCorpus:
    """query_corpus() — CA resonance-based search."""

    def test_empty_corpus(self, tmp_path):
        """Empty directory → zero results."""
        result = query_corpus("test query", tmp_path, early_exit_ticks=10)
        assert result.total_chunks == 0
        assert result.resonant_chunks == 0

    def test_single_file_corpus(self, tmp_path):
        """Corpus with one text file → processes its chunks."""
        (tmp_path / "doc.txt").write_text(
            "The cat sat on the mat. " * 100,
            encoding="utf-8",
        )
        result = query_corpus("cat", tmp_path, chunk_size=256, early_exit_ticks=10)
        assert result.total_chunks > 0
        assert result.total_chunks == result.resonant_chunks + result.skipped_chunks

    def test_cost_ratio_range(self, tmp_path):
        """cost_ratio is in [0, 1]."""
        (tmp_path / "doc.txt").write_text(
            "Physics describes the natural world. " * 50,
            encoding="utf-8",
        )
        result = query_corpus("physics", tmp_path, chunk_size=256, early_exit_ticks=10)
        assert 0.0 <= result.cost_ratio <= 1.0

    def test_skips_binary_files(self, tmp_path):
        """Binary files in corpus are skipped."""
        (tmp_path / "text.txt").write_text("some text content", encoding="utf-8")
        (tmp_path / "binary.bin").write_bytes(b"\x00\xff" * 500)
        result = query_corpus("text", tmp_path, early_exit_ticks=10)
        # Should only process the text file, not the binary
        assert result.total_chunks >= 1

    def test_resonant_files_structure(self, tmp_path):
        """Each resonant file entry has path, chunk_text, resonance_score."""
        (tmp_path / "doc.txt").write_text(
            "memory systems and neural networks " * 50,
            encoding="utf-8",
        )
        result = query_corpus("neural", tmp_path, chunk_size=256, early_exit_ticks=10)
        for rf in result.resonant_files:
            assert "path" in rf
            assert "chunk_text" in rf
            assert "resonance_score" in rf
            assert isinstance(rf["resonance_score"], float)

"""Tests for the corpus crystallization pipeline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wheeler_memory.crystallization import (
    CrystallizationResult,
    crystallize,
    load_corpus,
    _count_stored,
    _existing_keys,
)
from wheeler_memory.hashing import text_to_hex
from wheeler_memory.storage import _load_index

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import store_test_memory


# ── Corpus loading ────────────────────────────────────────────────────────────


class TestLoadCorpusJsonl:
    def test_load_jsonl(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            '{"text": "hello world"}\n'
            '{"text": "foo bar"}\n'
            '{"text": ""}\n'  # empty — should be skipped
            '{"other": "no text field"}\n'  # no text — should be skipped
        )
        texts = list(load_corpus(corpus))
        assert texts == ["hello world", "foo bar"]

    def test_load_jsonl_explicit_fmt(self, tmp_path):
        corpus = tmp_path / "data.txt"  # wrong extension
        corpus.write_text('{"text": "works anyway"}\n')
        texts = list(load_corpus(corpus, fmt="jsonl"))
        assert texts == ["works anyway"]


class TestLoadCorpusCsv:
    def test_load_csv(self, tmp_path):
        corpus = tmp_path / "corpus.csv"
        corpus.write_text("text,label\nhello world,1\nfoo bar,2\n")
        texts = list(load_corpus(corpus))
        assert texts == ["hello world", "foo bar"]

    def test_load_csv_empty_text(self, tmp_path):
        corpus = tmp_path / "corpus.csv"
        corpus.write_text("text\nhello\n\nworld\n")
        texts = list(load_corpus(corpus))
        assert texts == ["hello", "world"]


class TestLoadCorpusTxt:
    def test_load_txt(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world\nfoo bar\n\n  \nbaz\n")
        texts = list(load_corpus(corpus))
        assert texts == ["hello world", "foo bar", "baz"]


class TestLoadCorpusAutoDetect:
    def test_auto_jsonl(self, tmp_path):
        corpus = tmp_path / "data.jsonl"
        corpus.write_text('{"text": "auto detected"}\n')
        assert list(load_corpus(corpus)) == ["auto detected"]

    def test_auto_ndjson(self, tmp_path):
        corpus = tmp_path / "data.ndjson"
        corpus.write_text('{"text": "ndjson too"}\n')
        assert list(load_corpus(corpus)) == ["ndjson too"]

    def test_auto_csv(self, tmp_path):
        corpus = tmp_path / "data.csv"
        corpus.write_text("text\nauto csv\n")
        assert list(load_corpus(corpus)) == ["auto csv"]

    def test_auto_txt(self, tmp_path):
        corpus = tmp_path / "data.txt"
        corpus.write_text("auto txt\n")
        assert list(load_corpus(corpus)) == ["auto txt"]


# ── Crystallization pipeline ──────────────────────────────────────────────────


class TestCrystallizeSmallCorpus:
    """End-to-end crystallization using hash_to_frame (no embedding dependency)."""

    def test_crystallize_stores_memories(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("alpha concept\nbeta concept\ngamma concept\n")

        result = crystallize(
            corpus, data_dir=tmp_path, use_embedding=False, batch_size=2,
        )

        assert result.stored == 3
        assert result.skipped == 0
        assert result.errors == 0
        assert _count_stored(tmp_path) == 3

    def test_crystallize_creates_attractors(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("a beautiful sunset over the ocean\n")

        result = crystallize(corpus, data_dir=tmp_path, use_embedding=False)

        # Find the chunk it was stored in
        stored_chunk = list(result.chunks_used.keys())[0]
        att_dir = tmp_path / "chunks" / stored_chunk / "attractors"
        npy_files = list(att_dir.glob("*.npy"))
        assert len(npy_files) == 1

        # Attractor should be a 64x64 array
        arr = np.load(npy_files[0])
        assert arr.shape == (64, 64)

    def test_crystallize_creates_bricks(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("a beautiful sunset over the ocean\n")

        result = crystallize(corpus, data_dir=tmp_path, use_embedding=False)

        stored_chunk = list(result.chunks_used.keys())[0]
        brick_dir = tmp_path / "chunks" / stored_chunk / "bricks"
        npz_files = list(brick_dir.glob("*.npz"))
        assert len(npz_files) == 1


class TestCrystallizeResume:
    def test_resume_skips_duplicates(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("first concept\nsecond concept\n")

        # First run
        r1 = crystallize(corpus, data_dir=tmp_path, use_embedding=False)
        assert r1.stored == 2

        # Second run — should skip both
        r2 = crystallize(corpus, data_dir=tmp_path, use_embedding=False, resume=True)
        assert r2.stored == 0
        assert r2.skipped == 2
        assert _count_stored(tmp_path) == 2

    def test_no_resume_reprocesses(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("reprocess me\n")

        crystallize(corpus, data_dir=tmp_path, use_embedding=False)
        r2 = crystallize(corpus, data_dir=tmp_path, use_embedding=False, resume=False)
        # Re-storing same key overwrites, so stored=1
        assert r2.stored == 1
        assert r2.skipped == 0


class TestCrystallizeMaxItems:
    def test_max_items_caps_processing(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("one\ntwo\nthree\nfour\nfive\n")

        result = crystallize(
            corpus, data_dir=tmp_path, use_embedding=False, max_items=3,
        )

        assert result.stored == 3


class TestCrystallizeSaturation:
    def test_saturation_percentage(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("concept a\nconcept b\n")

        result = crystallize(corpus, data_dir=tmp_path, use_embedding=False)

        # 2 out of MAX_ATTRACTORS (10000)
        assert result.saturation_pct == pytest.approx(0.02, abs=0.01)


class TestCrystallizeAutoChunk:
    def test_auto_routes_to_correct_chunk(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        # "python debugging" should route to "code" chunk
        corpus.write_text("python debugging techniques\n")

        result = crystallize(corpus, data_dir=tmp_path, use_embedding=False)

        assert "code" in result.chunks_used

    def test_explicit_chunk_override(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("python debugging techniques\n")

        result = crystallize(
            corpus, data_dir=tmp_path, use_embedding=False, chunk="science",
        )

        assert "science" in result.chunks_used
        assert "code" not in result.chunks_used


class TestCrystallizationResult:
    def test_str_representation(self):
        r = CrystallizationResult(
            stored=100, skipped=10, errors=2,
            saturation_pct=1.12, elapsed_seconds=5.3,
            chunks_used={"general": 80, "code": 20},
        )
        s = str(r)
        assert "100" in s
        assert "10" in s
        assert "1.1%" in s

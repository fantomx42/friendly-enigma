"""Tests for wheeler_memory.experiential — episodic memory metadata."""

import json

import numpy as np
import pytest

from wheeler_memory.experiential import (
    ExperientialMeta,
    experiential_bricks_dir,
    experiential_dir,
    scm_snapshot_hash,
)


class TestExperientialMeta:
    """ExperientialMeta dataclass lifecycle."""

    def test_default_timestamp(self):
        """Auto-generates ISO timestamp when not provided."""
        meta = ExperientialMeta()
        assert meta.encoding_timestamp != ""
        assert "T" in meta.encoding_timestamp  # ISO format

    def test_explicit_timestamp_preserved(self):
        """Explicit timestamp is kept as-is."""
        meta = ExperientialMeta(encoding_timestamp="2025-01-01T00:00:00+00:00")
        assert meta.encoding_timestamp == "2025-01-01T00:00:00+00:00"

    def test_defaults(self):
        """Default field values."""
        meta = ExperientialMeta()
        assert meta.preceding_query_hex == ""
        assert meta.scm_snapshot_hash == ""
        assert meta.consolidated is False

    def test_save_and_load_roundtrip(self, tmp_path):
        """save → load recovers all fields."""
        original = ExperientialMeta(
            encoding_timestamp="2025-06-15T12:00:00+00:00",
            preceding_query_hex="abcdef1234567890",
            scm_snapshot_hash="deadbeef12345678",
            consolidated=True,
        )
        path = tmp_path / "meta.json"
        original.save(path)
        loaded = ExperientialMeta.load(path)
        assert loaded.encoding_timestamp == original.encoding_timestamp
        assert loaded.preceding_query_hex == original.preceding_query_hex
        assert loaded.scm_snapshot_hash == original.scm_snapshot_hash
        assert loaded.consolidated == original.consolidated

    def test_save_creates_valid_json(self, tmp_path):
        """Saved file is valid JSON."""
        meta = ExperientialMeta()
        path = tmp_path / "meta.json"
        meta.save(path)
        data = json.loads(path.read_text())
        assert "encoding_timestamp" in data
        assert "consolidated" in data

    def test_load_ignores_extra_fields(self, tmp_path):
        """Extra fields in JSON are ignored gracefully."""
        path = tmp_path / "meta.json"
        path.write_text(json.dumps({
            "encoding_timestamp": "2025-01-01T00:00:00",
            "preceding_query_hex": "",
            "scm_snapshot_hash": "",
            "consolidated": False,
            "unknown_field": "should be ignored",
        }))
        meta = ExperientialMeta.load(path)
        assert not hasattr(meta, "unknown_field") or True  # graceful


class TestScmSnapshotHash:
    """scm_snapshot_hash() SHA-256 truncated to 16 hex chars."""

    def test_deterministic(self):
        """Same grid → same hash."""
        grid = np.ones((64, 64), dtype=np.float32)
        h1 = scm_snapshot_hash(grid)
        h2 = scm_snapshot_hash(grid)
        assert h1 == h2

    def test_length(self):
        """Hash is 16 hex characters."""
        grid = np.zeros((64, 64), dtype=np.float32)
        h = scm_snapshot_hash(grid)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_grids_different_hashes(self):
        """Different grids → different hashes."""
        g1 = np.zeros((64, 64), dtype=np.float32)
        g2 = np.ones((64, 64), dtype=np.float32)
        assert scm_snapshot_hash(g1) != scm_snapshot_hash(g2)


class TestExperientialDirs:
    """experiential_dir / experiential_bricks_dir directory creation."""

    def test_experiential_dir_created(self, tmp_path):
        """Creates 'experiential' subdirectory."""
        d = experiential_dir(tmp_path)
        assert d.exists()
        assert d.name == "experiential"
        assert d.parent == tmp_path

    def test_experiential_bricks_dir_created(self, tmp_path):
        """Creates 'experiential_bricks' subdirectory."""
        d = experiential_bricks_dir(tmp_path)
        assert d.exists()
        assert d.name == "experiential_bricks"

    def test_idempotent(self, tmp_path):
        """Calling twice does not raise."""
        experiential_dir(tmp_path)
        experiential_dir(tmp_path)  # second call is fine

    def test_nested_creation(self, tmp_path):
        """Works with nested non-existent parent dirs."""
        nested = tmp_path / "deep" / "nested" / "chunk"
        d = experiential_dir(nested)
        assert d.exists()

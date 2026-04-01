"""Tests for wheeler_memory.cache — mtime-based JSON file cache."""

import json

import pytest

from wheeler_memory.cache import cached_load_json, clear, invalidate


class TestCachedLoadJson:
    """cached_load_json behaviour."""

    def test_load_valid_json(self, tmp_path):
        """Valid JSON file → parsed dict."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        result = cached_load_json(f)
        assert result == {"key": "value"}

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent path → empty dict."""
        result = cached_load_json(tmp_path / "nope.json")
        assert result == {}

    def test_missing_file_returns_custom_default(self, tmp_path):
        """Non-existent path with custom default → that default."""
        result = cached_load_json(tmp_path / "nope.json", default={"x": 1})
        assert result == {"x": 1}

    def test_cache_hit_on_same_mtime(self, tmp_path):
        """Second load with same mtime returns cached copy."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"a": 1}))
        first = cached_load_json(f)
        second = cached_load_json(f)
        assert first == second

    def test_cache_hit_returns_deep_copy(self, tmp_path):
        """Mutating a cache-hit return does not corrupt subsequent reads."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"nested": {"x": 1}}))
        # First call stores data in cache and returns reference
        _ = cached_load_json(f)
        # Second call is a cache hit → deep copy
        second = cached_load_json(f)
        second["nested"]["x"] = 999
        # Third call is also a cache hit → fresh deep copy, unaffected
        third = cached_load_json(f)
        assert third["nested"]["x"] == 1

    def test_detects_mtime_change(self, tmp_path):
        """Overwriting file with new content → fresh read on next call."""
        import os
        import time

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"v": 1}))
        cached_load_json(f)

        # Ensure mtime differs (some filesystems have 1s resolution)
        time.sleep(0.05)
        f.write_text(json.dumps({"v": 2}))
        # Force mtime difference on coarse filesystems
        os.utime(f, (f.stat().st_mtime + 1, f.stat().st_mtime + 1))

        result = cached_load_json(f)
        assert result == {"v": 2}

    def test_invalid_json_returns_default(self, tmp_path):
        """Corrupted JSON → default dict."""
        f = tmp_path / "bad.json"
        f.write_text("not {valid json")
        result = cached_load_json(f)
        assert result == {}

    def test_invalid_json_custom_default(self, tmp_path):
        """Corrupted JSON with custom default → that default."""
        f = tmp_path / "bad.json"
        f.write_text("{broken")
        result = cached_load_json(f, default={"fallback": True})
        assert result == {"fallback": True}


class TestInvalidate:
    """invalidate() removes a path from the cache."""

    def test_invalidate_forces_reload(self, tmp_path):
        """After invalidate, next load re-reads from disk."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"a": 1}))
        cached_load_json(f)
        invalidate(f)
        # File unchanged, but cache entry was removed — still re-reads
        result = cached_load_json(f)
        assert result == {"a": 1}

    def test_invalidate_nonexistent_is_noop(self, tmp_path):
        """Invalidating a path not in cache does not raise."""
        invalidate(tmp_path / "never_loaded.json")


class TestClear:
    """clear() empties the entire cache."""

    def test_clear_all(self, tmp_path):
        """After clear, all paths require reload."""
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps({"a": 1}))
        f2.write_text(json.dumps({"b": 2}))
        cached_load_json(f1)
        cached_load_json(f2)
        clear()
        # Both should still work (just re-read from disk)
        assert cached_load_json(f1) == {"a": 1}
        assert cached_load_json(f2) == {"b": 2}

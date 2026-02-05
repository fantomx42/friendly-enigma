"""
Tests for wheeler_weights.py - SCM Stability Metrics

Tests the StabilityTracker and PatternMetrics classes that implement
the Symbolic Collapse Model's stability scoring.
"""

import json
import os
import sys
import tempfile
import time

import pytest

# Ensure ralph_core is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ralph_core"))

from wheeler_weights import PatternMetrics, StabilityTracker, STABILITY_DB_FILE


class TestPatternMetrics:
    """Test stability_score computation."""

    def test_empty_pattern_scores_zero(self):
        m = PatternMetrics(pattern_id="test-empty")
        assert m.stability_score == 0.0

    def test_hit_count_increases_score(self):
        m = PatternMetrics(pattern_id="test-hits", hit_count=10)
        assert m.stability_score > 0.0
        # With only hits (no persistence or compression), max is 0.40
        assert m.stability_score <= 0.40

    def test_compression_survival_adds_025(self):
        m = PatternMetrics(pattern_id="test-compress", compression_survived=True)
        assert m.stability_score == pytest.approx(0.25, abs=0.01)

    def test_full_stability_near_one(self):
        m = PatternMetrics(
            pattern_id="test-full",
            hit_count=50,
            frame_persistence=20,
            context_switches_seen=20,
            compression_survived=True,
        )
        # Should be close to 1.0 (all dimensions maxed)
        assert m.stability_score >= 0.90

    def test_frame_persistence_ratio(self):
        # Half survival rate
        m = PatternMetrics(
            pattern_id="test-persist",
            frame_persistence=5,
            context_switches_seen=10,
        )
        # persist_score = 0.5, weighted at 0.35 => 0.175
        assert m.stability_score == pytest.approx(0.175, abs=0.01)

    def test_score_bounded_zero_to_one(self):
        # Extreme values should still be bounded
        m = PatternMetrics(
            pattern_id="test-extreme",
            hit_count=10000,
            frame_persistence=10000,
            context_switches_seen=1,
            compression_survived=True,
        )
        assert 0.0 <= m.stability_score <= 1.0


class TestStabilityTracker:
    """Test the StabilityTracker singleton behavior."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton for each test."""
        StabilityTracker._instance = None
        yield
        StabilityTracker._instance = None

    def test_record_hit_creates_pattern(self):
        tracker = StabilityTracker()
        metrics = tracker.record_hit("p1", "hello world")
        assert metrics.hit_count == 1
        assert metrics.text_preview == "hello world"

    def test_record_hit_increments(self):
        tracker = StabilityTracker()
        tracker.record_hit("p1", "hello")
        tracker.record_hit("p1", "hello")
        metrics = tracker.record_hit("p1", "hello")
        assert metrics.hit_count == 3

    def test_context_switch_increments_all(self):
        tracker = StabilityTracker()
        tracker.record_hit("p1", "first")
        tracker.record_hit("p2", "second")
        tracker.record_context_switch()

        m1 = tracker.get_metrics("p1")
        m2 = tracker.get_metrics("p2")
        assert m1.context_switches_seen == 1
        assert m2.context_switches_seen == 1

    def test_context_switch_persistence_for_recent(self):
        tracker = StabilityTracker()
        # Record a hit (will be recent)
        tracker.record_hit("p1", "recent pattern")
        # Record context switch - p1 was just accessed so should persist
        tracker.record_context_switch()
        m = tracker.get_metrics("p1")
        assert m.frame_persistence == 1

    def test_compression_survival(self):
        tracker = StabilityTracker()
        tracker.record_hit("p1", "pattern")
        tracker.record_compression_survival("p1")
        m = tracker.get_metrics("p1")
        assert m.compression_survived is True

    def test_get_stability_unknown_returns_zero(self):
        tracker = StabilityTracker()
        assert tracker.get_stability("nonexistent") == 0.0

    def test_get_all_scored_sorted(self):
        tracker = StabilityTracker()
        # p1: more hits = higher score
        for _ in range(10):
            tracker.record_hit("p1", "frequent")
        tracker.record_hit("p2", "rare")

        scored = tracker.get_all_scored()
        assert len(scored) == 2
        assert scored[0]["pattern_id"] == "p1"
        assert scored[0]["stability_score"] > scored[1]["stability_score"]

    def test_flush_and_reload(self, tmp_path):
        """Test persistence to disk."""
        import wheeler_weights

        # Override the DB file path to tmp
        original_path = wheeler_weights.STABILITY_DB_FILE
        test_db = str(tmp_path / "test_stability.json")
        wheeler_weights.STABILITY_DB_FILE = test_db

        try:
            tracker = StabilityTracker()
            tracker.record_hit("persist-test", "will this survive?")
            tracker.record_hit("persist-test", "second hit")
            tracker.flush()

            assert os.path.exists(test_db)

            with open(test_db) as f:
                data = json.load(f)

            assert data["version"] == 1
            patterns = data["patterns"]
            assert len(patterns) == 1
            assert patterns[0]["pattern_id"] == "persist-test"
            assert patterns[0]["hit_count"] == 2
        finally:
            wheeler_weights.STABILITY_DB_FILE = original_path

"""
Tests for context_budget.py - Stability-Weighted Token Allocation

Tests that the context budget system correctly prioritizes
high-stability patterns and respects token limits.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ralph_core"))

from context_budget import (
    WeightedPattern,
    get_weighted_context,
    rank_patterns,
    budget_summary,
    estimate_tokens,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min of 1

    def test_single_word(self):
        assert estimate_tokens("hello") == 1

    def test_sentence(self):
        tokens = estimate_tokens("the quick brown fox jumps over the lazy dog")
        # 9 words * 1.3 ~= 11-12
        assert 10 <= tokens <= 15


class TestGetWeightedContext:
    def test_empty_patterns(self):
        result = get_weighted_context([], max_tokens=1000)
        assert result == ""

    def test_single_high_stability(self):
        patterns = [
            WeightedPattern(text="important pattern", stability_score=0.9, pattern_id="p1")
        ]
        result = get_weighted_context(patterns, max_tokens=100)
        assert "0.90" in result
        assert "important pattern" in result
        assert "WHEELER CONTEXT" in result

    def test_low_stability_dropped(self):
        patterns = [
            WeightedPattern(text="garbage pattern", stability_score=0.01, pattern_id="p1")
        ]
        result = get_weighted_context(patterns, max_tokens=100)
        assert result == ""  # Below min_stability (0.05)

    def test_ordering_by_stability(self):
        patterns = [
            WeightedPattern(text="low pattern", stability_score=0.1, pattern_id="low"),
            WeightedPattern(text="high pattern", stability_score=0.9, pattern_id="high"),
            WeightedPattern(text="mid pattern", stability_score=0.5, pattern_id="mid"),
        ]
        result = get_weighted_context(patterns, max_tokens=500)
        # High should appear before mid, mid before low
        high_pos = result.find("high pattern")
        mid_pos = result.find("mid pattern")
        assert high_pos < mid_pos

    def test_token_budget_respected(self):
        # Create a pattern that would use ~100 tokens
        long_text = " ".join(["word"] * 100)
        patterns = [
            WeightedPattern(text=long_text, stability_score=0.9, pattern_id="p1")
        ]
        result = get_weighted_context(patterns, max_tokens=20)
        # Should be truncated
        assert "..." in result

    def test_medium_stability_proportional(self):
        patterns = [
            WeightedPattern(text="medium pattern one", stability_score=0.5, pattern_id="m1"),
            WeightedPattern(text="medium pattern two", stability_score=0.4, pattern_id="m2"),
        ]
        result = get_weighted_context(patterns, max_tokens=200)
        assert "medium pattern one" in result

    def test_low_stability_compressed(self):
        patterns = [
            WeightedPattern(
                text="This is a low stability pattern. It has multiple sentences. Only the first should appear.",
                stability_score=0.15,
                pattern_id="low",
            )
        ]
        result = get_weighted_context(patterns, max_tokens=200)
        if result:  # May be included as compressed
            assert "compressed" in result

    def test_min_stability_filter(self):
        patterns = [
            WeightedPattern(text="keep me", stability_score=0.5, pattern_id="keep"),
            WeightedPattern(text="drop me", stability_score=0.02, pattern_id="drop"),
        ]
        result = get_weighted_context(patterns, max_tokens=500, min_stability=0.1)
        assert "keep me" in result
        assert "drop me" not in result


class TestRankPatterns:
    def test_sorts_descending(self):
        patterns = [
            WeightedPattern(text="a", stability_score=0.3, pattern_id="a"),
            WeightedPattern(text="c", stability_score=0.9, pattern_id="c"),
            WeightedPattern(text="b", stability_score=0.6, pattern_id="b"),
        ]
        ranked = rank_patterns(patterns)
        assert ranked[0].pattern_id == "c"
        assert ranked[1].pattern_id == "b"
        assert ranked[2].pattern_id == "a"


class TestBudgetSummary:
    def test_tier_classification(self):
        patterns = [
            WeightedPattern(text="high", stability_score=0.8, pattern_id="h"),
            WeightedPattern(text="medium", stability_score=0.5, pattern_id="m"),
            WeightedPattern(text="low", stability_score=0.1, pattern_id="l"),
            WeightedPattern(text="drop", stability_score=0.01, pattern_id="d"),
        ]
        summary = budget_summary(patterns, max_tokens=1000)
        assert summary["total_patterns"] == 4
        assert summary["tiers"]["high"] == 1
        assert summary["tiers"]["medium"] == 1
        assert summary["tiers"]["low"] == 1
        assert summary["tiers"]["dropped"] == 1

"""
context_budget.py - Stability-Weighted Context Window Allocation

When building the LLM context window, token budget is finite.
This module uses Wheeler stability scores (from wheeler_weights.py)
to allocate tokens proportionally:

  - High-stability patterns get full token budget
  - Medium-stability patterns get proportional allocation
  - Low-stability patterns are compressed or dropped

Implements the Symbolic Collapse Model principle:
  "Meaning is what survives symbolic pressure."

Token constraints *are* the symbolic pressure. What survives
the budget cut is what has demonstrated meaning.

See /docs/SCM_AXIOMS.md for theoretical foundation.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WeightedPattern:
    """A Wheeler memory pattern with its stability score."""

    text: str
    stability_score: float
    pattern_id: str = ""
    source: str = ""  # "wheeler", "vector_db", "lesson", "guideline"


# Stability tiers matching SCM axioms
TIER_HIGH = 0.7    # Full allocation - survived pressure
TIER_MEDIUM = 0.3  # Proportional allocation - promising but untested
# Below TIER_MEDIUM: compressed or dropped


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate. ~1.3 tokens per whitespace-delimited word
    for English text processed by most LLM tokenizers.
    """
    return max(1, int(len(text.split()) * 1.3))


def get_weighted_context(
    patterns: List[WeightedPattern],
    max_tokens: int,
    min_stability: float = 0.05,
) -> str:
    """
    Build a context string from patterns, weighted by stability score.

    High-stability patterns are included first and get more token budget.
    Low-stability patterns are compressed (truncated) or dropped entirely.

    Args:
        patterns: List of WeightedPattern with text and stability_score
        max_tokens: Maximum token budget for the assembled context
        min_stability: Floor below which patterns are dropped entirely

    Returns:
        Assembled context string ready for LLM prompt injection
    """
    if not patterns:
        return ""

    # Filter out patterns below minimum stability
    viable = [p for p in patterns if p.stability_score >= min_stability]

    # Sort by stability descending - high stability patterns get priority
    viable.sort(key=lambda p: p.stability_score, reverse=True)

    # Calculate total stability weight for proportional allocation
    total_weight = sum(p.stability_score for p in viable)
    if total_weight == 0:
        return ""

    context_parts: List[str] = []
    tokens_used = 0

    for pattern in viable:
        if tokens_used >= max_tokens:
            break

        pattern_tokens = estimate_tokens(pattern.text)

        if pattern.stability_score >= TIER_HIGH:
            # High stability: include in full up to remaining budget
            available = max_tokens - tokens_used
            if pattern_tokens <= available:
                context_parts.append(
                    f"[{pattern.stability_score:.2f}] {pattern.text}"
                )
                tokens_used += pattern_tokens
            else:
                # Truncate to fit remaining budget
                truncated = _truncate_to_tokens(pattern.text, available - 5)
                if truncated:
                    context_parts.append(
                        f"[{pattern.stability_score:.2f}] {truncated}..."
                    )
                    tokens_used += estimate_tokens(truncated) + 2
                break

        elif pattern.stability_score >= TIER_MEDIUM:
            # Medium stability: proportional allocation
            proportion = pattern.stability_score / total_weight
            token_budget = int(max_tokens * proportion)
            available = min(token_budget, max_tokens - tokens_used)

            if available <= 0:
                continue

            if pattern_tokens <= available:
                context_parts.append(
                    f"[{pattern.stability_score:.2f}] {pattern.text}"
                )
                tokens_used += pattern_tokens
            else:
                truncated = _truncate_to_tokens(pattern.text, available - 5)
                if truncated:
                    context_parts.append(
                        f"[{pattern.stability_score:.2f}] {truncated}..."
                    )
                    tokens_used += estimate_tokens(truncated) + 2

        else:
            # Low stability: compress aggressively - first sentence only
            available = max_tokens - tokens_used
            if available <= 10:
                break

            compressed = _compress_pattern(pattern.text)
            compressed_tokens = estimate_tokens(compressed)

            if compressed_tokens <= available:
                context_parts.append(
                    f"[{pattern.stability_score:.2f}|compressed] {compressed}"
                )
                tokens_used += compressed_tokens

    if not context_parts:
        return ""

    header = "--- WHEELER CONTEXT (stability-weighted) ---"
    return header + "\n" + "\n".join(context_parts) + "\n"


def rank_patterns(patterns: List[WeightedPattern]) -> List[WeightedPattern]:
    """
    Sort patterns by stability score descending.
    Utility for callers that want ranking without context assembly.
    """
    return sorted(patterns, key=lambda p: p.stability_score, reverse=True)


def budget_summary(patterns: List[WeightedPattern], max_tokens: int) -> dict:
    """
    Return a diagnostic summary of how tokens would be allocated.

    Useful for debugging and logging the context budget decisions.
    """
    viable = [p for p in patterns if p.stability_score >= 0.05]
    viable.sort(key=lambda p: p.stability_score, reverse=True)

    tiers = {"high": [], "medium": [], "low": [], "dropped": []}

    for p in patterns:
        if p.stability_score >= TIER_HIGH:
            tiers["high"].append(p.pattern_id or p.text[:40])
        elif p.stability_score >= TIER_MEDIUM:
            tiers["medium"].append(p.pattern_id or p.text[:40])
        elif p.stability_score >= 0.05:
            tiers["low"].append(p.pattern_id or p.text[:40])
        else:
            tiers["dropped"].append(p.pattern_id or p.text[:40])

    return {
        "max_tokens": max_tokens,
        "total_patterns": len(patterns),
        "viable_patterns": len(viable),
        "tiers": {k: len(v) for k, v in tiers.items()},
        "tier_detail": tiers,
    }


# --- Internal helpers ---


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately fit within a token budget."""
    words = text.split()
    # ~1.3 tokens per word, so max_words = max_tokens / 1.3
    max_words = max(1, int(max_tokens / 1.3))
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _compress_pattern(text: str) -> str:
    """
    Compress a low-stability pattern to its essential content.
    Extracts the first sentence as a summary.
    """
    # Take first sentence
    for delimiter in [".", "!", "?", "\n"]:
        idx = text.find(delimiter)
        if 0 < idx < 200:
            return text[: idx + 1].strip()

    # Fallback: first 100 chars
    if len(text) > 100:
        return text[:100].strip()
    return text.strip()

"""Tests for wheeler_memory.theories.structured — theory construction and prompting."""

import numpy as np
import pytest

from wheeler_memory.theories.structured import (
    Theory,
    TheoryFrame,
    TheoryHypothesis,
    theory_to_prompt,
)


# ── Dataclasses ───────────────────────────────────────────────────────


class TestTheoryFrame:
    """TheoryFrame dataclass."""

    def test_construction(self):
        """All fields set correctly."""
        f = TheoryFrame(
            hex_key="abc123",
            text="cats are mammals",
            temperature=0.3,
            basin_width=0.5,
            confidence=0.15,
        )
        assert f.hex_key == "abc123"
        assert f.text == "cats are mammals"
        assert f.temperature == 0.3
        assert f.basin_width == 0.5
        assert f.confidence == 0.15

    def test_default_relationships(self):
        """relationships defaults to empty list."""
        f = TheoryFrame(
            hex_key="x", text="t", temperature=0.0, basin_width=0.0, confidence=0.0
        )
        assert f.relationships == []


class TestTheoryHypothesis:
    """TheoryHypothesis dataclass."""

    def test_default_status(self):
        """Status defaults to 'untested'."""
        h = TheoryHypothesis(
            synthesized_from=["a", "b"],
            description="Hypothesis text",
            confidence=0.7,
        )
        assert h.status == "untested"

    def test_custom_status(self):
        """Custom status is preserved."""
        h = TheoryHypothesis(
            synthesized_from=[], description="", confidence=0.0, status="converged"
        )
        assert h.status == "converged"


class TestTheory:
    """Theory dataclass."""

    def test_empty_theory(self):
        """Default Theory has no frames, hypotheses, or budget."""
        t = Theory()
        assert t.active_frames == []
        assert t.hypotheses == []
        assert t.context_budget == {}


# ── theory_to_prompt ──────────────────────────────────────────────────


class TestTheoryToPrompt:
    """theory_to_prompt() → LLM prompt string."""

    def test_empty_theory(self):
        """Empty theory → preamble + budget instruction, no concept section."""
        t = Theory()
        prompt = theory_to_prompt(t)
        assert "expressing a theory" in prompt
        assert "Active concepts" not in prompt
        assert "Hypotheses" not in prompt

    def test_with_frames(self):
        """Theory with frames → lists concepts in prompt."""
        t = Theory(
            active_frames=[
                TheoryFrame(
                    hex_key="aaa",
                    text="cats",
                    temperature=0.5,
                    basin_width=0.3,
                    confidence=0.15,
                ),
                TheoryFrame(
                    hex_key="bbb",
                    text="dogs",
                    temperature=0.4,
                    basin_width=0.6,
                    confidence=0.24,
                ),
            ],
            context_budget={"aaa": 0.4, "bbb": 0.6},
        )
        prompt = theory_to_prompt(t)
        assert "Active concepts:" in prompt
        assert '"cats"' in prompt
        assert '"dogs"' in prompt
        assert "confidence: 0.150" in prompt
        assert "context share: 60.0%" in prompt

    def test_with_hypotheses(self):
        """Theory with hypotheses → lists hypotheses in prompt."""
        t = Theory(
            hypotheses=[
                TheoryHypothesis(
                    synthesized_from=["aaa", "bbb"],
                    description="Pets are good",
                    confidence=0.8,
                    status="untested",
                ),
            ],
        )
        prompt = theory_to_prompt(t)
        assert "Hypotheses" in prompt
        assert "Pets are good" in prompt
        assert "untested" in prompt
        assert "aaa, bbb" in prompt

    def test_budget_instruction_always_present(self):
        """Budget allocation instruction is always appended."""
        prompt = theory_to_prompt(Theory())
        assert "Allocate your response proportionally" in prompt

    def test_prompt_is_string(self):
        """Returns a string."""
        assert isinstance(theory_to_prompt(Theory()), str)

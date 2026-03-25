"""Tests for the Wheeler-primary decoder."""

import json
from unittest.mock import patch, MagicMock

import pytest

from wheeler_memory.decoder import (
    CONFIDENCE_FLOOR,
    DecoderState,
    WheelerPrimaryAgent,
    _confidence_label,
    extract_state,
    format_state,
)


# ── State extraction ──────────────────────────────────────────────────────────


def _make_hit(
    text,
    similarity=0.5,
    temperature=0.5,
    tier="warm",
    state="CONVERGED",
    ticks=42,
    chunk="general",
):
    return {
        "text": text,
        "similarity": similarity,
        "temperature": temperature,
        "temperature_tier": tier,
        "state": state,
        "convergence_ticks": ticks,
        "chunk": chunk,
    }


class TestExtractStateConfident:
    def test_high_similarity_is_confident(self):
        hits = [
            _make_hit("concept A", similarity=0.8),
            _make_hit("concept B", similarity=0.7),
        ]
        state = extract_state("test query", hits)

        assert not state.uncertain
        assert state.confidence == pytest.approx(0.8, abs=0.01)
        assert state.query == "test query"
        assert len(state.attractors) == 2

    def test_single_strong_hit(self):
        hits = [_make_hit("strong memory", similarity=0.9)]
        state = extract_state("query", hits)

        assert not state.uncertain
        assert state.confidence == pytest.approx(0.9)


class TestExtractStateUncertain:
    def test_low_similarity_is_uncertain(self):
        hits = [
            _make_hit("weak A", similarity=0.05),
            _make_hit("weak B", similarity=0.15),
        ]
        state = extract_state("test query", hits)

        assert state.uncertain
        assert state.confidence < CONFIDENCE_FLOOR

    def test_empty_results_is_uncertain(self):
        state = extract_state("query with no matches", [])

        assert state.uncertain
        assert state.confidence == 0.0
        assert len(state.attractors) == 0


class TestExtractStateCoActivation:
    def test_same_chunk_co_activation(self):
        hits = [
            _make_hit("concept A", similarity=0.8, chunk="science"),
            _make_hit("concept B", similarity=0.7, chunk="science"),
        ]
        state = extract_state("query", hits)

        assert len(state.co_activated) == 1

    def test_different_chunks_no_co_activation(self):
        hits = [
            _make_hit("concept A", similarity=0.8, chunk="science"),
            _make_hit("concept B", similarity=0.7, chunk="code"),
        ]
        state = extract_state("query", hits)

        assert len(state.co_activated) == 0

    def test_low_sim_no_co_activation(self):
        hits = [
            _make_hit("concept A", similarity=0.1, chunk="science"),
            _make_hit("concept B", similarity=0.2, chunk="science"),
        ]
        state = extract_state("query", hits)

        assert len(state.co_activated) == 0


# ── Confidence labels ─────────────────────────────────────────────────────────


class TestConfidenceLabel:
    def test_high(self):
        assert _confidence_label(0.8) == "high"

    def test_medium(self):
        assert _confidence_label(0.35) == "medium"

    def test_low(self):
        assert _confidence_label(0.25) == "low"

    def test_uncertain(self):
        assert _confidence_label(0.1) == "uncertain"


# ── State formatting ──────────────────────────────────────────────────────────


class TestFormatState:
    def test_contains_query(self):
        state = DecoderState(query="test question")
        formatted = format_state(state)
        assert "QUERY: test question" in formatted

    def test_confident_has_instruction(self):
        state = DecoderState(
            query="q",
            attractors=[_make_hit("memory", similarity=0.8)],
            confidence=0.8,
            uncertain=False,
        )
        formatted = format_state(state)
        assert "INSTRUCTION:" in formatted
        assert "ONLY the above memories" in formatted

    def test_uncertain_has_warning(self):
        state = DecoderState(query="q", uncertain=True, confidence=0.1)
        formatted = format_state(state)
        assert "Confidence is low" in formatted
        assert "Do not fabricate" in formatted

    def test_memories_are_listed(self):
        state = DecoderState(
            query="q",
            attractors=[
                _make_hit("alpha memory", similarity=0.85, tier="hot"),
                _make_hit("beta memory", similarity=0.60, tier="warm"),
            ],
            confidence=0.7,
            uncertain=False,
        )
        formatted = format_state(state)
        assert "alpha memory" in formatted
        assert "beta memory" in formatted
        assert "sim=0.85" in formatted
        assert "temp=hot" in formatted

    def test_empty_memories(self):
        state = DecoderState(query="q")
        formatted = format_state(state)
        assert "none found" in formatted

    def test_co_activation_listed(self):
        state = DecoderState(
            query="q",
            co_activated=[("concept A", "concept B")],
            confidence=0.5,
            uncertain=False,
        )
        formatted = format_state(state)
        assert "CO-ACTIVATION" in formatted
        assert "concept A" in formatted
        assert "concept B" in formatted

    def test_ca_metadata_in_output(self):
        state = DecoderState(
            query="q",
            attractors=[_make_hit("mem", state="OSCILLATING", ticks=55)],
            confidence=0.5,
            uncertain=False,
        )
        formatted = format_state(state)
        assert "OSCILLATING" in formatted
        assert "55" in formatted


# ── WheelerPrimaryAgent ───────────────────────────────────────────────────────


class TestWheelerPrimaryAgentRun:
    @patch("wheeler_memory.decoder._ollama_generate")
    @patch("wheeler_memory.decoder.recall_memory")
    def test_full_pipeline(self, mock_recall, mock_ollama):
        """Agent calls recall → extract → format → ollama → returns text."""
        mock_recall.return_value = [
            _make_hit("stored knowledge", similarity=0.8),
        ]
        mock_ollama.return_value = {
            "message": {"content": "Based on memory: stored knowledge."}
        }

        agent = WheelerPrimaryAgent(model="test-model")
        reply = agent.run("test query")

        assert reply == "Based on memory: stored knowledge."
        mock_recall.assert_called_once()
        mock_ollama.assert_called_once()

        # Verify no tools were passed
        call_args = mock_ollama.call_args
        messages = call_args[0][0]
        assert len(messages) == 2  # system + user
        assert "language renderer" in messages[0]["content"]

    @patch("wheeler_memory.decoder._ollama_generate")
    @patch("wheeler_memory.decoder.recall_memory")
    def test_uncertain_state_propagated(self, mock_recall, mock_ollama):
        """When recall returns low similarity, uncertainty is signaled."""
        mock_recall.return_value = [
            _make_hit("vague", similarity=0.1),
        ]
        mock_ollama.return_value = {"message": {"content": "I'm not sure about this."}}

        agent = WheelerPrimaryAgent(confidence_floor=0.5)
        agent.run("obscure query")

        # Check the prompt contains uncertainty signal
        call_args = mock_ollama.call_args
        user_prompt = call_args[0][0][1]["content"]
        assert "Confidence is low" in user_prompt

    @patch("wheeler_memory.decoder._ollama_generate")
    @patch("wheeler_memory.decoder.recall_memory")
    def test_embedding_recall_always_used(self, mock_recall, mock_ollama):
        """Verify use_embedding=True is always passed to recall."""
        mock_recall.return_value = []
        mock_ollama.return_value = {"message": {"content": ""}}

        agent = WheelerPrimaryAgent()
        agent.run("anything")

        kwargs = mock_recall.call_args.kwargs
        assert kwargs.get("use_embedding") is True


class TestDecoderNoToolLoop:
    @patch("wheeler_memory.decoder._ollama_generate")
    @patch("wheeler_memory.decoder.recall_memory")
    def test_no_tool_calls_in_response(self, mock_recall, mock_ollama):
        """Decoder should never send tool definitions to the model."""
        mock_recall.return_value = []
        mock_ollama.return_value = {"message": {"content": "response"}}

        agent = WheelerPrimaryAgent()
        agent.run("query")

        # _ollama_generate should be called without any tools parameter
        call_args = mock_ollama.call_args
        # The function signature is (messages, model, base_url)
        # No tools arg
        assert len(call_args[0]) == 3  # messages, model, base_url

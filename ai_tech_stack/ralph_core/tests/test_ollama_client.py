"""
Tests for the OllamaClient streaming module.

These tests mock the HTTP layer so they run without a live Ollama server.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the ai_tech_stack root is importable
_AI_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _ndjson_lines(chunks: list[dict]) -> list[str]:
    """Convert a list of dicts into NDJSON strings."""
    return [json.dumps(c) for c in chunks]


def _mock_streaming_response(chunks: list[dict], status_code=200):
    """Return a mock Response whose iter_lines yields NDJSON."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines = MagicMock(
        return_value=iter(_ndjson_lines(chunks))
    )
    resp.text = ""
    return resp


# ------------------------------------------------------------------ #
#  Construction / defaults
# ------------------------------------------------------------------ #

class TestClientInit:
    def test_defaults(self):
        client = OllamaClient()
        assert client.base_url == DEFAULT_BASE_URL
        assert client.model == DEFAULT_MODEL
        assert client.timeout == 300

    def test_custom_params(self):
        client = OllamaClient(
            base_url="http://gpu-box:11434",
            model="llama3:70b",
            timeout=600,
        )
        assert client.base_url == "http://gpu-box:11434"
        assert client.model == "llama3:70b"
        assert client.timeout == 600

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_env_var_model(self):
        with patch.dict("os.environ", {"RALPH_MODEL": "mistral:7b"}):
            client = OllamaClient()
            assert client.model == "mistral:7b"

    def test_env_var_host(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://remote:9999"}):
            client = OllamaClient()
            assert client.base_url == "http://remote:9999"

    def test_explicit_overrides_env(self):
        with patch.dict("os.environ", {"RALPH_MODEL": "mistral:7b"}):
            client = OllamaClient(model="phi3:mini")
            assert client.model == "phi3:mini"

    def test_repr(self):
        client = OllamaClient()
        r = repr(client)
        assert "OllamaClient" in r
        assert DEFAULT_BASE_URL in r
        assert DEFAULT_MODEL in r


# ------------------------------------------------------------------ #
#  Streaming generate
# ------------------------------------------------------------------ #

class TestGenerate:
    def test_streaming_tokens(self):
        chunks = [
            {"response": "Hello", "done": False},
            {"response": " world", "done": False},
            {"response": "!", "done": True, "context": [1, 2, 3]},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=mock_resp):
            result = list(client.generate("Say hello"))

        assert len(result) == 3
        assert result[0]["response"] == "Hello"
        assert result[2]["done"] is True
        assert result[2]["context"] == [1, 2, 3]

    def test_generate_full(self):
        chunks = [
            {"response": "one ", "done": False},
            {"response": "two ", "done": False},
            {"response": "three", "done": True},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=mock_resp):
            text = client.generate_full("Count to three")

        assert text == "one two three"

    def test_model_override_per_call(self):
        chunks = [{"response": "ok", "done": True}]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient(model="qwen2.5:3b")
        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            list(client.generate("hi", model="llama3:70b"))

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3:70b"

    def test_system_prompt_sent(self):
        chunks = [{"response": "ok", "done": True}]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            list(client.generate("hi", system="Be helpful"))

        payload = mock_post.call_args[1]["json"]
        assert payload["system"] == "Be helpful"

    def test_options_passed(self):
        chunks = [{"response": "ok", "done": True}]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            list(client.generate("hi", options={"temperature": 0.1}))

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.1

    def test_context_forwarded(self):
        chunks = [{"response": "ok", "done": True}]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        ctx = [10, 20, 30]
        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            list(client.generate("hi", context=ctx))

        payload = mock_post.call_args[1]["json"]
        assert payload["context"] == ctx


# ------------------------------------------------------------------ #
#  Streaming chat
# ------------------------------------------------------------------ #

class TestChat:
    def test_streaming_chat(self):
        chunks = [
            {"message": {"role": "assistant", "content": "Hi"}, "done": False},
            {"message": {"role": "assistant", "content": "!"}, "done": True},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hello"}]
        with patch.object(client.session, "post", return_value=mock_resp):
            result = list(client.chat(messages))

        assert len(result) == 2
        assert result[0]["message"]["content"] == "Hi"

    def test_chat_full(self):
        chunks = [
            {"message": {"role": "assistant", "content": "Hello "}, "done": False},
            {"message": {"role": "assistant", "content": "there"}, "done": True},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        with patch.object(client.session, "post", return_value=mock_resp):
            text = client.chat_full(messages)

        assert text == "Hello there"

    def test_system_prepended_to_chat(self):
        chunks = [
            {"message": {"role": "assistant", "content": "ok"}, "done": True},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            list(client.chat(messages, system="Be concise"))

        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be concise"
        assert payload["messages"][1]["role"] == "user"


# ------------------------------------------------------------------ #
#  Error handling
# ------------------------------------------------------------------ #

class TestErrors:
    def test_connection_error(self):
        client = OllamaClient()
        with patch.object(
            client.session, "post",
            side_effect=__import__("requests").ConnectionError("refused"),
        ):
            with pytest.raises(OllamaConnectionError, match="Cannot connect"):
                list(client.generate("hi"))

    def test_http_error_status(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal server error"

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=resp):
            with pytest.raises(OllamaError, match="500"):
                list(client.generate("hi"))

    def test_error_in_stream(self):
        chunks = [
            {"response": "partial", "done": False},
            {"error": "model not found"},
        ]
        mock_resp = _mock_streaming_response(chunks)

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(OllamaError, match="model not found"):
                list(client.generate("hi"))

    def test_malformed_json_skipped(self):
        """Malformed lines are skipped; valid lines still yielded."""
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines = MagicMock(
            return_value=iter([
                '{"response": "ok", "done": false}',
                "NOT JSON{{{",
                '{"response": "!", "done": true}',
            ])
        )

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=resp):
            result = list(client.generate("hi"))

        assert len(result) == 2
        assert result[0]["response"] == "ok"
        assert result[1]["response"] == "!"


# ------------------------------------------------------------------ #
#  Server introspection
# ------------------------------------------------------------------ #

class TestIntrospection:
    def test_is_available_true(self):
        resp = MagicMock()
        resp.status_code = 200

        client = OllamaClient()
        with patch.object(client.session, "get", return_value=resp):
            assert client.is_available() is True

    def test_is_available_false(self):
        client = OllamaClient()
        with patch.object(
            client.session, "get",
            side_effect=__import__("requests").ConnectionError("nope"),
        ):
            assert client.is_available() is False

    def test_list_models(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:3b"},
                {"name": "llama3:8b"},
            ]
        }

        client = OllamaClient()
        with patch.object(client.session, "get", return_value=resp):
            models = client.list_models()

        assert models == ["qwen2.5:3b", "llama3:8b"]


# ------------------------------------------------------------------ #
#  Context manager
# ------------------------------------------------------------------ #

class TestContextManager:
    def test_context_manager_closes_session(self):
        with OllamaClient() as client:
            mock_session = MagicMock()
            client.session = mock_session

        # After exiting, session.close() should have been called
        mock_session.close.assert_called_once()

    def test_non_streaming_mode(self):
        """When stream=False, Ollama returns a single JSON object."""
        single = {"response": "full response", "done": True}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = single

        client = OllamaClient()
        with patch.object(client.session, "post", return_value=resp):
            result = list(client.generate("hi", stream=False))

        assert len(result) == 1
        assert result[0]["response"] == "full response"

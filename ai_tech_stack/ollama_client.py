"""
Ollama Streaming Client
=======================

Modular client for communicating with Ollama via persistent HTTP
streaming connections. Uses requests.Session for connection pooling
(HTTP keep-alive) so the TCP connection is reused across calls in
fast iterative loops.

The model is configurable at init or per-call, making it easy to
swap models without changing any other code.

Usage:
    from ollama_client import OllamaClient

    client = OllamaClient(model="qwen2.5:3b")

    # Stream tokens as they arrive
    for chunk in client.generate("Explain quicksort"):
        print(chunk["response"], end="", flush=True)

    # Get full response at once
    text = client.generate_chat("Explain quicksort")

    # Chat with message history
    messages = [{"role": "user", "content": "Hello"}]
    for chunk in client.chat(messages):
        print(chunk["message"]["content"], end="", flush=True)
"""

import json
import os
import logging
from typing import Iterator, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


class OllamaError(Exception):
    """Raised when the Ollama API returns an error."""


class OllamaConnectionError(OllamaError):
    """Raised when the client cannot reach the Ollama server."""


class OllamaClient:
    """Persistent streaming client for the Ollama HTTP API.

    Maintains a requests.Session with connection pooling so that
    iterative call loops reuse the same TCP socket instead of
    opening a new connection for every request.

    Args:
        base_url: Ollama server URL (default http://localhost:11434).
        model:    Default model name.  Can be overridden per call.
        timeout:  Request timeout in seconds (default 300 — large
                  models may be slow on first load).
        max_retries: Number of automatic retries on connection errors.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
    ):
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model = model or os.environ.get("RALPH_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

        # Persistent session — keeps TCP connections alive between calls
        self.session = requests.Session()

        # Retry strategy for transient network errors
        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=1,
            pool_maxsize=1,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # --------------------------------------------------------------------- #
    #  Core streaming generators
    # --------------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        context: Optional[list] = None,
        options: Optional[dict] = None,
        stream: bool = True,
    ) -> Iterator[dict]:
        """Stream a /api/generate response token-by-token.

        Yields one dict per NDJSON line from Ollama.  Each dict
        contains at minimum a "response" key with the new token text.
        The final dict has "done": true and includes timing stats.

        Args:
            prompt:  The prompt string.
            model:   Override the default model for this call.
            system:  Optional system prompt.
            context: Optional context array from a previous generate call.
            options: Model parameters (temperature, top_p, num_ctx, etc.).
            stream:  If False, Ollama buffers the full response server-side
                     and returns a single JSON object.
        """
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system is not None:
            payload["system"] = system
        if context is not None:
            payload["context"] = context
        if options is not None:
            payload["options"] = options

        yield from self._stream_request("/api/generate", payload, stream)

    def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[dict] = None,
        stream: bool = True,
    ) -> Iterator[dict]:
        """Stream a /api/chat response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            model:    Override the default model for this call.
            system:   Optional system prompt (prepended as a system message).
            options:  Model parameters.
            stream:   If False, buffer server-side.
        """
        if system is not None:
            messages = [{"role": "system", "content": system}] + list(messages)

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if options is not None:
            payload["options"] = options

        yield from self._stream_request("/api/chat", payload, stream)

    # --------------------------------------------------------------------- #
    #  Convenience helpers
    # --------------------------------------------------------------------- #

    def generate_full(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        context: Optional[list] = None,
        options: Optional[dict] = None,
    ) -> str:
        """Generate and return the complete text (non-streaming convenience).

        Returns the concatenated response text.
        """
        parts = []
        for chunk in self.generate(
            prompt,
            model=model,
            system=system,
            context=context,
            options=options,
            stream=True,
        ):
            token = chunk.get("response", "")
            parts.append(token)
        return "".join(parts)

    def chat_full(
        self,
        messages: list,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> str:
        """Chat and return the complete assistant message."""
        parts = []
        for chunk in self.chat(
            messages,
            model=model,
            system=system,
            options=options,
            stream=True,
        ):
            msg = chunk.get("message", {})
            parts.append(msg.get("content", ""))
        return "".join(parts)

    # --------------------------------------------------------------------- #
    #  Server introspection
    # --------------------------------------------------------------------- #

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list:
        """Return list of locally available model names."""
        resp = self._get("/api/tags")
        return [m["name"] for m in resp.get("models", [])]

    def model_info(self, model: Optional[str] = None) -> dict:
        """Return metadata for a model."""
        payload = {"name": model or self.model}
        resp = self.session.post(
            f"{self.base_url}/api/show",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    def _stream_request(
        self, endpoint: str, payload: dict, stream: bool
    ) -> Iterator[dict]:
        """POST to an Ollama streaming endpoint, yield parsed JSON lines."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.post(
                url,
                json=payload,
                stream=stream,
                timeout=self.timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the server running?"
            ) from exc

        if resp.status_code != 200:
            body = resp.text
            raise OllamaError(
                f"Ollama returned {resp.status_code}: {body}"
            )

        if not stream:
            yield resp.json()
            return

        # Stream NDJSON lines as they arrive over the persistent connection
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line: %s", raw_line)
                continue

            if "error" in data:
                raise OllamaError(data["error"])

            yield data

    def _get(self, endpoint: str) -> dict:
        """GET a JSON endpoint."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the server running?"
            ) from exc
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the underlying session and its connection pool."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (
            f"OllamaClient(base_url={self.base_url!r}, "
            f"model={self.model!r})"
        )

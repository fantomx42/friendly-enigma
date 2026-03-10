"""Wheeler-primary decoder: small model as a pure language renderer.

In this mode Wheeler Memory is the primary cognitive system.  The small
model (e.g. Qwen2.5-1B via Ollama) does NOT reason — it reads Wheeler's
attractor state and renders it as natural language.

The pipeline:
  query → embed → recall from Wheeler → extract state → format prompt
  → small model generates response grounded in Wheeler state

Usage
-----
>>> from wheeler_memory.decoder import WheelerPrimaryAgent
>>> agent = WheelerPrimaryAgent()
>>> reply = agent.run("What is quantum entanglement?")
>>> print(reply)

Or via CLI:
    wheeler-primary "What is quantum entanglement?"
    wheeler-primary --interactive
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .storage import recall_memory
from .warming import _load_associations


# ── System prompt: the small model is a renderer, not a thinker ───────────────

_DECODER_SYSTEM_PROMPT = """\
You are a language renderer for Wheeler Memory.

Your ONLY job is to express the memory state below as clear, natural language.
Do NOT add knowledge from your own training data. Do NOT speculate beyond
what the memory state provides. If the state is uncertain or thin, say so
honestly rather than filling gaps with guesses.

Ground your entire response in the MEMORY STATE provided with each query.
"""

DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
CONFIDENCE_FLOOR = 0.18


# ── State extraction ──────────────────────────────────────────────────────────


@dataclass
class DecoderState:
    """Structured representation of Wheeler's recall state for the decoder."""

    query: str
    attractors: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    co_activated: list[tuple[str, str]] = field(default_factory=list)
    uncertain: bool = True


def extract_state(
    query: str,
    recall_results: list[dict],
    confidence_floor: float = CONFIDENCE_FLOOR,
) -> DecoderState:
    """Extract structured state from recall_memory() results.

    Parameters
    ----------
    query : str
        The user's original query.
    recall_results : list[dict]
        Results from recall_memory(), each with keys:
        text, similarity, temperature, temperature_tier, state, chunk, etc.
    confidence_floor : float
        Below this mean similarity, the state is marked uncertain.

    Returns
    -------
    DecoderState
        Structured state ready for formatting.
    """
    if not recall_results:
        return DecoderState(query=query)

    similarities = [h.get("similarity", 0.0) for h in recall_results]
    confidence = max(similarities) if similarities else 0.0

    # Detect co-activation from shared chunks or high mutual similarity
    co_activated: list[tuple[str, str]] = []
    for i, a in enumerate(recall_results):
        for b in recall_results[i + 1 :]:
            # Same chunk and both reasonably similar = co-activated
            if (
                a.get("chunk") == b.get("chunk")
                and a.get("similarity", 0) > 0.18
                and b.get("similarity", 0) > 0.18
            ):
                co_activated.append(
                    (a["text"][:60], b["text"][:60])
                )

    return DecoderState(
        query=query,
        attractors=recall_results,
        confidence=confidence,
        co_activated=co_activated,
        uncertain=confidence < confidence_floor,
    )


# ── State formatting ──────────────────────────────────────────────────────────


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.40:
        return "high"
    if confidence >= 0.30:
        return "medium"
    if confidence >= 0.18:
        return "low"
    return "uncertain"


def format_state(state: DecoderState) -> str:
    """Serialize DecoderState into structured text for the small model.

    The output is a deterministic, parseable prompt that gives the model
    everything it needs to produce a grounded response.
    """
    lines = [
        f"QUERY: {state.query}",
        f"CONFIDENCE: {_confidence_label(state.confidence)}",
        "",
    ]

    if state.attractors:
        lines.append("ACTIVE MEMORIES (ranked by relevance):")
        for i, att in enumerate(state.attractors, 1):
            sim = att.get("similarity", 0.0)
            tier = att.get("temperature_tier", "cold")
            text = att.get("text", "")
            ca_state = att.get("state", "UNKNOWN")
            ticks = att.get("convergence_ticks", "?")
            lines.append(
                f"  {i}. [sim={sim:.2f}, temp={tier}, "
                f"CA={ca_state}/{ticks}] \"{text}\""
            )
        lines.append("")
    else:
        lines.append("ACTIVE MEMORIES: none found")
        lines.append("")

    if state.co_activated:
        lines.append("CO-ACTIVATION (memories that fired together):")
        for a_text, b_text in state.co_activated:
            lines.append(f"  \"{a_text}\" <-> \"{b_text}\"")
        lines.append("")

    if state.uncertain:
        lines.append(
            "NOTE: Confidence is low. Acknowledge uncertainty in your response. "
            "Do not fabricate information to fill gaps."
        )
    else:
        lines.append(
            "INSTRUCTION: Respond to the query using ONLY the above memories. "
            "Synthesize them into a clear, natural answer."
        )

    return "\n".join(lines)


# ── Ollama client (minimal, no tools) ─────────────────────────────────────────


def _ollama_generate(
    messages: list[dict],
    model: str,
    base_url: str,
    stream: bool = False,
) -> dict:
    """POST /api/chat to Ollama without tool definitions."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: ollama serve"
        ) from exc


def _ollama_generate_stream(
    messages: list[dict],
    model: str,
    base_url: str,
):
    """Streaming variant of _ollama_generate."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                yield chunk
                if chunk.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: ollama serve"
        ) from exc


# ── Wheeler-primary agent ─────────────────────────────────────────────────────


class WheelerPrimaryAgent:
    """Wheeler-driven agent where the small model is a pure decoder.

    Unlike WheelerAgent (which uses an LLM with tools for reasoning),
    this agent uses Wheeler Memory as the primary cognitive system.
    The small model receives Wheeler's attractor state and renders it
    as natural language — it does not reason, plan, or use tools.

    Parameters
    ----------
    model : str
        Ollama model name for the decoder (small, fast model).
    ollama_url : str
        Base URL of the Ollama server.
    data_dir : Path, optional
        Wheeler Memory data directory.
    recall_k : int
        Number of memories to recall per query.
    confidence_floor : float
        Below this mean similarity, responses signal uncertainty.
    reconstruct : bool
        If True, use reconstructive recall (context-blended).
    reconstruct_alpha : float
        Blend weight for reconstruction (0=stored, 1=query).
    verbose : bool
        Print state extraction details to stderr.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        data_dir: str | Path | None = None,
        recall_k: int = 5,
        confidence_floor: float = CONFIDENCE_FLOOR,
        reconstruct: bool = True,
        reconstruct_alpha: float = 0.3,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.data_dir = Path(data_dir) if data_dir else None
        self.recall_k = recall_k
        self.confidence_floor = confidence_floor
        self.reconstruct = reconstruct
        self.reconstruct_alpha = reconstruct_alpha
        self.verbose = verbose

    def run(self, user_message: str) -> str:
        """Full pipeline: encode → recall → extract state → decode.

        Returns the small model's response grounded in Wheeler state.
        """
        # 1. Recall from Wheeler (embedding-based)
        hits = recall_memory(
            user_message,
            top_k=self.recall_k,
            data_dir=self.data_dir,
            use_embedding=True,
            reconstruct=self.reconstruct,
            reconstruct_alpha=self.reconstruct_alpha,
        )

        # 2. Extract structured state
        state = extract_state(user_message, hits, self.confidence_floor)

        if self.verbose:
            print(
                f"[wheeler-primary] confidence={state.confidence:.2f} "
                f"uncertain={state.uncertain} hits={len(state.attractors)}",
                file=sys.stderr,
            )

        # 3. Format for decoder
        prompt = format_state(state)

        if self.verbose:
            print(f"[wheeler-primary] decoder prompt:\n{prompt}", file=sys.stderr)

        # 4. Small model generates — no tools, no reasoning loop
        messages = [
            {"role": "system", "content": _DECODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = _ollama_generate(messages, self.model, self.ollama_url)
        return resp.get("message", {}).get("content", "")

    def run_stream(self, user_message: str) -> Iterator[dict]:
        """Streaming variant yielding typed events.

        Events:
          {"type": "recall", "hits": [...]}
          {"type": "state", "confidence": float, "uncertain": bool}
          {"type": "token", "content": str}
          {"type": "done", "content": str}
        """
        # 1. Recall
        hits = recall_memory(
            user_message,
            top_k=self.recall_k,
            data_dir=self.data_dir,
            use_embedding=True,
            reconstruct=self.reconstruct,
            reconstruct_alpha=self.reconstruct_alpha,
        )
        yield {"type": "recall", "hits": hits}

        # 2. Extract state
        state = extract_state(user_message, hits, self.confidence_floor)
        yield {
            "type": "state",
            "confidence": state.confidence,
            "uncertain": state.uncertain,
        }

        # 3. Decode via streaming
        prompt = format_state(state)
        messages = [
            {"role": "system", "content": _DECODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        full_response = ""
        for chunk in _ollama_generate_stream(messages, self.model, self.ollama_url):
            msg = chunk.get("message", {})
            token = msg.get("content", "")
            if token:
                full_response += token
                yield {"type": "token", "content": token}
            if chunk.get("done"):
                break

        yield {"type": "done", "content": full_response}

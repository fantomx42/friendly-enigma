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
from .theories.metrics import classify_output
from .agent import _attractors_from_hits


def _query_seed_correlation(text: str) -> float | None:
    """Pearson(seed, attractor) for the query — measures how far it traveled."""
    try:
        import numpy as np
        from scipy.stats import pearsonr

        from .dynamics import evolve_and_interpret
        from .rotation import _get_frame_fn

        frame_fn = _get_frame_fn(True, encoder="blended")
        seed = frame_fn(text)
        result = evolve_and_interpret(seed)
        attractor = result["attractor"]
        r, _ = pearsonr(seed.flatten(), attractor.flatten())
        return round(float(r), 3)
    except Exception:
        return None


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

CONFIDENCE_HIGH = 0.40
CONFIDENCE_MEDIUM = 0.30
CONFIDENCE_LOW = 0.18
CONFIDENCE_FLOOR = CONFIDENCE_LOW


# ── State extraction ──────────────────────────────────────────────────────────


@dataclass
class DecoderState:
    """Structured representation of Wheeler's recall state for the decoder."""

    query: str
    attractors: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    co_activated: list[tuple[str, str]] = field(default_factory=list)
    uncertain: bool = True
    interference_state: str = ""  # GROUNDED/ABSORBED/UNCONSOLIDATED/CONTESTED
    scm_openness: float = 1.0  # fraction of SCM cells below gap threshold
    pairwise_distances: list[tuple[int, int, float]] = field(default_factory=list)
    landscape: str = ""  # TIGHT/SPREAD/ISOLATED/EMPTY
    query_seed_corr: float | None = None  # Pearson(seed, attractor)


def _compute_pairwise_distances(
    results: list[dict],
    data_dir: Path | None,
) -> list[tuple[int, int, float]]:
    """Compute pairwise Pearson correlation between top-K attractor arrays."""
    import numpy as np
    from scipy.stats import pearsonr

    if data_dir is None:
        from .storage import _get_data_dir

        data_dir = _get_data_dir(None)

    # Load attractor arrays from disk
    attractors: list[np.ndarray | None] = []
    for h in results:
        hex_key = h.get("hex_key", "")
        chunk = h.get("chunk", "general")
        path = data_dir / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
        if path.exists():
            attractors.append(np.load(path, mmap_mode="r"))
        else:
            attractors.append(None)

    # Pairwise Pearson (1-indexed for display)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(attractors)):
        for j in range(i + 1, len(attractors)):
            if attractors[i] is not None and attractors[j] is not None:
                r, _ = pearsonr(attractors[i].flatten(), attractors[j].flatten())
                pairs.append((i + 1, j + 1, round(float(r), 3)))
    return pairs


def _landscape_label(pairs: list[tuple[int, int, float]], n_results: int) -> str:
    """Derive landscape label from pairwise attractor distances."""
    if n_results == 0:
        return "EMPTY"
    if not pairs:
        return "ISOLATED"
    mean_abs_r = sum(abs(r) for _, _, r in pairs) / len(pairs)
    if mean_abs_r > 0.4:
        return "TIGHT"
    if mean_abs_r > 0.1:
        return "SPREAD"
    return "ISOLATED"


def extract_state(
    query: str,
    recall_results: list[dict],
    confidence_floor: float = CONFIDENCE_FLOOR,
    interference_state: str = "",
    scm_openness: float = 1.0,
    data_dir: "Path | None" = None,
    query_seed_corr: float | None = None,
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
    interference_state : str
        Dominant interference state from three-grid recall (e.g. GROUNDED).
    scm_openness : float
        Fraction of SCM cells below the gap threshold.
    data_dir : Path | None
        Wheeler data directory (for loading attractor arrays).
    query_seed_corr : float | None
        Pearson(query seed frame, query attractor) — how far the query traveled.

    Returns
    -------
    DecoderState
        Structured state ready for formatting.
    """
    if not recall_results:
        return DecoderState(query=query, landscape="EMPTY")

    similarities = [h.get("similarity", 0.0) for h in recall_results]
    confidence = max(similarities) if similarities else 0.0

    # Detect co-activation from shared chunks or high mutual similarity
    co_activated: list[tuple[str, str]] = []
    for i, a in enumerate(recall_results):
        for b in recall_results[i + 1 :]:
            # Same chunk and both reasonably similar = co-activated
            if (
                a.get("chunk") == b.get("chunk")
                and a.get("similarity", 0) > confidence_floor
                and b.get("similarity", 0) > confidence_floor
            ):
                co_activated.append((a["text"][:60], b["text"][:60]))

    # Pairwise attractor distances and landscape label
    pairs = _compute_pairwise_distances(recall_results, data_dir)
    landscape = _landscape_label(pairs, len(recall_results))

    return DecoderState(
        query=query,
        attractors=recall_results,
        confidence=confidence,
        co_activated=co_activated,
        uncertain=confidence < confidence_floor,
        interference_state=interference_state,
        scm_openness=scm_openness,
        pairwise_distances=pairs,
        landscape=landscape,
        query_seed_corr=query_seed_corr,
    )


# ── State formatting ──────────────────────────────────────────────────────────


def _confidence_label(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    if confidence >= CONFIDENCE_LOW:
        return "low"
    return "uncertain"


def format_state(state: DecoderState) -> str:
    """Serialize DecoderState into structured text for the small model.

    The output is a deterministic, parseable prompt that gives the model
    everything it needs to produce a grounded response.  Beyond ranked text,
    the format exposes attractor-space topology: energy (basin depth), cluster
    structure (pos/neg islands + boundary), interference fractions per memory,
    and pairwise distances between recalled attractors.
    """
    # ── Header: query + landscape context ──────────────────────────────────
    query_line = f"QUERY: {state.query}"
    if state.query_seed_corr is not None:
        query_line += f"  (seed_corr={state.query_seed_corr})"
    lines = [query_line]

    conf_land = f"CONFIDENCE: {_confidence_label(state.confidence)}"
    if state.landscape:
        conf_land += f"  LANDSCAPE: {state.landscape}"
    lines.append(conf_land)

    if state.interference_state:
        lines.append(
            f"INTERFERENCE: {state.interference_state} "
            f"(SCM openness: {state.scm_openness:.0%})"
        )

    lines.append("")

    # ── Per-memory attractor features ──────────────────────────────────────
    if state.attractors:
        lines.append("ACTIVE MEMORIES:")
        lines.append(
            "  # E=energy H=entropy +c/-c=clusters bnd=boundary "
            "live=alive spd=speed rdelta=drift"
        )
        for i, att in enumerate(state.attractors, 1):
            sim = att.get("similarity", 0.0)
            tier = att.get("temperature_tier", "cold")
            text = att.get("text", "")
            ca_state = att.get("state", "UNKNOWN")
            ticks = att.get("convergence_ticks", "?")

            # Convergence speed label
            if ca_state == "CONVERGED" and isinstance(ticks, int):
                spd = "F" if ticks <= 50 else ("M" if ticks <= 150 else "S")
            else:
                spd = "?"

            # Core metrics line
            parts = [
                f"sim={sim:.2f}",
                f"temp={tier}",
                f"CA={ca_state}/{ticks}/{spd}",
            ]

            # Energy (basin depth)
            energy = att.get("energy")
            if energy is not None:
                parts.append(f"E={energy:.4f}")

            # Structural features
            entropy = att.get("grid_entropy")
            if entropy is not None:
                parts.append(f"H={entropy:.2f}")

            pos_c = att.get("cluster_count")
            neg_c = att.get("neg_cluster_count")
            if pos_c is not None:
                neg_s = str(neg_c) if neg_c is not None else "?"
                parts.append(f"+c={pos_c}/-c={neg_s}")

            boundary = att.get("boundary_length")
            if boundary is not None:
                parts.append(f"bnd={boundary}")

            alive = att.get("alive_fraction")
            if alive is not None:
                parts.append(f"live={alive:.2f}")

            # Reconstruction delta
            corr_stored = att.get("correlation_with_stored")
            if corr_stored is not None:
                parts.append(f"rdelta={1.0 - corr_stored:.2f}")

            lines.append(f'  {i}. [{" ".join(parts)}]')

            # Interference fractions sub-line (only from interference pipeline)
            g = att.get("grounded_frac")
            if g is not None:
                a_frac = att.get("absorbed_frac", 0)
                u = att.get("unconsolidated_frac", 0)
                x = att.get("contested_frac", 0)
                lines.append(f"     ifr=[G={g} A={a_frac} U={u} X={x}]")

            lines.append(f'     "{text}"')

        lines.append("")
    else:
        lines.append("ACTIVE MEMORIES: none found")
        lines.append("")

    # ── Basin structure: pairwise attractor distances ──────────────────────
    if state.pairwise_distances:
        pair_strs = [
            f"{i}<>{j}: r={r}" for i, j, r in state.pairwise_distances
        ]
        lines.append("BASIN STRUCTURE:")
        # Show up to 10 pairs per line, wrap if needed
        lines.append(f"  {' '.join(pair_strs[:10])}")
        if len(pair_strs) > 10:
            lines.append(f"  {' '.join(pair_strs[10:])}")
        lines.append("")

    # ── Co-activation ──────────────────────────────────────────────────────
    if state.co_activated:
        lines.append("CO-ACTIVATION:")
        for a_text, b_text in state.co_activated:
            lines.append(f'  "{a_text}" <-> "{b_text}"')
        lines.append("")

    # ── Instruction ────────────────────────────────────────────────────────
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
            f"Could not reach Ollama at {base_url}. Is it running? Try: ollama serve"
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
            f"Could not reach Ollama at {base_url}. Is it running? Try: ollama serve"
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
        use_interference: bool = False,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.data_dir = Path(data_dir) if data_dir else None
        self.recall_k = recall_k
        self.confidence_floor = confidence_floor
        self.reconstruct = reconstruct
        self.reconstruct_alpha = reconstruct_alpha
        self.verbose = verbose
        self.use_interference = use_interference

    def run(self, user_message: str) -> str:
        """Full pipeline: encode → recall → extract state → decode.

        Returns the small model's response grounded in Wheeler state.
        """
        interference_state = ""
        scm_openness = 1.0

        if self.use_interference:
            from .interference import recall_with_interference

            hits, interference_state, scm_openness = recall_with_interference(
                user_message,
                top_k=self.recall_k,
                data_dir=self.data_dir,
                encoder="blended",
                use_embedding=True,
            )
        else:
            # 1. Recall from Wheeler (embedding-based)
            hits = recall_memory(
                user_message,
                top_k=self.recall_k,
                data_dir=self.data_dir,
                use_embedding=True,
                encoder="blended",
                reconstruct=self.reconstruct,
                reconstruct_alpha=self.reconstruct_alpha,
            )

        # Compute query seed correlation (how far the query traveled in CA space)
        query_seed_corr = _query_seed_correlation(user_message)

        # 2. Extract structured state
        state = extract_state(
            user_message,
            hits,
            self.confidence_floor,
            interference_state=interference_state,
            scm_openness=scm_openness,
            data_dir=self.data_dir,
            query_seed_corr=query_seed_corr,
        )

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
        content = resp.get("message", {}).get("content", "")
        classification = classify_output(content, _attractors_from_hits(hits, self.data_dir))
        if self.verbose:
            print(f"[hallucination-discrimination] {classification}", file=sys.stderr)
        return content

    def run_stream(self, user_message: str) -> Iterator[dict]:
        """Streaming variant yielding typed events.

        Events:
          {"type": "recall", "hits": [...]}
          {"type": "state", "confidence": float, "uncertain": bool}
          {"type": "token", "content": str}
          {"type": "done", "content": str}
        """
        # 1. Recall (interference-aware, mirroring run())
        interference_state = ""
        scm_openness = 1.0

        if self.use_interference:
            from .interference import recall_with_interference

            hits, interference_state, scm_openness = recall_with_interference(
                user_message,
                top_k=self.recall_k,
                data_dir=self.data_dir,
                encoder="blended",
                use_embedding=True,
            )
        else:
            hits = recall_memory(
                user_message,
                top_k=self.recall_k,
                data_dir=self.data_dir,
                use_embedding=True,
                encoder="blended",
                reconstruct=self.reconstruct,
                reconstruct_alpha=self.reconstruct_alpha,
            )
        yield {"type": "recall", "hits": hits}

        # 2. Extract state
        query_seed_corr = _query_seed_correlation(user_message)
        state = extract_state(
            user_message,
            hits,
            self.confidence_floor,
            interference_state=interference_state,
            scm_openness=scm_openness,
            data_dir=self.data_dir,
            query_seed_corr=query_seed_corr,
        )
        yield {
            "type": "state",
            "confidence": state.confidence,
            "uncertain": state.uncertain,
            "landscape": state.landscape,
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

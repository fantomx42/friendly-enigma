"""Topology signal-isolation test for the Wheeler decode/agent path.

Question
--------
The Wheeler-primary decoder injects an *interpreted-topology* block into the
prompt it sends to the small Ollama model (``format_state`` →
``_ollama_generate`` in ``wheeler_memory/decoder.py``).  The block encodes
basin energy, landscape label, +/- cluster counts, boundary length, the
interference fractions (grounded / absorbed / unconsolidated / contested),
pairwise basin distances and SCM openness.

Does that block actually *steer the decoder*, or is it decorative text the
1.5B model ignores?  This script answers it empirically.

Method
------
1. Run one real query through the normal recall path far enough to capture
   (a) the retrieved-memory list and (b) the real topology dict.
2. Freeze the retrieved-memory *identity* (text / similarity / tier / CA
   state / ticks / chunk) for the whole run.
3. Build ~6 synthetic topology dicts using the *same schema the live code
   emits*, spanning the range of attractor-space conditions.
4. For each, assemble the prompt at the real seam (``format_state``) with
   identical frozen memories and only the topology block swapped, then call
   the decoder with temperature 0 and a fixed seed.
5. Embed each output with the existing MiniLM encoder, compute pairwise
   cosine, and report the off-diagonal divergence (1 - cosine).
6. Control: same memories + same topology, called 6x, to establish the
   deterministic floor (Ollama is not perfectly deterministic even at temp 0).
7. Verdict: divergence within noise of the floor → decorative; materially
   above the floor → the block carries signal.

The swap happens at the prompt-assembly seam ONLY.  Topology is never
regenerated from grid state inside the loop — the synthetic ``DecoderState``
objects are constructed by hand and handed straight to ``format_state``.

Run as a script (it is a measurement, not a pass/fail unit test)::

    python tests/topology_isolation.py
    python tests/topology_isolation.py --query "what is entanglement" --model qwen2.5:1.5b

It is also importable under pytest; the pytest entry skips cleanly when
Ollama or sentence-transformers are unavailable.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

from wheeler_memory.decoder import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    _DECODER_SYSTEM_PROMPT,
    DecoderState,
    _landscape_label,  # reused verbatim by the embedding arm (clean analog)
    extract_state,
    format_state,
)

# Fixed seed handed to Ollama for determinism (the decoder's own
# _ollama_generate exposes no options, so this twin forces them).
OLLAMA_SEED = 42
DEFAULT_QUERY = "What is quantum entanglement?"


# ── Determinism-forcing decoder transport ─────────────────────────────────────
#
# Mirrors decoder._ollama_generate EXACTLY (same endpoint, same message shape)
# but adds options{temperature:0, seed} so repeated identical prompts collapse
# to the deterministic floor.  This touches transport only — prompt assembly
# still flows through the real format_state seam.


def _decode(prompt: str, model: str, base_url: str, *, system: str | None = None) -> str:
    """Render one already-assembled prompt through a model at temp 0.

    ``system`` defaults to the real decoder system prompt (so Arm A/B decodes are
    byte-identical to production); the directional-grading judge passes its own.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or _DECODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "seed": OLLAMA_SEED, "top_p": 1.0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    return body.get("message", {}).get("content", "")


def _decode_claude(prompt: str, model: str, *, system: str | None = None) -> str:
    """Render one prompt through the Claude API at temperature 0.

    Used only by the directional judge (``--judge-backend claude``); the decoder
    arms always stay on Ollama so production decode is reproduced byte-for-byte.
    Requires ``ANTHROPIC_API_KEY`` and the ``anthropic`` package.  The static
    system block is marked for prompt caching so repeated graded pairs reuse it.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        system=[{
            "type": "text",
            "text": system or _DECODER_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )


def _ollama_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as r:
            r.read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def _model_present(base_url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    # accept exact or ":latest"-stripped match
    return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)


def _decode_arm(
    prompt: str, model: str, base_url: str, reps: int, label: str
) -> list[str]:
    """Decode an assembled prompt ``reps + 1`` times, DISCARDING rep 0 (cold start).

    The first decode of a freshly-assembled prompt is anomalous even at temp 0
    (a cold-start artifact ~0.23 divergence), which — because Arm A doubles as the
    deterministic floor — inflates the floor *and* the A<->B cross.  Dropping rep 0
    per arm gives a true ~0 floor, so the ratio is a real distance and a faithful
    analog that genuinely collapses cannot be masked by a contaminated floor.
    """
    outputs: list[str] = []
    for rep in range(reps + 1):
        out = _decode(prompt, model, base_url)
        if rep == 0:
            print(f"[{label}/warmup] rep0 -> {len(out):4d} chars (discarded)")
            continue
        outputs.append(out)
        print(f"[{label}] rep{rep:<3} -> {len(out):4d} chars")
    return outputs


# ── Synthetic topology overlays (same schema the live code emits) ─────────────


@dataclass
class TopologyOverlay:
    """One synthetic topology in the schema extract_state/compute_features emit.

    ``per_memory`` is merged onto each frozen hit (overwriting only structural
    fields, never identity).  The scalar fields land on the DecoderState.
    """

    name: str
    landscape: str            # TIGHT / SPREAD / ISOLATED / EMPTY
    interference_state: str   # GROUNDED / ABSORBED / UNCONSOLIDATED / CONTESTED
    scm_openness: float
    per_memory: dict          # energy, cluster_count, boundary_length, *_frac ...
    pairwise_r: float | None  # value used for every basin pair (None → no pairs)


def _overlays() -> list[TopologyOverlay]:
    """Six topologies spanning the attractor-space range."""
    return [
        TopologyOverlay(
            name="high_grounded_low_absorbed",
            landscape="TIGHT",
            interference_state="GROUNDED",
            scm_openness=0.85,
            per_memory=dict(
                energy=0.0008, grid_entropy=0.45,
                cluster_count=2, neg_cluster_count=1, boundary_length=40,
                alive_fraction=0.52, correlation_with_stored=0.97,
                grounded_frac=0.88, absorbed_frac=0.05,
                unconsolidated_frac=0.04, contested_frac=0.03,
            ),
            pairwise_r=0.62,
        ),
        TopologyOverlay(
            name="high_absorbed_low_grounded",  # the inverse
            landscape="SPREAD",
            interference_state="ABSORBED",
            scm_openness=0.30,
            per_memory=dict(
                energy=0.0009, grid_entropy=0.60,
                cluster_count=3, neg_cluster_count=2, boundary_length=70,
                alive_fraction=0.49, correlation_with_stored=0.80,
                grounded_frac=0.06, absorbed_frac=0.86,
                unconsolidated_frac=0.05, contested_frac=0.03,
            ),
            pairwise_r=0.22,
        ),
        TopologyOverlay(
            name="high_contested",
            landscape="SPREAD",
            interference_state="CONTESTED",
            scm_openness=0.50,
            per_memory=dict(
                energy=0.0015, grid_entropy=0.78,
                cluster_count=4, neg_cluster_count=4, boundary_length=110,
                alive_fraction=0.50, correlation_with_stored=0.62,
                grounded_frac=0.10, absorbed_frac=0.12,
                unconsolidated_frac=0.13, contested_frac=0.65,
            ),
            pairwise_r=0.18,
        ),
        TopologyOverlay(
            name="flat_near_zero_energy",
            landscape="ISOLATED",
            interference_state="UNCONSOLIDATED",
            scm_openness=0.98,
            per_memory=dict(
                energy=0.00002, grid_entropy=0.05,
                cluster_count=1, neg_cluster_count=0, boundary_length=2,
                alive_fraction=0.50, correlation_with_stored=0.999,
                grounded_frac=0.20, absorbed_frac=0.10,
                unconsolidated_frac=0.68, contested_frac=0.02,
            ),
            pairwise_r=0.02,
        ),
        TopologyOverlay(
            name="high_cluster_count",
            landscape="SPREAD",
            interference_state="GROUNDED",
            scm_openness=0.60,
            per_memory=dict(
                energy=0.0021, grid_entropy=0.92,
                cluster_count=14, neg_cluster_count=12, boundary_length=320,
                alive_fraction=0.51, correlation_with_stored=0.71,
                grounded_frac=0.55, absorbed_frac=0.20,
                unconsolidated_frac=0.15, contested_frac=0.10,
            ),
            pairwise_r=0.30,
        ),
        TopologyOverlay(
            name="single_basin",
            landscape="TIGHT",
            interference_state="GROUNDED",
            scm_openness=0.90,
            per_memory=dict(
                energy=0.0005, grid_entropy=0.30,
                cluster_count=1, neg_cluster_count=0, boundary_length=18,
                alive_fraction=0.50, correlation_with_stored=0.99,
                grounded_frac=0.80, absorbed_frac=0.10,
                unconsolidated_frac=0.06, contested_frac=0.04,
            ),
            pairwise_r=0.75,
        ),
    ]


# ── State assembly at the seam ────────────────────────────────────────────────

# Identity fields that define "which memory was retrieved" — frozen across the
# whole run.  Everything else on a hit is topology and may be overwritten.
_IDENTITY_KEYS = (
    "text", "similarity", "temperature", "temperature_tier",
    "state", "convergence_ticks", "chunk", "hex_key",
)


def _build_state(
    query: str,
    frozen_hits: list[dict],
    base_confidence: float,
    base_uncertain: bool,
    base_co_activated: list[tuple[str, str]],
    overlay: TopologyOverlay,
) -> DecoderState:
    """Construct a DecoderState directly — topology swapped, memories frozen.

    NOTE: extract_state() is deliberately bypassed so no topology is
    regenerated from grid state.  The swap is purely at prompt assembly.
    """
    attractors: list[dict] = []
    for hit in frozen_hits:
        merged = {k: hit[k] for k in _IDENTITY_KEYS if k in hit}
        merged.update(copy.deepcopy(overlay.per_memory))
        attractors.append(merged)

    if overlay.pairwise_r is None:
        pairwise = []
    else:
        n = len(attractors)
        pairwise = [
            (i + 1, j + 1, overlay.pairwise_r)
            for i in range(n)
            for j in range(i + 1, n)
        ]

    return DecoderState(
        query=query,
        attractors=attractors,
        confidence=base_confidence,         # frozen → CONFIDENCE/INSTRUCTION block constant
        co_activated=base_co_activated,     # frozen
        uncertain=base_uncertain,           # frozen
        interference_state=overlay.interference_state,
        scm_openness=overlay.scm_openness,
        pairwise_distances=pairwise,
        landscape=overlay.landscape,
        query_seed_corr=None,               # frozen (omitted from every prompt)
    )


# ── CA-vs-embedding ablation: topology built from MiniLM geometry alone ───────
#
# The decode-sensitivity test above proved the topology block steers the
# decoder.  This asks the orthogonal question: is the *CA* what should produce
# that topology, or does MiniLM geometry already contain it?  We build the same
# DecoderState topology fields from embeddings (no automaton ticks) and feed
# them through the identical format_state seam.

# Structural topology fields the CA derives from each attractor's grid that DO
# have an embedding analog — these are the variable under test (Arm B differs
# from Arm A here, in value only).
_TOPO_STRUCT_KEYS = ("cluster_count", "neg_cluster_count", "boundary_length", "energy")
# CA-grid quantities with no *named* / clean embedding analog — held at the real
# captured values in both arms so they cannot contribute to the divergence.
_HELD_GRID_KEYS = ("grid_entropy", "alive_fraction", "correlation_with_stored")
# Three-grid interference fractions — intrinsically CA, no embedding analog.
_THREE_GRID_KEYS = ("grounded_frac", "absorbed_frac", "unconsolidated_frac", "contested_frac")

# Graph thresholds (cortex.build_graph defaults, cortex.py:155).  Free
# parameters — surfaced here, reported in output, never hidden.
_EMB_CLUSTER_THRESHOLD = 0.5
_EMB_NEAR_THRESHOLD = 0.3


def _local_cluster_count(adj: np.ndarray, i: int, threshold: float, *, sign: int) -> int:
    """Connected components in memory *i*'s LOCAL neighborhood (de-forced analog).

    The forced analog stamps one GLOBAL connected-component count on every memory.
    This instead clusters only memory *i*'s own neighborhood, so the count varies
    per memory — the embedding analog of the CA's per-grid cluster_count.

    sign=+1: neighborhood = neighbors with r > +threshold (positive basin), seed
             *i* included as its own anchor.
    sign=-1: neighborhood = neighbors with r < -threshold — the geometric antipode,
             the embedding stand-in for the CA's negative/antipodal basins.  No
             query-similarity sign split is imposed; the sign is the sign of r.
    Components are found by ``cortex.find_clusters`` on the induced sub-adjacency.
    """
    from wheeler_memory.cortex import find_clusters

    K = adj.shape[0]
    if sign >= 0:
        members = [i] + [j for j in range(K) if j != i and adj[i, j] > threshold]
    else:
        members = [j for j in range(K) if j != i and adj[i, j] < -threshold]
    if not members:
        return 0
    sub = adj[np.ix_(members, members)]
    return len(set(find_clusters(sub, threshold).tolist()))


def embedding_topology(
    query_emb: np.ndarray,
    memory_embs: np.ndarray,
    *,
    query: str = "",
    cluster_mode: str = "forced",
    cluster_threshold: float = _EMB_CLUSTER_THRESHOLD,
    near_threshold: float = _EMB_NEAR_THRESHOLD,
) -> DecoderState:
    """Build a topology block from MiniLM geometry alone — no automaton ticks.

    Maps each topology field of DecoderState to its embedding analog and flags,
    in comments, every field where that analog is *forced*.  Three-grid and
    held CA-grid fields are left at dataclass defaults here; ``_build_embedding_arm``
    fills them from the real CA capture so both arms share them identically.

    cluster_mode : "forced"   — legacy global clustering + imposed query-sim sign
                                 split, the same two counts stamped on every memory.
                   "faithful"  — de-forced per-memory LOCAL clustering; pos/neg from
                                 positive/negative-correlation neighborhoods (Variant B1).
    query_emb   : (D,)    MiniLM vector of the query (unnormalized).
    memory_embs : (K, D)  MiniLM vectors of the recalled memory texts.
    """
    from wheeler_memory.cortex import compute_adjacency, find_clusters

    embs = memory_embs.astype(np.float32)
    K = embs.shape[0]

    # ── pairwise_distances (CLEAN analog) ───────────────────────────────────
    # compute_adjacency centers + L2-normalizes each row → Pearson r between
    # memory embeddings.  This is the SAME metric family the CA path stores:
    # decoder._compute_pairwise_distances runs pearsonr on flattened attractors.
    # So the embedding pairwise list feeds the real classifier unchanged.
    adj = compute_adjacency(embs)  # (K, K), zero diagonal
    pairwise = [
        (i + 1, j + 1, round(float(adj[i, j]), 3))
        for i in range(K)
        for j in range(i + 1, K)
    ]

    # ── landscape (CLEAN analog) ────────────────────────────────────────────
    # The production _landscape_label thresholds on mean|r| over exactly these
    # pairs (>0.4 TIGHT, >0.1 SPREAD, else ISOLATED).  No re-derivation: we call
    # the real classifier verbatim.  (The task assumed this field was approximate;
    # it is not — it is purely a function of pairwise correlation.)
    landscape = _landscape_label(pairwise, K)

    # Global connected-component labels — used by the FORCED cluster analog below
    # AND by the FORCED boundary_length analog in either mode, so compute once.
    labels = find_clusters(adj, cluster_threshold)  # (K,) cluster ids

    # ── cluster_count / neg_cluster_count ───────────────────────────────────
    if cluster_mode == "faithful":
        # DE-FORCED per-memory local clusters (Variant B1).  pos = positive-
        # correlation neighborhood; neg = negative-correlation (geometric
        # antipode) neighborhood.  Count varies per memory; NO global stamp and
        # NO imposed query-similarity sign split — yet still 100% embedding-
        # derived.  This is the honest test of whether the *forcing*, not the
        # CA, drove the prior divergence.
        pos_counts = [_local_cluster_count(adj, i, near_threshold, sign=+1) for i in range(K)]
        neg_counts = [_local_cluster_count(adj, i, near_threshold, sign=-1) for i in range(K)]
    else:
        # FORCED legacy analog: embeddings yield ONE global clustering (connected
        # components over the thresholded correlation graph) — not a per-memory
        # ternary-grid cluster count.  pos/neg split by each memory's query-
        # similarity sign vs the set mean (embeddings have no native ternary
        # sign), the same two counts stamped on EVERY attractor.
        qn = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        mn = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        qsim = mn @ qn                                   # (K,) cosine(query, memory_i)
        pos_mask = qsim >= float(qsim.mean())            # FORCED sign split at set mean
        pos_g = len(set(labels[pos_mask].tolist())) if pos_mask.any() else 0
        neg_g = len(set(labels[~pos_mask].tolist())) if (~pos_mask).any() else 0
        pos_counts = [pos_g] * K
        neg_counts = [neg_g] * K

    # ── per-memory energy (APPROX) + boundary_length (FORCED) ───────────────
    # energy analog: a memory's mean distance (1 - r) to the others — high =
    # isolated = shallow basin.  Units are NOT calibrated to the CA's
    # mean-abs-step energy; only relative spread is meaningful.
    # boundary analog: count of a memory's *near* neighbors (r > near_threshold)
    # that landed in a DIFFERENT cluster — an inter-memory graph quantity,
    # whereas the CA's boundary_length is intra-grid.  Forced.
    attractors: list[dict] = []
    for i in range(K):
        others = [adj[i, j] for j in range(K) if j != i]
        energy = float(np.mean([1.0 - r for r in others])) if others else 0.0
        boundary = sum(
            1 for j in range(K)
            if j != i and adj[i, j] > near_threshold and labels[j] != labels[i]
        )
        attractors.append({
            # forced → global stamp; faithful → per-memory local / antipode count
            "cluster_count": int(pos_counts[i]),
            "neg_cluster_count": int(neg_counts[i]),
            "boundary_length": int(boundary),         # FORCED: inter-memory graph
            "energy": round(energy, 6),               # APPROX: uncalibrated spread
        })

    return DecoderState(
        query=query,
        attractors=attractors,
        pairwise_distances=pairwise,
        landscape=landscape,
        # confidence / co_activated / uncertain / interference_state /
        # scm_openness / query_seed_corr deliberately left at defaults — the
        # caller fills them from the real CA capture so both arms share them.
    )


# Fields that can be reverted to the real CA value via ``hold_real`` — the four
# per-memory struct keys plus the two state-level fields.  Holding a field puts
# the CA's own value into Arm B, so that field can no longer drive the divergence.
_HOLDABLE_FIELDS = (*_TOPO_STRUCT_KEYS, "landscape", "pairwise")


def _build_embedding_arm(
    query: str,
    hits: list[dict],
    real_state: DecoderState,
    emb_state: DecoderState,
    *,
    hold_real: tuple[str, ...] = (),
) -> DecoderState:
    """Treatment arm: embedding topology + everything else frozen to the real run.

    The ONLY thing differing from Arm A (the real CA state) is the named,
    embedding-derived topology.  Structural keys are overwritten only when the
    real hit actually carried them (storage re-attaches just a subset at recall
    time, storage.py:385), so Arm A and Arm B prompts differ in *value*, never
    in *shape*.  Held CA-grid + three-grid fields are copied verbatim.

    hold_real : fields reverted to the real CA value instead of the embedding
        analog.  Each name in ``_HOLDABLE_FIELDS``.  Holding {cluster_count,
        neg_cluster_count} is Variant B2 (real value injected); holding a single
        field is one leave-one-out step of the field-level isolation sweep.
    """
    hold = set(hold_real)
    attractors: list[dict] = []
    for hit, emb_att in zip(hits, emb_state.attractors):
        merged = {k: hit[k] for k in _IDENTITY_KEYS if k in hit}
        # structural fields — only where the real hit actually carried them; held
        # fields take the CA value, the rest take the embedding analog.
        for k in _TOPO_STRUCT_KEYS:
            if hit.get(k) is None:
                continue
            if k in hold:
                merged[k] = hit[k]              # real CA value (reverted)
            elif k in emb_att:
                merged[k] = emb_att[k]          # embedding analog
        # held CA-grid + three-grid fields — copied from the real capture verbatim
        for k in (*_HELD_GRID_KEYS, *_THREE_GRID_KEYS):
            if hit.get(k) is not None:
                merged[k] = hit[k]
        attractors.append(merged)

    landscape = real_state.landscape if "landscape" in hold else emb_state.landscape
    pairwise = (
        list(real_state.pairwise_distances) if "pairwise" in hold
        else list(emb_state.pairwise_distances)
    )
    return DecoderState(
        query=query,
        attractors=attractors,
        confidence=real_state.confidence,                 # frozen (== Arm A)
        co_activated=list(real_state.co_activated),        # frozen
        uncertain=real_state.uncertain,                    # frozen
        interference_state=real_state.interference_state,  # held (three-grid)
        scm_openness=real_state.scm_openness,              # held (three-grid)
        pairwise_distances=pairwise,                       # embedding-derived | held
        landscape=landscape,                               # embedding-derived | held
        query_seed_corr=None,                              # frozen (omitted)
    )


def _diff_states(
    real_state: DecoderState, emb_state: DecoderState
) -> list[tuple[str, float]]:
    """Rank topology fields by how much CA and embedding disagree (normalized).

    Only the *varied* fields can differ; held fields are equal by construction.
    A high-ranked field points at the next refinement: a field-level ablation.
    """
    diffs: list[tuple[str, float]] = []

    # landscape: categorical → 0.0 same label, 1.0 different
    diffs.append((
        "landscape",
        0.0 if real_state.landscape == emb_state.landscape else 1.0,
    ))

    # mean|r| over the pairwise lists (the quantity the classifier reads)
    def _mean_abs_r(state: DecoderState) -> float:
        ps = state.pairwise_distances
        return sum(abs(r) for _, _, r in ps) / len(ps) if ps else 0.0

    diffs.append((
        "pairwise_mean_abs_r",
        abs(_mean_abs_r(real_state) - _mean_abs_r(emb_state)),
    ))

    # per-memory structural fields: mean normalized abs delta across memories
    for k in _TOPO_STRUCT_KEYS:
        deltas: list[float] = []
        for ra, ea in zip(real_state.attractors, emb_state.attractors):
            rv, ev = ra.get(k), ea.get(k)
            if rv is None or ev is None:
                continue
            denom = max(abs(rv), abs(ev), 1e-9)
            deltas.append(abs(rv - ev) / denom)
        if deltas:
            diffs.append((k, sum(deltas) / len(deltas)))

    diffs.sort(key=lambda kv: kv[1], reverse=True)
    return diffs


# ── Embedding + divergence ────────────────────────────────────────────────────


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def _offdiag_divergence(cos: np.ndarray) -> tuple[float, float, float]:
    """min / mean / max of (1 - cosine) over the strict upper triangle."""
    n = cos.shape[0]
    iu = np.triu_indices(n, k=1)
    div = 1.0 - cos[iu]
    return float(div.min()), float(div.mean()), float(div.max())


def _normalize(vecs: np.ndarray) -> np.ndarray:
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


def _cross_divergence(
    a_vecs: np.ndarray, b_vecs: np.ndarray
) -> tuple[tuple[float, float, float], np.ndarray]:
    """(min, mean, max), full matrix of (1 - cosine) over every Arm-A × Arm-B pair."""
    cross = 1.0 - (_normalize(a_vecs) @ _normalize(b_vecs).T)
    return (float(cross.min()), float(cross.mean()), float(cross.max())), cross


# ── Directional / quality grading (gated step [3]) ────────────────────────────
#
# Distinctness (steps [1]/[2]) only proves CA topology is DIFFERENT.  This asks
# whether it is BETTER: an LLM judge, FULLY BLIND (no query, no reference, no
# pairwise key — only the two candidate answers), with the FIRST/SECOND
# presentation order alternated to cancel position bias, scores which answer is
# better on its own merits.  The default contrast is Arm A (full CA) vs the
# iso_pairwise arm (pairwise reverted to CA), so the grade isolates the decode
# signal the CA's attractor correlations actually carry.

_JUDGE_SYSTEM = (
    "You are a careful, impartial evaluator. You are given two candidate answers "
    "(FIRST and SECOND) to the same withheld question. You are given no question, no "
    "reference material and no scoring key — judge the answers only on their own "
    "merits: clarity, coherence, specificity and internal consistency. Do not reward "
    "length for its own sake. If the two are equally good, answer TIE. Reply with "
    "exactly one line 'VERDICT: FIRST' | 'VERDICT: SECOND' | 'VERDICT: TIE', then one "
    "short justifying sentence."
)


def _judge_prompt(first: str, second: str) -> str:
    return (
        "--- ANSWER FIRST ---\n" + first
        + "\n\n--- ANSWER SECOND ---\n" + second
        + "\n\nWhich answer is better on its own merits?"
    )


def _parse_verdict(text: str) -> str:
    up = text.upper()
    for token in ("FIRST", "SECOND", "TIE"):
        if f"VERDICT: {token}" in up:
            return token
    # lenient fallback if the model dropped the prefix
    for token in ("TIE", "FIRST", "SECOND"):
        if token in up:
            return token
    return "TIE"


def _grade_direction(
    a_outputs: list[str],
    b_outputs: list[str],
    judge_model: str,
    ollama_url: str,
    *,
    max_pairs: int = 4,
    judge_backend: str = "ollama",
) -> dict:
    """Score whether Arm A (full CA) decodes BETTER than the contrast arm.

    Fully blind: the judge sees only the two candidate answers — no query, no
    recalled-memory reference, no pairwise key — and picks the better answer on its
    own merits.  For up to ``max_pairs`` aligned (A_i, B_i) decode pairs the
    presentation order is alternated so a position-biased judge nets to zero.
    Returns the signed margin per pair (+1 CA better, -1 CA worse, 0 tie) and the
    mean — positive mean ⇒ the CA arm is directionally better.

    judge_backend : "ollama" (local, via ``ollama_url``) or "claude" (Anthropic
        API; ``ollama_url`` is then unused and ``judge_model`` is a Claude model id).
    """
    n = min(len(a_outputs), len(b_outputs), max_pairs)
    margins: list[int] = []
    details: list[dict] = []
    flips = 0
    for k in range(n):
        flip = bool(k % 2)  # alternate which arm is shown FIRST
        first, second = (b_outputs[k], a_outputs[k]) if flip else (a_outputs[k], b_outputs[k])
        prompt = _judge_prompt(first, second)
        if judge_backend == "claude":
            raw = _decode_claude(prompt, judge_model, system=_JUDGE_SYSTEM)
        else:
            raw = _decode(prompt, judge_model, ollama_url, system=_JUDGE_SYSTEM)
        verdict = _parse_verdict(raw)
        if verdict == "TIE":
            signed = 0
        else:
            ca_is_first = not flip
            signed = 1 if ((verdict == "FIRST") == ca_is_first) else -1
        if verdict != "TIE":
            flips += int(flip)
        margins.append(signed)
        details.append({"pair": k, "flip": flip, "verdict": verdict, "signed": signed})
        mark = "+" if signed > 0 else "-" if signed < 0 else "="
        print(f"[grade] pair{k} flip={int(flip)} -> {verdict} (CA {mark})")
    mean = sum(margins) / len(margins) if margins else 0.0
    return {
        "judge_model": judge_model, "judge_backend": judge_backend, "n": len(margins),
        "margins": margins, "mean": mean, "details": details,
    }


def _print_matrix(label: str, names: list[str], cos: np.ndarray) -> None:
    print(f"\n{label} — divergence matrix (1 - cosine):")
    w = max(len(n) for n in names)
    print(" " * (w + 2) + " ".join(f"{i:>6}" for i in range(len(names))))
    for i, name in enumerate(names):
        cells = " ".join(f"{1.0 - cos[i, j]:6.3f}" for j in range(len(names)))
        print(f"  {name:<{w}} {cells}")


# ── Main measurement ──────────────────────────────────────────────────────────


def run_isolation(
    query: str,
    model: str,
    ollama_url: str,
    recall_k: int,
    data_dir,
) -> dict:
    from wheeler_memory.embedding import embed_text_batch
    from wheeler_memory.interference import recall_with_interference

    # 1. Real recall path, captured once.  Mirrors WheelerPrimaryAgent.run().
    print(f"[capture] query={query!r}  model={model}")
    hits, interference_state, scm_openness = recall_with_interference(
        query, top_k=recall_k, data_dir=data_dir,
        encoder="blended", use_embedding=True,
    )
    if not hits:
        raise SystemExit(
            "Recall returned no memories — store some first (scripts/wheeler_store.py) "
            "or point --data-dir at a populated store."
        )

    real_state = extract_state(
        query, hits, interference_state=interference_state,
        scm_openness=scm_openness, data_dir=data_dir,
    )
    print(
        f"[capture] {len(hits)} memories  confidence={real_state.confidence:.3f} "
        f"uncertain={real_state.uncertain}  landscape={real_state.landscape}  "
        f"interference={interference_state or 'none'}  scm_openness={scm_openness:.2f}"
    )

    # 2. Freeze memory identity + the derived confidence context for the run.
    frozen_hits = [
        {k: h[k] for k in _IDENTITY_KEYS if k in h} for h in hits
    ]
    base_confidence = real_state.confidence
    base_uncertain = real_state.uncertain
    base_co_activated = list(real_state.co_activated)

    # 3. Synthetic topologies + a control overlay (the real topology, reused 6x).
    overlays = _overlays()

    real_overlay = TopologyOverlay(
        name="CONTROL_real_topology",
        landscape=real_state.landscape or "ISOLATED",
        interference_state=real_state.interference_state,
        scm_openness=real_state.scm_openness,
        per_memory={
            # carry whatever structural fields the live hits actually had;
            # missing ones simply won't render, matching production behaviour.
            k: hits[0].get(k)
            for k in (
                "energy", "grid_entropy", "cluster_count", "neg_cluster_count",
                "boundary_length", "alive_fraction", "correlation_with_stored",
                "grounded_frac", "absorbed_frac", "unconsolidated_frac",
                "contested_frac",
            )
            if hits[0].get(k) is not None
        },
        pairwise_r=None,  # will instead reuse the real pairwise distances below
    )

    # 4. TREATMENT: 6 distinct topologies, frozen memories, swap at the seam.
    #    Each overlay is decoded once, so there is no per-prompt rep to drop; instead
    #    prime the session once (discarded) so the first overlay is not a cold-start
    #    relative to the rest — the same warmup artifact _decode_arm drops per arm.
    _prime = _build_state(
        query, frozen_hits, base_confidence, base_uncertain,
        base_co_activated, overlays[0],
    )
    _decode(format_state(_prime), model, ollama_url)
    print("[treat/warmup] prime -> discarded")
    treat_names: list[str] = []
    treat_outputs: list[str] = []
    for ov in overlays:
        state = _build_state(
            query, frozen_hits, base_confidence, base_uncertain,
            base_co_activated, ov,
        )
        prompt = format_state(state)            # ← the real assembly seam
        out = _decode(prompt, model, ollama_url)  # ← the real Ollama call (temp 0)
        treat_names.append(ov.name)
        treat_outputs.append(out)
        print(f"[treat] {ov.name:<28} -> {len(out):4d} chars")

    # 5. CONTROL: identical memories AND identical topology, 6 repeats.
    #    Reuse the real captured topology verbatim (incl. real pairwise list).
    control_state = DecoderState(
        query=query,
        attractors=[
            {**{k: h[k] for k in _IDENTITY_KEYS if k in h},
             **real_overlay.per_memory}
            for h in hits
        ],
        confidence=base_confidence,
        co_activated=base_co_activated,
        uncertain=base_uncertain,
        interference_state=real_overlay.interference_state,
        scm_openness=real_overlay.scm_openness,
        pairwise_distances=list(real_state.pairwise_distances),
        landscape=real_overlay.landscape,
        query_seed_corr=None,
    )
    control_prompt = format_state(control_state)
    # Floor: identical prompt repeated, rep 0 (cold start) discarded so the
    # deterministic floor reads a true ~0 instead of being inflated by warmup.
    control_outputs = _decode_arm(
        control_prompt, model, ollama_url, len(overlays), "ctrl"
    )
    control_names = [f"rep{i}" for i in range(len(control_outputs))]

    # 6. Embed everything with the existing MiniLM encoder, in one batch.
    all_vecs = embed_text_batch(treat_outputs + control_outputs)
    t_vecs = all_vecs[: len(treat_outputs)]
    c_vecs = all_vecs[len(treat_outputs) :]

    t_cos = _cosine_matrix(t_vecs)
    c_cos = _cosine_matrix(c_vecs)
    t_min, t_mean, t_max = _offdiag_divergence(t_cos)
    c_min, c_mean, c_max = _offdiag_divergence(c_cos)

    return {
        "treat_names": treat_names,
        "control_names": control_names,
        "t_cos": t_cos,
        "c_cos": c_cos,
        "treatment": (t_min, t_mean, t_max),
        "control": (c_min, c_mean, c_max),
        "treat_outputs": treat_outputs,
        "control_outputs": control_outputs,
    }


# Arm B variants: (cluster_mode for embedding_topology, fields held to real CA value).
#   forced   — legacy global/imposed analog (baseline column, back-compat).
#   faithful — de-forced per-memory local clustering (Variant B1).
#   real     — faithful geometry but cluster_count/neg held to the CA value
#              (Variant B2); identical to a leave-one-out isolation on cluster_count.
_VARIANT_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "forced": ("forced", ()),
    "faithful": ("faithful", ()),
    "real": ("faithful", ("cluster_count", "neg_cluster_count")),
}


def run_ablation(
    query: str,
    model: str,
    ollama_url: str,
    recall_k: int,
    data_dir,
    reps: int = 6,
    variants: tuple[str, ...] = ("forced", "faithful", "real"),
    field_isolation: bool = False,
    grade: bool = False,
    judge_model: str | None = None,
    judge_backend: str = "ollama",
) -> dict:
    """CA-vs-embedding topology ablation, de-contaminated and multi-variant.

    Arm A (reference) = the real CA-derived DecoderState; its post-warmup decodes
    double as the deterministic floor.  Each Arm-B *variant* is an embedding-derived
    topology built differently (see ``_VARIANT_SPEC``), with all non-topology fields
    frozen to the real capture.  Every arm is decoded ``reps + 1`` times with rep 0
    discarded (``_decode_arm``), so the floor is a true ~0 and a faithful analog that
    collapses cannot be masked.  All arms share the one clean floor.

    field_isolation : also run a leave-one-out sweep (revert each field to the CA
        value atop the faithful arm) to attribute the surviving divergence per field.
    grade : run the directional/quality judge on Arm A vs the faithful Arm B.
    """
    from wheeler_memory.embedding import embed_text, embed_text_batch
    from wheeler_memory.interference import recall_with_interference

    # 1. Real recall + CA-derived state, captured once (same path as run_isolation).
    print(f"[capture] query={query!r}  model={model}")
    hits, interference_state, scm_openness = recall_with_interference(
        query, top_k=recall_k, data_dir=data_dir,
        encoder="blended", use_embedding=True,
    )
    if not hits:
        raise SystemExit(
            "Recall returned no memories — store some first (scripts/wheeler_store.py) "
            "or point --data-dir at a populated store."
        )
    real_state = extract_state(
        query, hits, interference_state=interference_state,
        scm_openness=scm_openness, data_dir=data_dir,
    )
    print(
        f"[capture] {len(hits)} memories  confidence={real_state.confidence:.3f} "
        f"landscape={real_state.landscape}  interference={interference_state or 'none'}"
    )

    # 2. Embedding-derived topology (no CA ticks), forced + faithful constructions.
    #    Embeddings are never persisted, but hit['text'] is the exact stored source
    #    and MiniLM is deterministic, so re-embedding reproduces the recall geometry.
    query_emb = embed_text(query)
    memory_embs = embed_text_batch([h["text"] for h in hits])
    emb_by_mode = {
        "forced": embedding_topology(query_emb, memory_embs, query=query, cluster_mode="forced"),
        "faithful": embedding_topology(query_emb, memory_embs, query=query, cluster_mode="faithful"),
    }
    for mode, st in emb_by_mode.items():
        pos = [a["cluster_count"] for a in st.attractors]
        neg = [a["neg_cluster_count"] for a in st.attractors]
        spread = (
            f"+{min(pos)}..{max(pos)} / -{min(neg)}..{max(neg)}" if mode == "faithful"
            else f"+{pos[0]} / -{neg[0]} (global)"
        )
        print(f"[emb-topo:{mode:<8}] landscape={st.landscape}  "
              f"pairs={len(st.pairwise_distances)}  clusters={spread}")

    # 3. Decode every arm, each with rep 0 discarded.  Arm A first (it is the floor).
    arm_outputs: dict[str, list[str]] = {}
    arm_outputs["A_floor"] = _decode_arm(
        format_state(real_state), model, ollama_url, reps, "armA/floor"
    )

    variant_emb: dict[str, DecoderState] = {}
    for v in variants:
        cluster_mode, hold = _VARIANT_SPEC[v]
        emb_state = emb_by_mode[cluster_mode]
        variant_emb[v] = emb_state
        arm = _build_embedding_arm(query, hits, real_state, emb_state, hold_real=hold)
        arm_outputs[f"B_{v}"] = _decode_arm(
            format_state(arm), model, ollama_url, reps, f"armB/{v}"
        )

    # Field-level isolation: leave-one-out, each field reverted to the CA value
    # atop the faithful (de-forced) embedding arm.
    iso_fields: tuple[str, ...] = ()
    if field_isolation:
        iso_fields = tuple(
            f for f in _HOLDABLE_FIELDS
            if f in ("landscape", "pairwise")
            or any(h.get(f) is not None for h in hits)
        )
        for f in iso_fields:
            arm = _build_embedding_arm(
                query, hits, real_state, emb_by_mode["faithful"], hold_real=(f,)
            )
            arm_outputs[f"iso_{f}"] = _decode_arm(
                format_state(arm), model, ollama_url, reps, f"iso/-{f}"
            )

    # 4. Embed every output once (one MiniLM batch), then slice back per arm.
    order = list(arm_outputs)
    flat = [o for name in order for o in arm_outputs[name]]
    vecs = embed_text_batch(flat)
    arm_vecs: dict[str, np.ndarray] = {}
    idx = 0
    for name in order:
        k = len(arm_outputs[name])
        arm_vecs[name] = vecs[idx:idx + k]
        idx += k

    # 5. One clean floor (within-Arm-A, warmup already dropped); A<->B per variant.
    a_vecs = arm_vecs["A_floor"]
    floor = _offdiag_divergence(_cosine_matrix(a_vecs))

    variant_results: dict[str, dict] = {}
    for v in variants:
        stats, cross = _cross_divergence(a_vecs, arm_vecs[f"B_{v}"])
        variant_results[v] = {"ab": stats, "cross": cross}

    # State-level field-divergence ranking from the FAITHFUL (honest) analog.
    field_diffs = _diff_states(real_state, emb_by_mode["faithful"])

    faithful_mean = variant_results.get("faithful", {}).get("ab", (0.0, 0.0, 0.0))[1]
    iso_results: dict[str, dict] = {}
    for f in iso_fields:
        stats, _cross = _cross_divergence(a_vecs, arm_vecs[f"iso_{f}"])
        # drop = how much the divergence FALLS when field f is reverted to the CA
        # value; a large drop means f carried the faithful-arm divergence.
        iso_results[f] = {"ab": stats, "drop": faithful_mean - stats[1]}

    grading = None
    if grade:
        # Re-aimed at pairwise: contrast Arm A (full CA) against the iso_pairwise
        # arm (pairwise reverted to CA) so the grade isolates the decode signal the
        # CA's attractor correlations carry.  Falls back to the faithful arm if the
        # leave-one-out sweep was not run (no iso_pairwise arm to grade against).
        grade_target = "iso_pairwise" if "iso_pairwise" in arm_outputs else "B_faithful"
        if grade_target in arm_outputs:
            grading = _grade_direction(
                arm_outputs["A_floor"], arm_outputs[grade_target],
                judge_model or model, ollama_url,
                judge_backend=judge_backend,
            )
            grading["grade_target"] = grade_target
        else:
            print("[grade] skipped: no iso_pairwise or B_faithful arm present — "
                  "run with --field-isolation or include the faithful variant")

    return {
        "reps": reps,
        "variants": list(variants),
        "arm_outputs": arm_outputs,
        "floor": floor,
        "floor_n": int(a_vecs.shape[0]),
        "variant_results": variant_results,
        "field_diffs": field_diffs,
        "iso_fields": list(iso_fields),
        "iso_results": iso_results,
        "grading": grading,
    }


def _report(res: dict) -> None:
    t_min, t_mean, t_max = res["treatment"]
    c_min, c_mean, c_max = res["control"]

    _print_matrix("TREATMENT (topology swapped)", res["treat_names"], res["t_cos"])
    _print_matrix("CONTROL (identical inputs x6)", res["control_names"], res["c_cos"])

    print("\n" + "=" * 64)
    print("OFF-DIAGONAL DIVERGENCE  (1 - cosine, MiniLM embeddings)")
    print("=" * 64)
    print(f"  control (det. floor):  min={c_min:.4f}  mean={c_mean:.4f}  max={c_max:.4f}")
    print(f"  treatment (topology):  min={t_min:.4f}  mean={t_mean:.4f}  max={t_max:.4f}")

    # Verdict: treatment materially above the floor → signal.
    # "Material" = treatment mean exceeds floor mean by both an absolute margin
    # and a multiplicative one (guards against a near-zero floor inflating ratios).
    abs_margin = t_mean - c_mean
    floor = max(c_mean, 1e-4)
    ratio = t_mean / floor
    signal = abs_margin > 0.02 and ratio > 2.0

    print("\nVERDICT")
    print("-" * 64)
    print(f"  treatment mean - control mean = {abs_margin:+.4f}   (ratio {ratio:.1f}x)")
    if signal:
        print(
            "  >> SIGNAL: swapping the topology block moves the decoder output "
            "materially\n     beyond the deterministic floor. The block CARRIES SIGNAL."
        )
    else:
        print(
            "  >> DECORATIVE: topology-swap divergence is within noise of the "
            "deterministic\n     floor. The decoder largely IGNORES the topology block."
        )
    print("=" * 64)


def _report_ablation(res: dict) -> None:
    reps = res["reps"]
    f_min, f_mean, f_max = res["floor"]
    floor_guard = max(f_mean, 1e-4)

    print("\n" + "=" * 64)
    print("CA-vs-EMBEDDING TOPOLOGY DIVERGENCE  (1 - cosine, MiniLM)")
    print("=" * 64)
    print(f"  reps/arm={reps}  (rep 0 discarded per arm as cold-start warmup)")
    print(f"  floor (Arm A self, det., n={res['floor_n']}): "
          f"min={f_min:.4f}  mean={f_mean:.4f}  max={f_max:.4f}")
    if f_mean > 0.02:
        print("  !! floor mean > 0.02 — warmup discard did NOT fully clean the floor;")
        print("     ratios below sit on a contaminated baseline — treat with caution.")

    print("\n  HELD CONSTANT across both arms (NOT under test):")
    print("    interference_state, scm_openness, grounded/absorbed/unconsolidated/")
    print("    contested_frac, grid_entropy, alive_fraction, correlation_with_stored")
    print("                                       (intrinsically CA / no clean analog)")
    print("  RESIDUAL FORCING still active in Arm B (flagged, never hidden):")
    print("    boundary_length — inter-memory graph stands in for intra-grid  (FORCED)")
    print("    energy          — uncalibrated mean (1 - r) spread             (APPROX)")
    print("    cluster_count/neg: forced→global+imposed-sign; "
          "faithful→per-memory local/antipode (DE-FORCED)")

    # Per-variant verdict against the ONE shared clean floor (dual guard, as before).
    print("\n  VARIANT DIVERGENCE vs shared floor (guard: margin>0.02 AND ratio>2x):")
    print(f"    {'variant':<10}{'A<->B mean':>12}{'margin':>10}{'ratio':>9}   verdict")
    for v in res["variants"]:
        ab_mean = res["variant_results"][v]["ab"][1]
        margin = ab_mean - f_mean
        ratio = ab_mean / floor_guard
        signal = margin > 0.02 and ratio > 2.0
        verdict = "DISTINCT" if signal else "collapsed->floor"
        print(f"    {v:<10}{ab_mean:>12.4f}{margin:>+10.4f}{ratio:>8.1f}x   {verdict}")

    print("\n  READING (the two faithful variants bracket the interpretation):")
    print("    forced   = legacy baseline — compare to the prior 2.1x run")
    print("    faithful = de-forced; staying DISTINCT ⇒ divergence is NOT a")
    print("               cluster_count forcing artifact (a real CA signal survives)")
    print("    real     = CA cluster_count injected; if this collapses while faithful")
    print("               stays up, the surviving signal lives in cluster_count itself")

    print("\n  STATE-LEVEL FIELD DIVERGENCE (faithful analog, pre-decode, ranked):")
    for k, v in res["field_diffs"]:
        print(f"    {k:<24} {v:.3f}")
    print("    (numeric field disagreement ONLY — not decode influence; a field can")
    print("     top this ranking yet be ignored by the decoder. See the leave-one-out")
    print("     DECODE-LEVEL ISOLATION below for what the decoder actually acts on.)")

    if res["iso_fields"]:
        print("\n  DECODE-LEVEL FIELD ISOLATION (leave-one-out atop the faithful arm):")
        print(f"    {'field reverted->CA':<22}{'A<->B mean':>12}{'drop':>10}")
        for f in res["iso_fields"]:
            ab = res["iso_results"][f]["ab"][1]
            drop = res["iso_results"][f]["drop"]
            print(f"    {f:<22}{ab:>12.4f}{drop:>+10.4f}")
        print("    (large positive drop ⇒ that field carried the faithful-arm divergence)")

    if res["grading"] is not None:
        g = res["grading"]
        verdict = "CA BETTER" if g["mean"] > 0 else "CA WORSE" if g["mean"] < 0 else "TIE"
        target = g.get("grade_target", "B_faithful")
        backend = g.get("judge_backend", "ollama")
        print(f"\n  DIRECTIONAL/QUALITY GRADE (judge={g['judge_model']} via {backend}):")
        print(f"    A_floor (full CA)  vs  {target}")
        print(f"    n={g['n']}  mean signed margin={g['mean']:+.3f}  ->  {verdict}")
        print("    (+1 CA better answer, -1 worse, 0 tie; FULLY BLIND, position-debiased)")
    print("=" * 64)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--recall-k", type=int, default=5)
    p.add_argument("--data-dir", default=None)
    p.add_argument(
        "--dump", action="store_true",
        help="print the full decoder outputs after the report",
    )
    p.add_argument(
        "--ablation", action="store_true",
        help="run the CA-vs-embedding topology ablation instead of the "
             "decode-sensitivity isolation",
    )
    p.add_argument(
        "--reps", type=int, default=6,
        help="decodes per arm in --ablation mode (+1 warmup discarded; Arm A "
             "post-warmup reps also serve as the floor)",
    )
    p.add_argument(
        "--variants", default="forced,faithful,real",
        help="comma list of Arm-B variants to run "
             f"(choices: {','.join(_VARIANT_SPEC)})",
    )
    p.add_argument(
        "--field-isolation", action="store_true",
        help="also run the leave-one-out field-isolation sweep atop the faithful arm",
    )
    p.add_argument(
        "--grade", action="store_true",
        help="also run the directional/quality judge on Arm A vs the faithful Arm B",
    )
    p.add_argument(
        "--judge-model", default=None,
        help="model for --grade (default: --model for ollama, claude-opus-4-8 for "
             "claude); use a stronger model for a sharper directional verdict",
    )
    p.add_argument(
        "--judge-backend", default="ollama", choices=("ollama", "claude"),
        help="judge transport for --grade: 'ollama' (local, via --ollama-url) or "
             "'claude' (Anthropic API; set ANTHROPIC_API_KEY). Decoder arms always "
             "use ollama; only the judge changes. Pair with --field-isolation so the "
             "grade contrasts Arm A vs the iso_pairwise arm.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not _ollama_up(args.ollama_url):
        print(f"Ollama not reachable at {args.ollama_url}. Try: ollama serve", file=sys.stderr)
        return 2
    if not _model_present(args.ollama_url, args.model):
        print(f"Model {args.model!r} not pulled. Try: ollama pull {args.model}", file=sys.stderr)
        return 2

    if args.ablation:
        variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
        unknown = [v for v in variants if v not in _VARIANT_SPEC]
        if unknown:
            print(f"Unknown --variants {unknown}; choices: {list(_VARIANT_SPEC)}",
                  file=sys.stderr)
            return 2
        judge_model = args.judge_model
        if args.judge_backend == "claude" and not judge_model:
            judge_model = "claude-opus-4-8"
        if (args.judge_backend == "ollama" and judge_model
                and not _model_present(args.ollama_url, judge_model)):
            print(f"Judge model {judge_model!r} not pulled. "
                  f"Try: ollama pull {judge_model}", file=sys.stderr)
            return 2
        res = run_ablation(
            args.query, args.model, args.ollama_url, args.recall_k,
            args.data_dir, reps=args.reps, variants=variants,
            field_isolation=args.field_isolation, grade=args.grade,
            judge_model=judge_model, judge_backend=args.judge_backend,
        )
        _report_ablation(res)
        if args.dump:
            print("\n" + "#" * 64 + "\nFULL OUTPUTS\n" + "#" * 64)
            for name, outs in res["arm_outputs"].items():
                for i, out in enumerate(outs):
                    print(f"\n--- {name} rep{i} ---\n{out}")
        return 0

    res = run_isolation(
        args.query, args.model, args.ollama_url, args.recall_k, args.data_dir,
    )
    _report(res)

    if args.dump:
        print("\n" + "#" * 64 + "\nFULL OUTPUTS\n" + "#" * 64)
        for name, out in zip(res["treat_names"], res["treat_outputs"]):
            print(f"\n--- {name} ---\n{out}")
        for name, out in zip(res["control_names"], res["control_outputs"]):
            print(f"\n--- {name} ---\n{out}")
    return 0


# ── pytest entry (skips cleanly without Ollama / embeddings) ──────────────────


def test_topology_isolation():
    import pytest

    from wheeler_memory.embedding import embed_available

    url = DEFAULT_OLLAMA_URL
    if not embed_available():
        pytest.skip("sentence-transformers not installed")
    if not _ollama_up(url):
        pytest.skip("Ollama not reachable")
    if not _model_present(url, DEFAULT_MODEL):
        pytest.skip(f"model {DEFAULT_MODEL} not pulled")

    try:
        res = run_isolation(DEFAULT_QUERY, DEFAULT_MODEL, url, 5, None)
    except SystemExit as e:
        pytest.skip(str(e))

    _report(res)
    # Measurement, not assertion: only sanity-check the shapes so a broken
    # harness fails loudly while the empirical verdict stays informational.
    assert res["t_cos"].shape == (6, 6)
    assert res["c_cos"].shape == (6, 6)


def test_ca_vs_embedding_ablation():
    import pytest

    from wheeler_memory.embedding import embed_available

    url = DEFAULT_OLLAMA_URL
    if not embed_available():
        pytest.skip("sentence-transformers not installed")
    if not _ollama_up(url):
        pytest.skip("Ollama not reachable")
    if not _model_present(url, DEFAULT_MODEL):
        pytest.skip(f"model {DEFAULT_MODEL} not pulled")

    reps = 4
    variants = ("forced", "faithful", "real")
    try:
        res = run_ablation(
            DEFAULT_QUERY, DEFAULT_MODEL, url, 5, None, reps=reps, variants=variants,
        )
    except SystemExit as e:
        pytest.skip(str(e))

    _report_ablation(res)
    # Warmup excluded: the floor and every arm hold EXACTLY `reps` post-warmup
    # decodes (rep 0 dropped), so a regression that re-admits the cold start fails.
    assert res["floor_n"] == reps
    assert len(res["arm_outputs"]["A_floor"]) == reps
    for v in variants:
        assert len(res["arm_outputs"][f"B_{v}"]) == reps
        vr = res["variant_results"][v]
        assert vr["cross"].shape == (reps, reps)
        assert len(vr["ab"]) == 3
    assert res["field_diffs"]  # state-level ranking populated


def test_embedding_topology_and_hold_real():
    """Pure-numpy coverage (no Ollama / sentence-transformers): the de-forced
    faithful analog and the hold_real revert mechanism."""
    rng = np.random.default_rng(0)
    dim = 16
    base = rng.standard_normal((5, dim)).astype(np.float32)
    base[1] = base[0] + 0.01 * rng.standard_normal(dim)  # 0,1,2 tightly grouped
    base[2] = base[0] + 0.01 * rng.standard_normal(dim)
    memory_embs = base
    query_emb = base[0].copy()

    forced = embedding_topology(query_emb, memory_embs, query="q", cluster_mode="forced")
    faithful = embedding_topology(query_emb, memory_embs, query="q", cluster_mode="faithful")

    # forced stamps ONE global count on every memory; faithful is per-memory.
    f_counts = [a["cluster_count"] for a in forced.attractors]
    assert len(set(f_counts)) == 1
    fa_counts = [a["cluster_count"] for a in faithful.attractors]
    assert all(isinstance(c, int) and c >= 0 for c in fa_counts)

    # hold_real reverts a field to the real CA value; without it, the embedding
    # analog is used.
    hits = [
        {"text": f"m{i}", "similarity": 0.5, "cluster_count": 99,
         "neg_cluster_count": 7, "boundary_length": 3, "energy": 0.1}
        for i in range(5)
    ]
    real_state = DecoderState(
        query="q", attractors=[], landscape="TIGHT",
        pairwise_distances=[(1, 2, 0.9)], confidence=0.5, co_activated=[],
        uncertain=False, interference_state="GROUNDED", scm_openness=0.5,
    )
    held = _build_embedding_arm("q", hits, real_state, faithful,
                                hold_real=("cluster_count",))
    assert all(a["cluster_count"] == 99 for a in held.attractors)
    free = _build_embedding_arm("q", hits, real_state, faithful, hold_real=())
    assert all(a["cluster_count"] != 99 for a in free.attractors)


if __name__ == "__main__":
    raise SystemExit(main())

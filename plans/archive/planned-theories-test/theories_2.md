# Planned Theory Tests — Set 2

> Discussed 2026-03-17. Analysis of 6 further theoretical models.

---

## 1. Trajectory Similarity as Second Retrieval Metric

**The theory**: Wheeler currently matches by attractor *destination* (Pearson on final frame). Two
inputs could take similar paths through dynamics even if they end at different attractors. Path-similar
inputs are thematically related concepts that crossed a basin boundary. That's a retrieval signal
being discarded entirely.

**What exists**:
- Bricks store the full `(Time, X, Y)` history — the trajectory data is on disk already
- `convergence_ticks` in index — coarse path length signal
- `oscillation.py` — detects late-stage role-space oscillation (partial path characterisation)
- `theories/metrics.py` — `energy()` per frame (could be computed along trajectory)

**What's missing**:
- No trajectory comparison function — need a way to measure path similarity between two histories
- Bricks are never read at recall time — only the final attractor `.npy` is loaded
- No secondary retrieval pass that uses trajectory after primary Pearson recall

**Approach**:
Downsample each brick history to N keyframes (already partially done in `consolidation.py`).
Compute per-tick Pearson between keyframe sequences. Use as a re-ranking signal after primary recall:
top-K by attractor similarity, re-ranked by trajectory similarity.

DTW (dynamic time warping) would handle different convergence speeds cleanly — two paths that took
different numbers of ticks to reach similar waypoints should still score as similar.

**Test to build**: `tests/theories/test_trajectory_recall.py`
- Store two concepts that should be thematically related but end at different attractors
- Assert primary recall (Pearson on attractor) ranks them low
- Assert secondary recall (trajectory similarity) ranks them high
- Measure: does trajectory re-ranking improve MMLU accuracy on ambiguous questions?

---

## 2. Fork Points as Ambiguity Markers

**The theory**: When a trajectory passes near a saddle point (the boundary between two basins),
the input was semantically ambiguous. Storing those near-misses lets you answer "what almost matched
this but went a different direction?" Ambiguity becomes explicit rather than silent.

**What exists**:
- `oscillation.py` — detects role-space oscillation (late-stage oscillation = near-miss saddle point)
- `evolve_and_interpret()` returns `state: OSCILLATING` for inputs that didn't cleanly converge
- Brick history captures the full trajectory including the oscillation period
- `theories/basin.py` — `find_basin_gaps()` maps where saddle regions exist topologically

**What's missing**:
- Fork points are not stored — oscillating trajectories are recorded but the saddle location isn't
  extracted or indexed
- No "near-miss" query: "what almost matched this?" is not a supported retrieval mode
- No explicit ambiguity flag on stored memories — OSCILLATING state is in brick metadata but not
  surfaced in recall results

**Approach**:
During evolution, detect the tick at which the trajectory comes closest to a saddle point (maximum
oscillation amplitude). Record the frame at that tick as the `fork_frame`. Store it alongside the
attractor. At recall time, also score query against stored `fork_frames` — a hit on a fork frame
means "this query is semantically near an ambiguous region."

**Test to build**: `tests/theories/test_fork_points.py`
- Create a trajectory that oscillates before converging (input near a basin boundary)
- Assert `fork_frame` is extracted and stored
- Query with an input known to be ambiguous between two concepts
- Assert the query's fork frame correlates with stored fork frames for both concepts
- Assert non-ambiguous queries do NOT hit fork frames

---

## 3. Every Query is a Write

**The theory**: There is no pure read mode. Every query reshapes the landscape. The cumulative
terrain IS the memory, not any individual attractor in it. This was a foundational reframe —
the goal was never a lookup table with separate read/write modes.

**What exists**:
- `hit_count` increments on recall — queries already write something (temperature/warmth)
- `warming.py` — co-recall associations update on every recall (queries write edges)
- `eviction.py` — temperature influences which attractors survive (queries vote on what persists)

**What's missing**:
- The attractor itself is never modified by queries — the landscape geometry is static
- Recall doesn't perturb nearby attractors (only the recalled attractor's temperature changes)
- No mechanism where repeated querying of a region gradually reinforces or reshapes that basin
- The daydream plan (see `ternary_dynamics_and_daydream.md`) partially addresses this but only
  during explicit daydream cycles, not at query time

**The full reframe requires**:
At recall time, for each top-K hit: apply a small positive perturbation to the stored attractor
in the direction of the query attractor (blend + partial re-evolve). Hot memories resist
perturbation (strong basin). Cold memories drift toward the query (landscape reshaping).
This makes every recall a write proportional to `(1 - temperature)`.

```python
# At recall time, after scoring:
if hit.temperature < WARM_THRESHOLD:
    alpha = QUERY_WRITE_STRENGTH * (1 - hit.temperature)
    reshaped = (1 - alpha) * stored_attractor + alpha * query_attractor
    # re-evolve one step only — don't fully converge, just nudge
    np.save(attractor_path, apply_ca_dynamics(reshaped))
```

**Test to build**: `tests/theories/test_query_writes.py`
- Store a memory, record its attractor
- Recall it repeatedly with a nearby but distinct query
- Assert the stored attractor drifts toward the query attractor over N recalls
- Assert a hot memory drifts less than a cold memory (temperature gates plasticity)
- Assert the drift is bounded (doesn't escape its basin)

---

## 4. Wheeler as Semantic Gyroscope

**The theory**: Wheeler is a real-time stability instrument for text — not just scoring it, but
giving it an *orientation* in symbolic space and detecting drift. SCM variables as attitude sensors.
A gyroscope doesn't tell you where you are; it tells you how fast you're turning and in what direction.

**What exists**:
- `theories/metrics.py` — `energy()` (how far from attractor), `hallucination_score()` (drifting
  without basin pull), `topology_consistency()` (how well basins tile space)
- `oscillation.py` — role-space oscillation = rotational instability in symbolic space
- `decoder.py` — extracts confidence from top Pearson similarity (static, per-query)

**What's missing**:
- No *sequence* tracking — gyroscope needs a time series of orientations, not a single score
- No drift rate: how fast is meaning changing across a sequence of inputs or outputs?
- No orientation vector: where in symbolic space is this text, relative to known attractors?
- No real-time mode: currently all metrics are computed post-hoc per query, not as a stream

**The gyroscope instrument**:
Input: a stream of text segments (e.g. LLM output tokens, conversation turns, document paragraphs)
Output per segment: `{energy, orientation_vector, drift_rate, oscillation_period or None}`

```python
class SemanticGyroscope:
    def __init__(self, known_attractors):
        self.history = []  # last N orientation vectors

    def update(self, text: str) -> dict:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(frame)
        orientation = result["attractor"].flatten()  # position in symbolic space
        e = energy(result["attractor"])
        drift = np.mean(np.abs(orientation - self.history[-1])) if self.history else 0.0
        osc = detect_oscillation(result["history"])
        self.history.append(orientation)
        return {"energy": e, "drift_rate": drift, "oscillating": osc["oscillating"]}
```

**Test to build**: `tests/theories/test_gyroscope.py`
- Feed a coherent sequence of related sentences → assert drift_rate stays low
- Feed a sequence that shifts topic suddenly → assert drift_rate spikes at the shift
- Feed hallucinated text → assert energy stays high, oscillation detected
- Feed grounded text → assert energy drops, orientation stabilizes near known attractors

---

## 5. SCM Telemetry over IPv6-like Addressing

**The theory**: Encode symbolic stability metrics into a broadcast-able address structure. Not just
scoring content but making meaning-health a network-layer property. Every piece of content carries
its own symbolic telemetry that infrastructure can read without understanding the content.

**What exists**:
- `theories/metrics.py` — the metrics that would be encoded: energy, basin_width, context_weight,
  hallucination_score
- Fractal cube address space (see `fractal_cube_address_space.md`) — the address structure this
  would sit on top of
- `text_to_hex()` in `hashing.py` — content addressing already exists (SHA-256 = 32 bytes)

**What's missing**:
- Everything. This is the most architectural of the theories.
- No telemetry encoding scheme
- No broadcast/transmission layer
- No infrastructure reader

**The encoding sketch**:
IPv6 is 128 bits. A Wheeler telemetry address could be:

```
[32 bits: content hash prefix]
[16 bits: energy * 65535 (fixed point)]
[16 bits: basin_width * 65535]
[16 bits: convergence_ticks, capped at 65535]
[16 bits: oscillation_period * 1000 or 0xFFFF if none]
[32 bits: top Pearson similarity * 1e9 (fixed point)]
```

Any infrastructure node reading this address knows the symbolic health of the content without
decoding or understanding it. Routers could deprioritise high-energy (unstable) content.
Filters could drop content with hallucination signatures.

**Connection to fractal cube**: the content hash prefix IS the fractal cube address root.
Depth of known traversal could be encoded in remaining bits.

**Test to build**: `tests/theories/test_scm_telemetry.py`
- Encode a known-stable text as a telemetry address
- Encode a known-hallucinated text
- Assert the two addresses differ in predictable bit fields (energy bits, oscillation bits)
- Assert the content hash prefix matches `text_to_hex()` output prefix
- Round-trip: decode address back to metrics, assert within tolerance of directly-computed metrics

---

## 6. Hallucination vs Synthesis Discrimination

**The theory**: A synthesized frame has a real basin of attraction — it runs the dynamics and
stabilizes. A hallucination is a confident output with no basin — it doesn't survive symbolic pressure.
The convergence test is the literal distinguisher. This is SCM's core axiom made executable.

**What exists**:
- `evolve_and_interpret()` returns `state: CONVERGED | OSCILLATING | CHAOTIC` — this IS the test
- `theories/metrics.py` — `hallucination_score()` = `energy * (1 - max_correlation)` — high score
  = drifting without basin pull
- `theories/synthesis.py` — `synthesize_from_gap()` generates synthesized frames that converge

**What's missing**:
- The test is not formally run on LLM outputs — no pipeline that takes text → evolve → classify
- No threshold calibrated: at what energy / convergence_ticks do we call something a hallucination
  vs synthesis?
- `hallucination_score()` is not wired into any agent or decoder loop

**The test is almost free**:
```python
def classify_output(text: str, known_attractors: list) -> str:
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)

    if result["state"] == "CONVERGED":
        score = hallucination_score(result["attractor"], known_attractors)
        if score < HALLUCINATION_THRESHOLD:
            return "SYNTHESIS"   # converged AND near a known basin
        else:
            return "NOVEL"       # converged but in unknown territory
    else:
        return "HALLUCINATION"   # didn't stabilize = no basin = confident but groundless
```

Three categories, not two: synthesis (grounded), novel (stable but unknown), hallucination (unstable).

**Test to build**: `tests/theories/test_hallucination_discrimination.py`
- Known grounded statement → assert SYNTHESIS
- Known hallucination (e.g. false fact about physics) → assert HALLUCINATION or high score
- Novel but coherent synthesis (apple test output) → assert NOVEL or SYNTHESIS
- Calibration: sweep HALLUCINATION_THRESHOLD, find value that maximises true positive rate
- Regression: if ternary fix changes convergence dynamics, discrimination accuracy must hold

---

## Implementation Priority

| Theory | Existing Code | Test Difficulty | Expected Impact |
|--------|--------------|-----------------|-----------------|
| Hallucination Discrimination | ✓✓ nearly free | Low — 3 lines + threshold | Very High — SCM core axiom executable |
| Semantic Gyroscope | ✓ metrics exist | Low-Medium — add sequence tracking | High — real-time stability instrument |
| Fork Points / Ambiguity | ✓ oscillation exists | Medium — extract + store fork frame | Medium — richer recall signal |
| Trajectory Similarity | ✓ data on disk | Medium — add brick read to recall | High — second retrieval dimension |
| Every Query is a Write | ✓ partial (warmth) | Medium-High — modify recall pipeline | Very High — foundational reframe |
| SCM Telemetry / IPv6 | ✗ encoding only | High — architectural | High (long-term) |

**Start with Hallucination Discrimination** — three lines of code, uses fully existing functions,
makes SCM's core axiom literally executable. Run it on MMLU incorrect answers and measure.

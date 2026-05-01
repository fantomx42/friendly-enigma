# Work Priority — Wheeler Memory

> Organised 2026-03-17. Covers all 5 plan files.

---

## Blocker × Critical

Work that is either prerequisite to everything else, or has the highest leverage/effort ratio.

| Item | File | Why critical |
|------|------|-------------|
| `is_flat` dynamics fix | `ternary_dynamics_and_daydream.md` | 0 is not a stable fixed point. Every downstream theory (ternary semantics, 0=uncertainty signal, Lichtenberg, fractal cube face geometry, daydream stability) assumes ternary works. Nothing built on top is correct until this is fixed. |
| Hallucination vs Synthesis discrimination | `theories_2.md` § 6 | 3 lines of code using fully existing functions. Makes SCM's core axiom literally executable. Run on MMLU wrong answers immediately. Highest leverage/effort ratio in the entire backlog. |
| `--mode recall-text` reconstruction scoring | `reconstruction_scoring.md` | Current benchmark focus. The path where "it from bit" becomes the scoring mechanism, not just storage philosophy. Explicitly the next MMLU step. |

---

## Major / High

High impact, moderate effort. Core system capabilities.

| Item | File | Notes |
|------|------|-------|
| Daydream mechanism | `ternary_dynamics_and_daydream.md` | Enables landscape plasticity. Cold memories drift, crowded basins erode, landscape self-organises. Prerequisite for "every query is a write" to function at idle time. |
| Every Query is a Write | `theories_2.md` § 3 | Foundational reframe — queries perturb nearby attractors proportional to `(1 - temperature)`. Changes recall architecture. Hot memories resist, cold memories drift toward query. |
| Trajectory similarity as second retrieval metric | `theories_2.md` § 1 | Data already exists on disk (bricks). Adds a second retrieval dimension beyond Pearson on final attractor. Path-similar inputs = thematically related concepts that crossed a basin boundary. |
| Parallel face expansion | `fractal_cube_address_space.md` | `expand_cube()` — 6 deterministic child seeds per node, all independent, all GPU-parallelisable. Also enables full-cube storage (6 attractors per memory = 6x richer recall signal). |
| Wheeler as Theorist | `theories.md` § 1 | Wire `synthesize_from_gap()` into the recall path for unknown concepts. Instead of returning low similarity, return a synthesized prediction. Changes query behaviour fundamentally. |
| Semantic Gyroscope | `theories_2.md` § 4 | Thin wrapper around existing `metrics.py`. Adds sequence tracking + drift rate. Real-time stability instrument for text streams. Enables SCM applications. |
| SCM as LLM Output Filter | `theories.md` § 6 | Wire `hallucination_score()` into decoder/agent as a post-generation filter. Metrics exist, just not connected. |

---

## Minor / Medium

Moderate impact, low-to-moderate effort. Refinements and formalisations.

| Item | File | Notes |
|------|------|-------|
| Apple Test formal harness | `theories.md` § 4 | `synthesis.py` + `apple_test_semantic.py` already exist. Convert to `tests/theories/test_apple.py` with parameterized domains + regression coverage. |
| Fork points as ambiguity markers | `theories_2.md` § 2 | Oscillation detection already runs. Extract `fork_frame` at peak oscillation tick, store alongside attractor, add to recall scoring. |
| `attractor_portal_hash` + `expand_cube` | `fractal_cube_address_space.md` | Two small functions in `hashing.py`. Low effort, foundational for fractal cube traversal. |
| Time axis metadata | `fractal_cube_address_space.md` | Compute per-tick entropy curve from brick history on store. Downsample to 10-point summary. Store in index. Costs one brick read per store. |
| Bridge Sentences tooling | `theories.md` § 5 | `topology_map.py` already detects disconnected pairs. Add tooling to measure before/after Jaccard for a bridge sentence. Minimum viable topology patch validation. |
| Lichtenberg as retrieval mechanism | `theories.md` § 2 | `lichtenberg.py` is currently viz-only. Implement the model as actual retrieval: query = ground leader, recall = circuit completion. Channel deepening = hit_count (already tracked). |
| Query-Driven Crystallization | `theories.md` § 3 | Cost scales with query complexity, not corpus size. Architectural — chunk routing is currently keyword-based, needs topology-based relevance filtering. Defer until chunking is stable. |

---

## Trivial / Low

Low priority. Speculative, long-term, or purely exploratory.

| Item | File | Notes |
|------|------|-------|
| Fractal cube frontend | `fractal_cube_address_space.md` | The math is the point; the visual is optional. Build only after backend traversal is working and validated. |
| SCM Telemetry / IPv6-like addressing | `theories_2.md` § 5 | Most architectural and speculative. Meaningful only after gyroscope + filter are proven. Long-term vision item. |
| Cross-references between plan files | — | Add `See also:` pointers in plan docs (ternary fix → fractal cube, daydream → query writes, fork points → time axis). Housekeeping only. |

---

## Dependency Order

```
is_flat fix
  └── daydream mechanism
        └── every query is a write
              └── query-driven crystallization

is_flat fix
  └── ternary semantics (0 = uncertainty)
        └── hallucination discrimination
              └── SCM filter
                    └── SCM telemetry

attractor_portal_hash + expand_cube
  └── parallel face expansion
        └── full-cube storage
              └── fractal cube frontend

reconstruction scoring (--mode recall-text)   [independent, current focus]

trajectory similarity                          [independent, data exists]
fork points                                    [independent, oscillation exists]
apple test harness                             [independent, code exists]
semantic gyroscope                             [independent, metrics exist]
bridge sentences tooling                       [independent, topology_map exists]
```

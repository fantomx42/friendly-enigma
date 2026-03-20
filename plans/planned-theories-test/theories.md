# Planned Theory Tests

> Discussed 2026-03-17. Analysis of 6 theoretical models — what exists, what's missing, what to build.

---

## 1. Wheeler as Theorist

**The theory**: Wheeler doesn't just retrieve — it predicts what attractors *should* exist before
they're observed. Novel input arrives → Wheeler says "this should sit between X and Y with these
properties" rather than "I have nothing on this." Falsifiable: the prediction either stabilizes or
collapses under CA evolution.

**What exists**:
- `theories/synthesis.py` — `synthesize_from_gap()` generates candidate attractors from basin gaps
- `theories/basin.py` — `find_basin_gaps()` locates regions of topology with no stable attractor
- `scripts/apple_test_semantic.py` — tests whether excluded concepts can be predicted from neighbours

**What's missing**:
- The prediction is never surfaced at query time. A query for an unknown concept currently returns
  low similarity scores and stops. It should instead trigger `synthesize_from_gap()` automatically
  and return the synthesized candidate with a confidence flag.
- No prediction→validation pipeline: synthesize candidate → evolve → measure stability → report
  whether the prediction held.
- No storage of predictions as provisional attractors (flagged `"memory_type": "predicted"`).

**Test to build**: `tests/theories/test_theorist.py`
- Store a semantic domain (e.g. physics concepts), exclude one concept
- Query for the excluded concept
- Assert that `synthesize_from_gap()` returns a candidate with correlation > 0.15 to nearest neighbours
- Assert the candidate stabilizes under CA evolution (state = CONVERGED)
- Assert it collapses (correlation drops) when queried on an out-of-domain concept

---

## 2. Lichtenberg Model

**The theory**: The corpus is the charged sky (potential, not yet real). The query converts to a
frame = the ground leader reaching up. CA dynamics = both sides propagating toward each other. Frame
stabilization = circuit completion. The answer was always in the corpus; the query made it real.
Channel deepening = hit count. Novel synthesis = predicting where a channel must exist before
lightning has run through it.

**What exists**:
- `theories/lichtenberg.py` — visualization: PCA projection of topology, basin widths as node size,
  hit counts as brightness, query seed as ground point, propagation paths as branching lines
- `hit_count` in index metadata — channel deepening is already tracked
- `synthesis.py` — predicting where channels must exist (unused at query time)

**What's missing**:
- `lichtenberg.py` is a visualization tool only. The model isn't implemented as a retrieval mechanism.
- The "ground leader + sky leader meeting" framing isn't in recall logic — query and corpus evolve
  independently, not toward each other.
- No bidirectional propagation: currently only query → attractor, not corpus → query simultaneously.
- Novel synthesis (predicting missing channels) is not triggered at query time.

**Test to build**: `tests/theories/test_lichtenberg.py`
- Crystallize a corpus, run a query that should complete a circuit
- Verify that `hit_count` increments (channel deepening)
- Verify that a novel query near a basin gap triggers synthesis (channel prediction)
- Optionally: render the Lichtenberg figure and assert the query ground point visually connects
  to the nearest corpus terminal node

---

## 3. Query-Driven Corpus Crystallization

**The theory**: Cost scales with query complexity, not corpus size. Only the topology directly
relevant to the query activates. Irrelevant domains contribute nothing. The corpus doesn't fully
activate per query — only the relevant sub-topology does.

**What exists**:
- Chunked storage (`chunking.py`) routes memories to domain chunks — partial relevance filtering
- Per-chunk recall already limits search to one domain at a time (if chunk is specified)
- Temperature system deprioritizes cold/irrelevant memories

**What's missing**:
- Crystallization is currently uniform: `wheeler-crystallize` processes every corpus entry at the
  same cost regardless of query relevance. There's no query-driven selective crystallization.
- No mechanism to crystallize *only* the topology relevant to a given query or query class.
- The chunk routing is keyword-based, not topology-based — it's a rough filter, not query-driven.

**Test to build**: `tests/theories/test_query_crystallization.py`
- Crystallize two distinct domains (e.g. physics + cooking)
- Run a physics query, measure how many cooking attractors were touched during recall
- Assert: cooking domain activates 0 (or near-0) attractors for a physics query
- Measure compute cost (ticks, Pearson ops) as a function of corpus size vs query domain size
- Assert cost scales with relevant topology size, not total corpus size

---

## 4. Option C Frame Synthesis (Apple Test)

**The theory**: Wheeler predicts novel attractors before direct exposure by interpolating from basin
topology. Train on a semantic domain, deliberately exclude one concept, then see if Wheeler can
synthesize a stable frame for it from topology alone.

**What exists**:
- `theories/synthesis.py` — `synthesize_from_gap()` — already implements this
- `theories/basin.py` — `find_basin_gaps()` — locates where the synthesis should happen
- `scripts/apple_test_semantic.py` — test harness exists and has been run
- **Real results already in README**: ML Architecture TOPOLOGY (0.251), Physics TOPOLOGY (0.328),
  Biology weak topology (0.149)

**What's missing**:
- The apple test is a standalone script, not an automated test suite
- No regression: if dynamics change (e.g. the ternary fix), does synthesis quality hold?
- No systematic coverage across domains — only 3 domains tested manually
- Synthesis confidence not calibrated: what Pearson threshold separates real topology from noise?

**Test to build**: `tests/theories/test_apple.py`
- Parameterized across domains (physics, ML, biology, chemistry, etc.)
- Exclude one concept per domain, crystallize neighbours, synthesize, assert correlation > threshold
- Assert synthesized frame CONVERGES (not chaotic)
- Assert synthesis quality degrades gracefully for out-of-domain exclusions
- Run as regression: if ternary fix changes convergence, synthesis scores must not drop

---

## 5. Bridge Sentences as Atomic Repair Unit

**The theory**: When two concepts show 0.000 Jaccard overlap (zero co-activation), the fix isn't
more data — it's a single well-formed bridge sentence that connects the two basins. Bridge sentences
are the minimum viable topology patch.

**What exists**:
- `scripts/topology_map.py` — co-activation adjacency map, identifies disconnected concept pairs
- Real result from topology work: self-attention / transformer showed 0.000 overlap, then 0.111
  after a bridge sentence — in the README empirical results table
- `warming.py` — association tracking already records co-activation edges

**What's missing**:
- No automatic detection of topology gaps that need bridge sentences
- No tooling to suggest or generate bridge sentences for a given disconnected pair
- No test that validates a bridge sentence actually repairs the gap (before/after Jaccard)
- Bridge sentence quality not defined: what makes a bridge sentence effective vs ineffective?

**Test to build**: `tests/theories/test_bridge.py`
- Store two concepts with 0.000 co-activation (verified by topology_map)
- Store one bridge sentence connecting them
- Re-run topology_map, assert co-activation is now > 0.0
- Assert the bridge's effect is local: unrelated concept pairs are unaffected
- Measure minimum bridge sentence count needed to achieve stable co-activation

---

## 6. SCM as LLM Output Filter

**The theory**: LLM generates candidate outputs → Wheeler scores each for symbolic stability →
only outputs that survive Wheeler's stability pressure are returned. Meaning is what survives
symbolic pressure, applied literally as a pipeline stage. Wheeler's attractor stability metrics
become a filter, not just a memory store.

**What exists**:
- `decoder.py` — Wheeler-primary mode: Wheeler state conditions LLM input (upstream filter)
- `agent.py` — Wheeler provides context to LLM (advisory, not filtering)
- Pearson similarity and convergence_ticks already available as stability metrics
- `theories/metrics.py` — `hallucination_score()` for detecting attractor drift without basin pull

**What's missing**:
- No downstream filter: LLM output is never scored by Wheeler after generation
- No candidate scoring pipeline: generate N outputs → score each → return highest-stability one
- `hallucination_score()` exists but isn't wired into any agent loop
- No definition of what "symbolic stability" means for a text string (text → attractor → stability
  score → threshold)

**Test to build**: `tests/theories/test_scm_filter.py`
- Given two candidate LLM outputs (one factually grounded, one hallucinated)
- Score both via Wheeler: text → attractor → convergence_ticks + Pearson top similarity
- Assert the grounded output scores higher stability
- Assert the hallucinated output scores lower (slower convergence, lower similarity, more oscillation)
- Integration test: wire into `decoder.py` as a post-generation filter, verify output improves

---

## Implementation Priority

| Theory | Existing Code | Test Difficulty | Expected Impact |
|--------|--------------|-----------------|-----------------|
| Option C / Apple Test | ✓✓ nearly complete | Low — formalize existing script | Medium — validates synthesis |
| Bridge Sentences | ✓ topology map exists | Low-Medium | High — direct corpus fix tooling |
| Wheeler as Theorist | ✓ synthesis + basin | Medium — wire into recall | High — changes query behaviour |
| Lichtenberg | ✓ viz only | Medium — implement as retrieval | Medium — conceptual validation |
| SCM Filter | ✓ metrics exist | Medium — wire into agent | High — measurable output quality |
| Query Crystallization | ✗ chunk routing only | High — architectural change | High — efficiency + relevance |

**Start with Option C and Bridge Sentences** — both have working code, just need formal test harnesses.

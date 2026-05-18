# Wheeler Memory — Project Canon

Cellular automaton-based associative memory engine.
Also called Project Darman / Project Ralph.
Repo: github.com/fantomx42/wheeler-memory
Owner: Tristan. Solo project. Claude is translation layer only.

**Tagline:** *Darman doesn't retrieve. Darman reconstructs.*

---

## Status tag legend

- `[BUILT]` — exists in code, working
- `[PARTIAL]` — implementation started, not complete
- `[DESIGNED]` — specified, not yet implemented
- `[PROVEN]` — formal result, code may or may not follow
- `[OPEN]` — known unsolved problem with current best thinking
- `[SPECULATIVE]` — direction, not yet specified
- `[REJECTED]` — tried, discarded, with mechanical reason

---

# 1. Foundation

## 1.1 Core axiom

> Meaning is what survives symbolic pressure.

Memory is not a lookup over a stored set. Memory is what the system
reconstructs when perturbed and allowed to settle. Stable attractors
under perturbation are what "remembering" means. Unstable patterns
collapse and are forgotten. This is the load-bearing claim of the
entire architecture.

## 1.2 The encoder layer is plural and contested `[ACTIVE RESEARCH]`

Wheeler runs multiple encoder backends and treats their relative
SimLex performance as the live signal of architectural progress.
The architectural claim is that meaning can be reconstructed
natively if the encoder is good enough; MiniLM is the bar to clear,
not the canonical answer.

**Primary surface:** `hash` (deterministic, default for `wheeler-recall`
and reproducible benchmarks), `hippocampus` (Wheeler-native n-gram,
default for `wheeler-simlex` and the active production target),
`embedding` (MiniLM via sentence-transformers, the external baseline
to clear), `blended` (convex combination, default for user-facing
surfaces).

**Research variants** (live in `scripts/wheeler_simlex.py:60–71`
`ALL_ENCODERS`): `word`, `hippo-word`, `context`, `context-blended`,
`word-blended`, `language`. These compete in sweeps; survivors get
promoted.

**Wheeler still handles memory** — reconstruction via attractor
dynamics on the three grids. The CA is not a similarity function.
That part of the division is correct and non-negotiable: any proposal
to "add nearest-neighbor fallback to recall" is a category error and
gets refused.

## 1.3 KISS as design discipline

Keep it stoopid simple. The "stoopid" is intentional — a self-aware
circuit breaker against over-engineering. Whenever a component starts
growing parameters, knobs, or branches, the question is whether the
new complexity is *load-bearing* or *defensive*. Defensive complexity
is removed. Load-bearing complexity stays and is documented.

KISS has saved this project from:
- Multi-agent systems (Ollama phi3 / deepseek / mistral) — consolidated
  to single model + Wheeler.
- Hippocampus encoder development — narrowed to one Wheeler-native
  path (rho 0.10 → 0.26 across v0.3.0–v0.3.5) instead of branching
  into parallel encoder strategies.
- Notebook-driven exploration committed to repo — kept as scratch only.

## 1.4 What this is not

- Not GPU-bound. CPU-targeted CA. No CUDA, no ROCm, no Vulkan paths
  inside the engine. Inference stack (llama.cpp-vulkan + aider) is
  outside this repo.
- Not retrieval. No nearest-neighbor fallback "for robustness."
- Not multi-agent. One model, one CA, one process.
- Not a transformer. Not a RAG layer. Not an embedding store.

---

# 2. State space — balanced ternary

State alphabet: `{-1, 0, +1}`. `[BUILT]`

- `+1` — assertion / activation
- `-1` — negation / inhibition
- `0` — quiescent

Zero is special. Zero is the **reconstruction root** — the state from
which a pattern can settle into either polarity under interference.
Quiescence is not absence; it is potential.

Independently derived; later mapped to Setun (1958, USSR balanced
ternary computer) and BitNet b1.58 (2024). Mapping is confirmation,
not influence.

---

# 3. The three grids

The architecture is a tensor product of three same-shaped grids,
each with different temporal dynamics:

| Grid | Temperature | Update rate | Role |
|------|-------------|-------------|------|
| Corpus | Cold | Slow / batch | Stable knowledge layer |
| Experiential | Hot | Fast / per-event | Recent activation, working memory |
| SCM | Glacial | Hardens over time | Symbolic compression, waveguide |

## 3.1 Corpus grid `[BUILT]`

The cold layer. Encodes durable structure derived from the training
corpus. Updates are batch-scale, expensive, and infrequent. Think of
this as long-term memory consolidated through repetition.

## 3.2 Experiential grid `[BUILT]`

The hot layer. Updates per recall event. Decays. Encodes *what just
happened* and *what is currently being thought about*. Working memory,
in human terms.

## 3.3 SCM — Structural Coherence Map `[BUILT]`

The SCM grid is a 64×64 float32 trust topology storing **where**
interference between Corpus and Experiential is permitted — not
content, but permission. Cells live in `[-1, +1]`; magnitude near
zero means "open / untested" and full magnitude means "closed /
hardened." Implementation: `wheeler_memory/scm_grid.py`.

Hardening is per-cell update-count weighted: well-tested regions
resist further change proportional to their update count. The SCM
does **not** evolve autonomously — there are no CA dynamics on the
grid; it is feedback-driven only.

### 3.3.1 Two feedback pathways

1. **Self-consistency check (coincidence erosion)** — opens or
   closes cells based on output fidelity. Implemented in
   `scm_grid.py:112–149` `update()`.
2. **Recall-driven feedback (κ)** — adjusts gate magnitude based on
   recall quality. Implemented in `scm_grid.py:151–245`
   `update_from_recall()`.

### 3.3.2 Note on terminology collision

The codebase has a separate `cortex_scm.py` that defines a
Structural Coherence **Measure** — a unified score classifying
outputs as SYNTHESIS / NOVEL / HALLUCINATION. Different object,
same acronym. See §3.5 for the disambiguation; this section is the
**Map**, that one is the **Measure**.

### 3.3.5 SCM feedback loop `[BUILT]`

(Was: "sleeping giant problem" — closed in v0.3.4.)

Earlier canon framed SCM topology as not participating in learning
because no feedback loop ran from recall outcomes back into the
grid. That framing was wrong: the recall-driven `κ` feedback path
**is** the closed loop. It shipped in:

- `72d05d5f` Recall-driven SCM feedback: outcome quality (κ) adjusts gate topology
- `b05d6a8e` SCM feedback loop: cold-start bootstrap, docstring fix, test coverage
- `fdd47bb9` v0.3.4: SCM telemetry, gradient sanity test, closed-loop A/B eval

Gradient direction is verified by `tests/test_scm_gradient_direction.py`.

## 3.5 Cortex — three-tier semantic scoring `[BUILT]`

A scoring layer over retrieved attractors, distinct from the three
grids and structurally separate from §3.3 SCM. It operates
post-recall, but lives here as a §3 sibling to keep architectural
components together.

1. **L1 — Correlation graph.** Pearson correlation adjacency over
   the retrieved attractor set, with BFS clustering to identify
   coherent neighborhoods. Implemented in `wheeler_memory/cortex.py`.
2. **L2 — Settlement CA.** Opinion diffusion on the correlation
   graph until convergence. This *is* a CA — a second one in the
   system — but it runs on the abstract graph topology, not on the
   64×64 grid. Distinct dynamics from the three-grid CA.
3. **L3 — Native classifier.** `wheeler_memory/cortex_classifier.py`
   scores choices without external models. Trained via
   `train_cortex_classifier.py`.

Output is a `SCMResult` (`cortex_scm.py:228`) with classifications
`SYNTHESIS / NOVEL / HALLUCINATION`, ten layer scores, and a net
warrant.

### 3.5.1 Map vs Measure — the SCM acronym collision

Two unrelated objects in the codebase share the SCM acronym:

- **`scm_grid.py`** — Structural Coherence **Map** (§3.3). A 64×64
  trust topology controlling where interference is permitted.
- **`cortex_scm.py`** — Structural Coherence **Measure** (§3.5). A
  scoring function classifying recall outputs.

Canon distinguishes them by full name. Code does not. When reading
`SCM` in commit messages or comments, check which file is in scope.

### 3.5.2 L3 classifier 7-vector vs SCM v2.0 namespace `[CLARIFICATION]`

Audited 2026-05-18. The L3 classifier (`cortex_classifier.py:151`
`classify()`) takes a 7-vector parameter named `scm_layers` with the
docstring label `[T, S, E, I, P, NW, ERF]`. These letters are
first-initial abbreviations of `SCMResult` fields in `cortex_scm.py:228`,
**not** the canonical SCM v2.0 variables:

| Label | `SCMResult` field        | What it measures |
|-------|--------------------------|------------------|
| T     | `temperature`            | recall confidence (function of similarity statistics) |
| S     | `salience`               | convergence sharpness |
| E     | `energy`                 | `1 − max(\|settled − prev\|)` (settlement stability) |
| I     | `integration`            | graph-level integration |
| P     | `polarity`               | ternary balance over adjacency |
| NW    | `net_warrant`            | product `T·S·E·I·P` |
| ERF   | `explanation_readiness`  | diagnostic readiness |

The letters happen to collide with canonical SCM v2.0 (Temporal
Stability / Semantic Stability / Environmental Stability / Integration
Coherence / Pressure Sensitivity / Nostalgia Weight / Emotional
Resonance Field) but the underlying *concepts* are entirely different.
`temperature` here is post-recall confidence, not the per-basin drift
controller from §3.6.

Verdict: docstring-vs-canon namespace drift, not an implementation gap.
All 7 inputs are computed and supplied by callers (`scripts/wheeler_mmlu.py:913`,
`scripts/train_cortex_classifier.py:205`). The conventional fix is to
update the classifier and trainer docstrings to use the full field names
rather than letter-pair labels that collide with canon. Tracked as a
separate small commit; no architectural change required.

## 3.6 Substrate vs above-substrate stability variables `[CLARIFICATION]`

The canonical SCM v2.0 namespace (T, E, S, I, P, NW, ERF, FC) was
specified before all of its variables had load-bearing implementations.
This section records the audit (2026-05-18) of which variables actually
live at the substrate level — i.e., are consulted by the per-tick
`apply_ca_dynamics` rule — versus which live above it.

**Only T (Temporal Stability) crosses into the substrate. The other
seven do not.**

| Variable | Status | Where it lives |
|----------|--------|----------------|
| **T**  (Temporal Stability)      | `[BUILT]` per-basin only | `wheeler_memory/t_metadata.py`, `recall_api.py:308` — `plasticity = (1 − T) · BASIN_DRIFT_BASE_RATE`. Stored in `index.json` metadata as `t_stability`. The local variable name `plasticity` is canon's `drift_rate`; rename deferred to recall-wiring ticket. |
| **E**  (Environmental Stability) | not implemented as substrate variable | letter collides with `SCMResult.energy` (§3.5.2), different concept |
| **S**  (Semantic Stability)      | not implemented as substrate variable | letter collides with `SCMResult.salience` (§3.5.2), different concept |
| **I**  (Integration Coherence)   | not implemented as substrate variable | letter collides with `SCMResult.integration` (§3.5.2), partial concept overlap |
| **P**  (Pressure Sensitivity)    | not implemented as substrate variable | letter collides with `SCMResult.polarity` (§3.5.2), different concept |
| **NW** (Nostalgia Weight)        | not implemented                       | additive in canon; no code surface |
| **ERF**(Emotional Resonance Field)| not implemented                      | additive in canon; no code surface |
| **FC** (Fusion Coefficient)      | not implemented                       | core-var blending in canon; no code surface |

### 3.6.1 Empirical confirmation via the read-only diagnostic

`wheeler_memory/diagnostics.py:decompose_tick` (added v0.3.6) is a
pure-analytical decomposition of one `apply_ca_dynamics` step into
5W1H components. It returns observable values for **What** (cell
state), **Where** (neighbor stack), **Who** (neighbor identity via
argmax), and **How** (delta + clipped next-frame), and explicit
`None` for **When** and **Why**, with module-level documentation
explaining the gap:

- **When** (drift contribution from T) is `None` because the per-tick
  rule does not consult T. T acts at the basin level via
  `recall_api.py:279–308`, not per-tick.
- **Why** (SCM gradient pressure) is `None` because the per-tick rule
  does not consult SCM. SCM gates the answer equation at recall time
  in `compute_interference`, not per-tick. The SCM gradient itself
  (§3.3.5) is feedback-driven post-recall, not per-tick.

If T or SCM operated at the substrate level, `decompose_tick` would
have to read them per-cell. Its structural inability to do so — without
synthesizing data the rule never reads — is the load-bearing evidence
that the canonical SCM v2.0 variables E/S/I/P/NW/ERF/FC are
above-substrate concepts, not substrate ones.

### 3.6.2 Implication for future planning

The next ultraplan should treat the canonical SCM v2.0 variables as
**above-substrate concepts** unless explicitly redesigning the per-tick
rule. Substrate changes touch `apply_ca_dynamics` and require benchmark
re-baselining; above-substrate changes (basin-level T updates,
post-recall scoring, cortex-layer scoring) do not.

A proposal to "wire T into apply_ca_dynamics" should be treated as a
substrate redesign with full benchmark implications, not a small
plumbing change. Same for any proposal to make the per-tick rule
SCM-aware.

---

# 4. Recall — the interference formula `[BUILT]`

Answer at grid position (i, j):

```
Answer(i, j) = Corpus(i, j) × Experiential(i, j) × (1 - |SCM(i, j)|)
```

Mechanically:

- Corpus contributes durable signal.
- Experiential modulates by recency / current focus.
- SCM acts as a **gate**, not a contributor — `(1 - |SCM|)` means
  high-magnitude SCM regions are *suppressed*, low-magnitude regions
  let interference through. SCM shapes which channels are open.

This is the waveguide interpretation: SCM does not generate signal,
it routes it. Hardened SCM regions become opaque; quiescent ones
become transparent.

## 4.1 Scoring — `interference_score` `[BUILT]`

The κ signal fed back into `SCMGrid.update_from_recall` (§3.3.5) is
computed by `wheeler_memory/interference.py:interference_score()`. Per
the v0.3.6 fix (commit `39fb8fce`):

```
score = ρ_w(C_q, C_s | w) + ρ_w(X_q, X_s | w)
    where w_ij = 1 − |SCM(i, j)|
    and   ρ_w  = Σw·δx·δy / √(Σw·δx² · Σw·δy²)
```

`ρ_w` is the weighted Pearson correlation between query and stored
attractors on the corpus (C) and experiential (X) channels, with the
SCM permission topology entering as per-cell weights.

The pre-fix implementation factored as `(c_pearson + e_pearson) ×
mean_openness` — collapsing the 64×64 SCM to a scalar. Equal-mean-openness
SCM configurations produced identical scores regardless of spatial pattern,
so the gradient update at §3.3.5 received a spatially blind `advantage =
κ − κ_base` and could not differentiate frozen vs. learning SCM arms.

The fix preserves what the SCM v2.0 design calls the "Who" axis: which
cells get to vote on the score. Hardened cells (`|M| → 1`) contribute
weight ≈ 0 and effectively recuse themselves from the correlation;
quiescent cells (`|M| → 0`) contribute weight ≈ 1 and shape the score
fully. Score range is `[−2, 2]` (sum of two ρ); historical `results.tsv`
rows pre-`39fb8fce` are not directly comparable.

Sacred files were untouched by this fix: `storage.py`, `hashing.py`,
`chunking.py`, `rotation.py`, `dynamics.py`, `scm_grid.py`,
`constants.py`, `scripts/bench_quality.py`.

---

# 5. Encoder layer — see §1.2

Folded into §1.2 in the 2026-05-04 audit pass. The encoder layer
is plural; see §1.2 for the current surface and research variants.
Section number kept stable to preserve cross-references.

---

# 6. Address layer — FCAS `[DESIGNED]`

Fractal Cube Address Space. Primitives present in the codebase, the
address layer itself not yet built.

## 6.1 Key structure

Addresses are tuples: `(hash, depth)`.

## 6.2 The triple-role of SHA256

The SHA256 of a terminal attractor serves *simultaneously* as:

1. **Coordinate** — the address at which this attractor lives.
2. **Reconstruction seed** — the initialization for re-instantiating
   the attractor under perturbation.
3. **Origin of a new cube³:0** — the (0,0,0) of a fresh sub-grid
   nested at this address.

This collapse-of-roles is the load-bearing trick. It is what makes
the address space fractal: every attractor is also a coordinate,
which is also a new origin.

## 6.3 Build status

- Hash primitives: `[BUILT]`
- Attractor identification: `[PARTIAL]`
- Address resolution: `[DESIGNED]`
- Fractal nesting: `[DESIGNED]`
- Cross-cube interference: `[SPECULATIVE]`

---

# 7. Stability theory — Lyapunov `[SPECULATIVE]`

A Lyapunov framing is intended for the three-grid system, with a
cross-coupling term `V_cross` capturing interactions between Corpus,
Experiential, and SCM:

```
V_total = V_corpus + V_exp + V_scm + V_cross
```

No proof or implementation is committed yet. Grep finds no
`V_total`, `V_cross`, or Lyapunov-decrease test in code. Open
question: can `V_cross` be defined to make `V_total` monotone
non-increasing under recall? If so, its gradient is a natural
candidate for further SCM tuning, but the SCM feedback loop
(§3.3.5) already closes via the recall-quality `κ` signal — so a
Lyapunov gradient would be an alternative or refinement, not a
prerequisite.

---

# 8. Evaluation surface

## 8.1 SimLex-999 `[ACTIVELY TRACKED]`

Live numbers come from `wheeler-simlex --sweep`. As of v0.3.3:

- hippocampus / context-RI: rho ≈ 0.22–0.26 and climbing.
- MiniLM: rho ≈ 0.43 (verify against latest sweep run).

The eval drives encoder development — see §1.2. Don't lock these
numbers in canon; refresh from sweep output.

## 8.2 MMLU `[CHANCE FLOOR]`

Currently sits near 25% (chance for 4-option multiple choice).
**Diagnosis: corpus-limited, not architecture-limited.** Do not
treat MMLU as an architecture defect signal. The Corpus grid has
not been populated with sufficient world-knowledge structure. MMLU
will move when corpus does.

Treat MMLU as a *corpus health* metric, not a *recall quality* metric.

## 8.3 Wheeler-native eval `[SPECULATIVE]`

The right eval for an attractor-reconstruction memory is not a
multiple-choice benchmark. It is something like: perturb a known
attractor, measure settling time and final-state fidelity.
Not yet specified or implemented.

---

# 9. Open work items

In priority order:

1. **FCAS address resolution** — wire the (hash, depth) tuple keys
   into the recall path. (§6.3)
2. **Wheeler-native eval design** — reconstruction-fidelity benchmark
   to replace reliance on MMLU as architecture signal. (§8.3)
3. **Corpus population strategy** — what gets ingested, how it gets
   ternarized, how to budget across the grid. Affects MMLU directly.
4. **Cross-cube interference semantics** — what does it mean for a
   nested cube³:0 to interfere with its parent? Speculative until
   FCAS resolution is done.

---

# 10. Applications and extensions `[SPECULATIVE]`

Direction, not specification. Listed for completeness.

## 10.1 Symbolic Integrity Engineering

A discipline using SCM dynamics as a measurement instrument:

- Monitoring symbolic stress in a corpus (news, discourse, code)
- Detecting collapse events (sudden meaning shifts)
- Modeling cultural pressure dynamics

## 10.2 SCM as LLM output filter

Pipeline:

```
LLM Output → SCM Scoring Engine → Symbolic Stability Filter → Final Response
```

Use the SCM to score generated text on semantic stability and reject
outputs that score as collapsing. A meaning-survival filter.

## 10.3 SCM Browser

A web browser with real-time stability scoring of page content.
Optional high-collapse-content blocking. Reader-side defense against
symbolic instability in the information environment.

## 10.4 SCM Telemetry

IPv6-like address encoding of SCM stability metrics. Distributed
information-ecosystem monitoring at the network layer rather than
the application layer.

---

# 11. CFP — Cellular Fabric Processor `[SPECULATIVE]`

Architecture theory. Deterministic spatial mesh of SPEs (Stream
Processing Elements) and PPEs (Pattern Processing Elements) with
an opcode class system. Independently derived; mapped post-hoc to
systolic arrays and dataflow machines. Confirmation, not influence.

The CFP is what Wheeler would run on if Wheeler had ideal silicon.
Currently academic.

---

# 12. Theoretical homologies — independently derived

Pattern: internal simulation first, external mapping afterward.
These are confirmations, **not influences**. Order matters because
inverting it inverts the credit and the meaning.

## 12.1 Causal set theory

Rideout–Sorkin sequential growth dynamics. Maps onto the
discrete-frame irreversibility model and the address-layer growth
of FCAS.

## 12.2 Wheeler pregeometry / "law without law"

John Wheeler's program of deriving physical law from substrate-free
combinatorial structure. The three-grid CA + SCM topology is in
the same family of moves: structure as primitive, law as emergent.

The project name nods to this lineage.

## 12.3 Cytowic synesthesia research

Three-grid architecture maps to Cytowic's structural account of
synesthesia as cross-modal binding through a shared topological
substrate. The mapping was discovered after the architecture was
specified. Strong structural homology.

## 12.4 Setun / BitNet b1.58

Balanced ternary state space. Setun (1958) and BitNet (2024) both
arrive at the same alphabet for different reasons. Wheeler arrives
at it for a third reason: 0 as reconstruction root.

## 12.5 Systolic arrays / dataflow machines

CFP architecture (§11.4) maps onto these post-hoc.

---

# 13. Cosmological framing `[SPECULATIVE]`

Wheeler Memory has a cosmological angle separate from the engineering
project. Discrete-frame irreversibility model: time as an indexing
relationship between total states rather than a manifold coordinate.

Connections:
- Causal set theory (Rideout–Sorkin)
- Wolfram hypergraph rewrites
- Wheeler pregeometry (the namesake)

This thread is academic and does not gate any code work. Listed
here so the engineering project doesn't lose track of where the
ideas came from.

---

# 14. Project meta

## 14.1 Naming history

- **Project Ralph** — original name. Began as a QR-code-to-Coral-TPU
  hardware idea.
- **Project Darman** — middle period name, used in some commits.
- **Wheeler Memory** — current canonical name. Honors Wheeler
  pregeometry.

All three names refer to the same project.

## 14.2 Origin sequence

1. QR-code-to-Coral-TPU hardware concept (Ralph)
2. Pivot to software memory engine
3. Multi-agent Ollama experimentation (phi3, deepseek, mistral)
4. KISS consolidation to single model + Wheeler
5. Three-grid architecture breakthrough
6. SCM v2.0 variable specification
7. Lyapunov stability proof + V_cross extension
8. SimLex/MMLU evaluation phase, encoder reinstatement
9. Theory experiments branch + PR #18 cherry-pick
10. Current: SCM sleeping giant work, FCAS specification

## 14.3 Working principles

- **Whole before parts.** No component without containing structure.
- **Mechanical why, not preference.** "A yields X because of Y."
- **Corrections update assumptions immediately.** No defensive
  re-explaining.
- **"GitHub is updated"** = read and review what was committed,
  not implement new changes.
- **Independent derivation, then mapping.** Don't import external
  frameworks before the internal one is specified.
- **Token economy in long sessions.** Verbose tool output bleeds
  context across limit-bounded continuations. Bounded reads only.

## 14.4 Tooling stack (as of current)

- Inner loop: pytest + ruff + mypy
- Encoder: hippocampus default (Wheeler-native n-gram); MiniLM
  available as `--encoder embedding` for baseline comparison
- OS: CachyOS, i7-265K 
- VCS: git, theory experiments branch → cherry-pick to main
- Editor: agnostic; Claude Code as translation layer



---

Audit history: `docs/audit-*.md`

*End of canon. Edit in place. Status tags update with reality.*

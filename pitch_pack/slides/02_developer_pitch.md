---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #c084fc;
  }
  h2 {
    color: #67e8f9;
  }
  code {
    background: #1e1b4b;
    padding: 2px 6px;
    border-radius: 3px;
  }
  table {
    font-size: 0.85em;
  }
---

# Wheeler Memory
## Open-Source Cellular Automaton Memory for LLMs

**Build agentic systems that remember, forget, and reconstruct like humans do.**

---

## Why This Matters

### The LLM Memory Problem

- **Vector DBs ≠ Memory**: Similarity search is not episodic recall. Same query = same result always.
- **Reconstruction is novel**: Human memory doesn't retrieve verbatim. Context reshapes recall.
- **Forgetting is useful**: Models without decay never reach epistemic humility. They hallucinate confidently.
- **Associative recall is missing**: Semantic search finds similar documents. CA finds related *concepts* through learned dynamics.

**Status quo**: We're treating databases like memory. Time for something different.

---

## What You Can Build

### Use Cases, Research & Production

| Use Case | What's Possible |
|---|---|
| **Agentic systems** | Agents with persistent state, forgetting curves, contextual decision-making |
| **Chatbots** | Personalized conversation that remembers and forgets naturally |
| **Semantic search** | Query expansion via associative warmth (2-hop spreading activation) |
| **Memory evaluation** | Benchmark reconstruction fidelity, oscillation detection, temperature dynamics |
| **Research** | Test neuroscience theories via symbolic collapse (IIT connections) |
| **Privacy-first LLMs** | Local-only memory, no cloud, no data escape |

**No proprietary backend. Pure Python. Run anywhere.**

---

## Quick Start: 3 Commands

### Install and Run

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
pip install -e .
```

### Launch the Dashboard

```bash
wheeler-ui
# Opens http://localhost:7437
```

### Store and Recall a Memory

```bash
wheeler-store "remind me to debug the API timeout"
wheeler-recall "api performance issues"  # finds it via semantic similarity
```

**Done. No GPU required. Runs on CPU. Python 3.11+ only.**

---

## Two Recall Modes

### Exact vs. Semantic

#### Exact Recall (Default)
```bash
wheeler-store "fix the python syntax error"
wheeler-recall "fix the python syntax error"  # exact match only
```
- Uses SHA-256 hashing
- Deterministic: same text always produces same fingerprint
- Fast, zero dependencies
- One word changes = completely different pattern

#### Semantic Recall (Install Extra)
```bash
pip install -e ".[embed]"
wheeler-store --embed "fix the python syntax error"
wheeler-recall --embed "debug python code"  # finds it!
```
- Uses `sentence-transformers` (all-MiniLM-L6-v2)
- Meaning-based: "debugging code" finds "fix the syntax error"
- Similar meanings produce similar CA seeds
- ~80 MB model, runs locally

**Caveat**: Seeds stored with `--embed` can only be recalled with `--embed`.

---

## The Science: Symbolic Collapse Model

### What We're Actually Computing

**Hypothesis**: Meaning is what *survives symbolic pressure*.

```
Input text
    ↓
Dimensionality reduction (hash → 64×64)
    ↓
3-state CA evolution (local rules only)
    ↓
Convergence → Attractor (unique fixed point)
    ↓
Symbolic pressure: Information collapses to essential structure
```

**Key insight**: The attractor is not the text. It's what remains after lossy compression through dynamics.

### Connections to IIT (Integrated Information Theory)
- **Integrated information**: CA convergence measures how much the grid constraints itself
- **Phi**: Oscillation metrics relate to irreducibility
- **Markov blanket**: Von Neumann neighborhoods create local causal closure

**Research opportunity**: Validate IIT predictions against empirical CA behavior.

---

## The 3-State CA Rule

### How Attractors Form

```python
# Simplified pseudocode
for each cell in grid:
    neighbors = von_neumann(cell)
    local_max = max(neighbors) > threshold
    local_min = min(neighbors) < threshold
    slope_uphill = uphill_direction_exists()

    if local_max:
        cell += 1  # push toward +1 (35% probability)
    elif local_min:
        cell -= 1  # push toward -1 (35% probability)
    elif slope_uphill:
        cell += sign(uphill_direction)  # flow (20% probability)
```

- **Von Neumann neighborhood**: Only 4-connected (up/down/left/right)
- **Wrapping boundaries**: Grid is a torus (no edge artifacts)
- **Stochastic rule**: Each operation has fixed probability
- **Convergence**: Typically 40-100 ticks. ~3 ms on CPU.

### Three End States

| State | Meaning | Action |
|---|---|---|
| **CONVERGED** | Fixed point reached | Store attractor, proceed to recall |
| **OSCILLATING** | Periodic cycle detected | Indicates ambiguous input, log for user |
| **CHAOTIC** | Unbounded growth | Input may need rephrasing |

---

## Temperature & Forgetting

### How Memories Decay

#### Formula

```
temp = base_from_hits × decay_from_time

base_from_hits  = min(1.0,  0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (−days_since_last_access / 7)
```

#### Constants
- **Half-life**: 7 days (configurable)
- **Hit saturation**: 10 recalls (configurable)
- **Fresh memory baseline**: 0.3 (warm on arrival)

#### Temperature Tiers

```
Hot:  temp ≥ 0.6  (frequently accessed, recent)
Warm: temp ≥ 0.3  (default for new memories)
Cold: temp < 0.3  (stale, candidate for archival)
```

### Tracking Access
```bash
wheeler-temps  # List all memories with temperature, decay, hit count
```

---

## Reconstructive Recall

### The Context-Dependent Memory Mechanism

#### Formula
```
blend = (1 - α) × stored_attractor + α × query_seed
reconstructed = evolve_and_interpret(blend)
```

**Default**: α = 0.3 (memory-dominant, 70% stored + 30% query)

#### Why This Works
- **Like human memory**: Loftus research shows recall is reconstructive, not retrieval
- **Prevents rigidity**: Same memory reconstructs differently for different queries
- **Preserves stability**: Memory-dominant bias means recall is mostly stable

#### Tuning
- **α = 0.3**: Conservative (recall looks like stored memory)
- **α = 0.5**: Balanced (equal weighting)
- **α = 0.7**: Query-driven (stress testing, hallucination detection)

---

## Validation: 19 Modules, 167+ Tests

### What We Prove

#### Datasets
| Dataset | Size | Domain |
|---|---|---|
| **MBPP** | 10k problems | Python code snippets |
| **SWE-bench** | Software engineering tasks | Real-world repo operations |
| **BABILong** | Multi-hop reasoning | Long-context synthesis |

#### Test Coverage
- **Dynamics**: CA convergence, rotation retry, oscillation detection
- **Storage**: Pearson correlation, chunking, metadata persistence
- **Reconstruction**: Blending fidelity, context-dependent variance
- **Temperature**: Decay curves, tier thresholds, access tracking
- **GPU**: HIP/ROCm vs CPU parity, speedup benchmarks

### Running Tests
```bash
pytest tests/  -v
# 167 tests across all modules
```

---

## Architecture Deep Dive

### 7-Layer Stack

```
1. agent.py          ← Ollama integration, tool dispatch, chat loop
2. chunking.py       ← Domain routing (code, hardware, science, general, …)
3. reconstruction.py ← Blending, α parameter, re-evolution
4. storage.py        ← Pearson correlation, recall ranking
5. temperature.py    ← Decay formulas, tier logic
6. polarity.py       ← Dual encoding, aversion attractors, decay count
7. dynamics.py       ← CA rules, convergence detection, GPU dispatch
```

### Key Modules

| Module | Responsibility |
|---|---|
| `dynamics.py` | 3-state CA evolution, convergence logic |
| `gpu_dynamics.py` | HIP/ROCm CUDA kernels, CPU fallback |
| `storage.py` | Chunked storage, Pearson search |
| `brick.py` | Memory brick (history + metadata) |
| `hashing.py` | SHA-256 or embedding-based hashing |
| `chunking.py` | Keyword routing to 6 domain chunks |
| `temperature.py` | Decay, hit tracking, tier assignment |
| `polarity.py` | Dual attractors, aversion patterns |
| `warming.py` | Spreading activation, associative recall |
| `oscillation.py` | Cycle detection, period extraction |
| `rotation.py` | 0°/90°/180°/270° retry logic |
| `embedding.py` | Sentence-transformers integration |
| `reconstruction.py` | Blend + re-evolve pipeline |
| `consolidation.py` | Sleep consolidation, stale brick pruning |
| `eviction.py` | Memory limit enforcement |
| `attention.py` | Selective recall, focus weighting |
| `hardware.py` | Platform detection (CPU/GPU) |

---

## Extensibility Points

### Pluggable Layers

#### Custom Hashing
```python
from wheeler_memory import store_memory

# Replace SHA-256 with your own encoder
store_memory("my text", hash_fn=my_custom_encoder)
```

#### Custom Chunks
```python
# Add a domain-specific chunk beyond the built-in 6
chunking.add_chunk("robotics", keywords=["servo", "motor", "ros", ...])
```

#### Custom CA Rules
```python
# Extend dynamics.py with alternative evolution rules
class MyCustomDynamics(CADynamics):
    def update_rule(self, cell, neighbors):
        # Your rule here
```

#### GPU Kernels
```python
# Implement custom HIP/ROCm kernel in gpu_dynamics.py
# Auto-fallback to CPU if kernel unavailable
```

---

## Testing & Benchmarks

### What the Test Suite Covers

#### Oscillation Detection
```python
# Tests ensure we catch periodic cycles reliably
brick = evolve(seed)
assert brick.state in ["CONVERGED", "OSCILLATING", "CHAOTIC"]
```

#### Rotation Retry
```python
# Same seed at 0°/90°/180°/270° should converge consistently
for angle in [0, 90, 180, 270]:
    brick = evolve(seed, rotation=angle)
    assert brick.state != "CHAOTIC"
```

#### GPU Benchmarks
```bash
wheeler-bench-gpu
# Outputs: CPU time, GPU time, speedup factor
# Typical: 2-5x speedup on HIP/ROCm, 5-10x on CUDA
```

#### Temperature Decay
```python
# Verify 7-day half-life
old_temp = compute_temperature(hits=5, days_since=0)
halved_temp = compute_temperature(hits=5, days_since=7)
assert abs(halved_temp - old_temp / 2) < 0.01
```

---

## Web UI & CLI Ecosystem

### 10 CLI Tools

| Tool | Purpose |
|---|---|
| `wheeler-store` | Save a memory |
| `wheeler-recall` | Find similar memories |
| `wheeler-ui` | Web dashboard |
| `wheeler-temps` | List all memories + temperatures |
| `wheeler-forget` | Delete a memory |
| `wheeler-sleep` | Consolidate stale memories |
| `wheeler-agent` | Start the Darman chatbot |
| `wheeler-info` | System info + config |
| `wheeler-scrub` | Visualize attractor formation |
| `wheeler-bench-gpu` | Benchmark GPU vs CPU |

### Web Dashboard (Wheeler UI)

- **Live recall**: Search by text or embedding
- **Temperature display**: See memory age/heat at a glance
- **Brick visualization**: Interactive slider through CA evolution
- **Oscillation log**: Detect problematic inputs
- **Chunk breakdown**: See distribution across domains
- **Export**: Dump memories as JSON or CSV

---

## Integration Points

### Ecosystem

| Tool | Integration | Notes |
|---|---|---|
| **Ollama** | Native | `--ollama-host` flag in agent |
| **Open WebUI** | Pipeline function | Drop-in Wheeler memory layer |
| **Hugging Face** | Hub integration | Load/share memory archives |
| **LangChain** | Adapter (contrib) | Use Wheeler as memory backend |
| **LlamaIndex** | Vector store (planned) | Pluggable recall engine |
| **FastAPI** | HTTP API (demo) | Stateless recall endpoints |

### Running the Agent
```bash
wheeler-agent --ollama-host "http://localhost:11434"
# Starts interactive chatbot with auto-recall and auto-store
```

---

## Roadmap (6–12 Months)

### Phase 1: Core Hardening (Now)
- ✓ Test suite expansion (done)
- ✓ GPU parity validation (done)
- 🔄 Documentation polish
- 🔄 Community outreach

### Phase 2: Multimodal (Q3 2026)
- Image embedding (CLIP)
- Audio embeddings (wav2vec)
- Cross-modal reconstruction

### Phase 3: Distributed (Q4 2026)
- Peer-to-peer memory sync
- Federated recall (query multiple nodes)
- Gossip consolidation

### Phase 4: WebAssembly (Q1 2027)
- Client-side CA in browser
- Offline-first web app
- Edge memory inference

---

## How to Contribute

### High-Impact Areas

#### GPU Optimization
- Improve HIP/ROCm kernels for 64×64 grids
- Add METAL (Apple Silicon) support
- Optimize rotation retry parallelism

#### Multimodal Encoders
- CLIP integration for images
- Audio embeddings for speech
- Multimodal reconstruction pipeline

#### Distributed Protocols
- Memory replication scheme
- Federated query aggregation
- Conflict resolution for contradictory memories

#### Benchmarks & Research
- Validate IIT predictions against empirical CA
- Compare reconstruction fidelity vs. vector DBs
- Publish neuroscience connections

#### Documentation
- Tutorials for agentic systems
- Case studies (chatbot, code assistant, etc.)
- Blog posts on symbolic collapse

---

## Philosophy: "The Formula Is the Foundation"

### Backward Compatibility is Sacred

**Rule**: The CA rule, convergence detection, and Pearson recall are load-bearing walls.

- Changes to `dynamics.py` affect *all* existing memories
- Changes to `hashing.py` require regeneration
- Changes to `storage.py`'s similarity function shift recall rankings

**Consequence**: Modifications are treated as breaking changes. Version bumps are semantic. Deprecation is documented.

### Everything Else is Commentary

- UI can be replaced
- Agent prompt can be tweaked
- CLI tools are extensible
- GPU kernels can be optimized

**Design principle**: Upstream changes are rare; downstream extensions are encouraged.

---

## Community & Support

### GitHub: `fantomx42/wheeler-memory`

- **Issues**: Bug reports, feature requests, discussions
- **Discussions**: Q&A, architecture deep-dives, use cases
- **Contributing**: Pull request workflow, code review, CLA
- **Roadmap**: Public issue tracking, milestone planning

### Documentation
- **Architecture**: Layer-by-layer walkthrough
- **API Reference**: Every class, function, parameter
- **CLI Guide**: All 10 tools with examples
- **Design Principles**: 7 axioms + rationale
- **Concepts**: Reconstruction theory, symbolic collapse, IIT

### Whitepaper & Research
- "Symbolic Collapse and Reconstructive Memory" (in prep)
- Connections to Integrated Information Theory
- Benchmark suite against vector DBs and RAG systems

---

## Call to Action

### Start Contributing Today

```bash
# 1. Clone the repo
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory

# 2. Set up dev environment
pip install -e ".[dev]"
pytest tests/ -v

# 3. Check the roadmap
# See CONTRIBUTING.md for high-impact areas

# 4. Open a PR or discussion
```

### Your Project Ideas

- **Chatbot**: Personal memory assistant
- **Code memory**: Remember debugging patterns
- **Research**: Test neuroscience theories
- **Agent**: Long-running autonomous system
- **Search**: Associative document recall

**We provide the engine. You build the application.**

---

## Next Steps

### For Researchers
- Read `docs/concepts.md` (symbolic collapse model)
- Explore `docs/design.md` (7 principles)
- Open an issue with your research questions

### For Practitioners
- Follow the [Installation Guide](docs/install.md)
- Try the [Interactive Demo](ui/demo.html)
- Build something and share on Discussions

### For Contributors
- Pick a [roadmap task](CONTRIBUTING.md)
- Review the [architecture](docs/architecture.md)
- Comment on a draft PR to help shape features

---

# "The formula is the foundation."

**Star the repo. Read the docs. Build with us.**

`github.com/fantomx42/wheeler-memory`

---

# Ralph AI - Claude Code Configuration

## Autonomous Task Delegation

### IMMEDIATE parallel Task launch for:
- File/codebase exploration -> Explore agent
- Any research or docs lookup -> Research agent
- Code review -> Review agent
- Test writing -> separate from implementation

### Rules:
- Launch multiple agents concurrently when possible
- Don't ask permission to delegate - just do it
- Use background tasks for long-running work

---

## Project Architecture

Ralph AI supports **two execution modes**:

### V1: Multi-Agent Swarm (Legacy)
```
Human Input -> [Translator] -> [Orchestrator] -> [Middle Management] -> [ASICs]
                                                    bidirectional
                                                [Engineer] <-> [Designer]
```
- Set `RALPH_V2=1` to enable V2 message-driven mode
- Entry point: `ralph_loop.sh` -> `ralph_core/runner.py`

### V2: Single-Model Architecture (New)
```
Human Input -> [qwen3-coder-next] -> Wheeler Memory -> [qwen3-coder-next] -> Output
                  (all roles)         (stability-         (self-review)
                                       weighted)
```
- Entry point: `ralph_simple.py`
- Uses `qwen3-coder-next` (80B MoE, 3B active, 256K context)
- Wheeler Memory provides stability-weighted context via SCM axioms

### Tiers (V1):
1. **Translator** (`phi3:mini`) - Converts human input to structured TaskSpec
2. **Middle Management** - Engineer & Designer collaborate bidirectionally
3. **LLM ASICs** - Ultra-small specialists spawned for micro-tasks

---

## Project Structure

```
ralph_core/
|-- agents/              # Agent implementations (V1 multi-agent)
|   |-- translator/      # Human -> TaskSpec
|   |-- orchestrator/    # Strategic planning
|   |-- engineer/        # Code generation
|   |-- designer/        # Review & verification
|   |-- reflector/       # Learn from history
|   |-- debugger/        # Error analysis
|   |-- estimator/       # Task valuation
|   +-- sleeper/         # REM sleep memory consolidation
|-- asic/                # Dynamic specialist models
|   |-- spawner.py       # ASIC lifecycle management
|   |-- registry.py      # Available ASIC types
|   +-- handler.py       # Message bus integration
|-- protocols/           # Message bus for agent communication
|   |-- bus.py           # Central MessageBus (FIFO, circuit breaker)
|   +-- messages.py      # Message types (WORK_REQUEST, CODE_OUTPUT, etc.)
|-- security/            # Security checkpoint layer
|   |-- checkpoint.py    # Output validation gateway
|   |-- guards.py        # Type checking, permissions, sanitization
|   |-- dogs.py          # Malware and secret detection
|   |-- towers.py        # Audit logging
|   +-- gate_out.py      # Output rate limiting
|-- runner.py            # Main execution pipeline (V1)
|-- executor.py          # Shell/Docker command execution
|-- memory.py            # Context, vector DB, Wheeler integration
|-- wheeler.py           # Wheeler Memory Bridge (spatial/dynamics)
|-- wheeler_weights.py   # SCM stability metrics (hit count, persistence, compression)
|-- context_budget.py    # Stability-weighted token allocation
|-- npu_engine.py        # OpenVINO NPU acceleration for Wheeler
|-- vram_sentinel.py     # AMD GPU VRAM monitoring (ROCm)
|-- forklift.py          # Intelligent memory loading for agents
|-- consolidator.py      # Memory consolidation and lesson synthesis
|-- controller.py        # High-level control flow
|-- vector_db.py         # ChromaDB/JSON semantic memory
|-- swarm.py             # Agent interface exports
|-- tools.py             # Agent tool system
|-- metrics.py           # Performance tracking
|-- git_manager.py       # Git history and memory
|-- compressor.py        # History compression
|-- planning.py          # Plan management
|-- ollama_client.py     # Ollama API wrapper
|-- dreamer.py           # Background dream task generation
|-- librarian_daemon.py  # File watching for dynamic updates
|-- vision.py            # Vision/image processing
|-- voice.py             # Voice input/output
+-- web.py               # Web utilities

ralph_ui/                # Web UI
|-- backend/main.py      # FastAPI + WebSocket server
+-- frontend/index.html  # Vanilla JS frontend

docs/
+-- SCM_AXIOMS.md        # Symbolic Collapse Model theoretical foundation

conductor/               # Project management tracks
|-- workflow.md           # Task lifecycle and TDD guidelines
|-- product.md            # Product definition and vision
|-- tracks.md             # Active/archived work tracks
+-- code_styleguides/     # Style documentation

tests/                   # Test suite (pytest)
|-- test_vram_sentinel.py
|-- test_vram_ollama.py
|-- test_npu_visibility.py
|-- test_wheeler_npu_kernels.py
|-- test_bridge_integration.py
|-- test_loop_bus.py
|-- test_loop_controller.py
|-- test_wheeler_weights.py
+-- benchmark_npu_memory.py

ralph_simple.py          # Single-model entry point (V2)
ralph_loop.sh            # Multi-agent entry point (V1)
ralph_daemon.py          # Background service with REM sleep
ralph_voice.py           # Voice input interface
wheeler_recall.py        # CLI: query Wheeler memory
wheeler_store.py         # CLI: store to Wheeler memory
```

---

## Key Commands

```bash
# V2: Single-model architecture (recommended)
python ralph_simple.py "Create a Python function that validates email addresses"

# V1: Multi-agent swarm
./ralph_loop.sh "Create a Python function that validates email addresses"

# V1 with Docker sandbox
./ralph_loop.sh --sandbox "objective here"

# Check installed models
ollama list

# Run tests
pytest tests/

# Pull the single-model
ollama pull qwen3-coder-next
```

---

## Models

### V2 (Single-Model)

| Role | Model | Architecture |
|------|-------|-------------|
| All roles | `qwen3-coder-next` | 80B MoE (3B active), 256K context |
| Embeddings | `nomic-embed-text` | 274MB |

### V1 (Multi-Agent)

| Role | Model | Size |
|------|-------|------|
| Translator | `phi3:mini` | 2.2GB |
| Orchestrator | `deepseek-r1:14b` | 9.0GB |
| Engineer | `qwen2.5-coder:14b` | 9.0GB |
| Designer | `mistral-nemo:12b` | 7.1GB |
| ASIC (tiny code) | `deepseek-coder:1.3b` | 776MB |
| ASIC (small code) | `qwen2.5-coder:1.5b` | 986MB |
| ASIC (ultra-tiny) | `tinyllama:1.1b` | 637MB |
| Embeddings | `nomic-embed-text` | 274MB |

---

## Key Systems

### Wheeler Memory (Spatial Associative Memory)
- Cellular automata-based memory using 2D spatial dynamics
- NPU-accelerated via OpenVINO on Intel AI Boost
- Bridge: `ralph_core/wheeler.py` (singleton `WheelerMemoryBridge`)
- CLI tools: `wheeler_recall.py`, `wheeler_store.py`

### Symbolic Collapse Model (SCM)
- Theoretical foundation: `/docs/SCM_AXIOMS.md`
- Implementation: `ralph_core/wheeler_weights.py`
- Core principle: *meaning is what survives symbolic pressure*
- Three stability dimensions: hit count, frame persistence, compression survival
- Stability score (0.0-1.0) drives context token allocation

### Context Budget Weighting
- Implementation: `ralph_core/context_budget.py`
- High stability (>0.7): full token allocation
- Medium stability (0.3-0.7): proportional allocation
- Low stability (<0.3): compressed or dropped
- Token constraints *are* the symbolic pressure from SCM

### Forklift Protocol (Intelligent Memory Loading)
- Implementation: `ralph_core/forklift.py`
- Task-aware memory retrieval (replaces "dump everything")
- Scope presets: minimal, standard, comprehensive
- Loads: lessons, guidelines, facts, context, file suggestions

### Security Checkpoint
- All outputs validated before execution
- Guards (type checking, permissions, sanitization)
- Dogs (malware sniffing, secret detection)
- Towers (audit logging)
- Gate Out (rate limiting)

---

## Development Guidelines

- Follow the Ralph Wiggum Method: iterate until `<promise>COMPLETE</promise>`
- Git is memory - commit frequently
- Lessons are stored in `~/.ralph/global_memory/lessons.json`
- Guidelines (meta-rules from REM Sleep) in `~/.ralph/global_memory/global_guidelines.md`
- Use vector DB for semantic recall across projects
- Stability metrics persisted in `memory/stability_metrics.json`
- Test with `pytest tests/` before committing
- See `conductor/workflow.md` for task lifecycle and TDD process

---

## Hardware Environment

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core Ultra 7 265K (Arrow Lake) |
| Cores | 20 (8 P-cores + 12 E-cores) |
| Max Frequency | 5.5 GHz |
| L3 Cache | 30 MB |
| NPU | Intel AI Boost (NPU 3) - Wheeler dynamics |
| GPU | AMD RX 9070 XT (16GB VRAM) |
| Motherboard | ASRock Z890 Taichi OCF |
| RAM | 32 GB DDR5 |
| Storage | 2 TB Solidigm NVMe SSD |
| OS | CachyOS (Linux 6.19.x) |

VRAM monitoring via `ralph_core/vram_sentinel.py` (ROCm/rocm-smi).
NPU acceleration via `ralph_core/npu_engine.py` (OpenVINO).

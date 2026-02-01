# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ralph AI is a hierarchical autonomous agent swarm system that uses local LLMs (via Ollama) to accomplish tasks iteratively. The system follows the "Ralph Wiggum Method": iterate until `<promise>COMPLETE</promise>` is detected.

The repository contains three main components:
1. **ai_tech_stack/** - The core Ralph AI swarm system
2. **ai_tech_stack/conductor/** - Project management framework with tracks, workflows, and development guidelines
3. **wheeler_ai_training/** - Wheeler Memory neural network for text-to-2D-grid encoding

## Key Commands

```bash
# Run Ralph with an objective (legacy pipeline)
cd ai_tech_stack
./ralph_loop.sh "Your objective here"

# Run with V2 message-driven pipeline
./ralph_loop.sh --v2 "Your objective here"

# Run with Docker sandbox isolation
./ralph_loop.sh --sandbox "Your objective here"

# Run the daemon (monitors queue, handles REM sleep consolidation)
python ralph_daemon.py

# Start the web UI
./start_ui.sh

# Run tests
pytest ralph_core/tests/ -v
pytest ralph_core/tests/test_bus.py -v  # Single test file

# Wheeler Memory with local Ollama
./ralph_wheeler_local.sh task.md --model llama3

# Check available Ollama models
ollama list
```

## Architecture

Ralph AI uses a 3-tier "compound" architecture:

```
Human Input → [Translator] → [Orchestrator] → [Middle Management] → [ASICs]
                                                   ↕ bidirectional
                                               [Engineer] ↔ [Designer]
```

### Core Agents (in `ai_tech_stack/ralph_core/agents/`)

| Agent | Model | Purpose |
|-------|-------|---------|
| Translator | phi3:mini | Human input → TaskSpec |
| Orchestrator | deepseek-r1:14b | Strategic planning |
| Engineer | qwen2.5-coder:14b | Code generation |
| Designer | mistral-nemo:12b | Code review |
| Reflector | - | Learns from past runs |
| Debugger | - | Error analysis |
| Estimator | - | Task prioritization |
| Sleeper | - | REM sleep memory consolidation |

### ASICs (micro-specialists in `ai_tech_stack/ralph_core/asic/`)

Small, fast models for specific tasks: regex, json, sql, test, fix, doc, tiny_code, sm_code.

### Message Bus (`ai_tech_stack/ralph_core/protocols/`)

V2 mode uses a message-driven pipeline where agents communicate via typed messages:

| Category | Messages |
|----------|----------|
| Work | WORK_REQUEST, CODE_OUTPUT, REVISION_REQUEST, COMPLETE, ERROR |
| ASIC | ASIC_REQUEST, ASIC_RESPONSE |
| Memory | FORKLIFT_REQUEST, FORKLIFT_RESPONSE |
| Tools | TOOL_REQUEST, TOOL_RESPONSE, TOOL_CONFIRM |
| Sleep | REM_SLEEP_START, REM_SLEEP_COMPLETE |
| Learning | CONSOLIDATION_REQUEST, CONSOLIDATION_RESPONSE |
| Debug | DIAGNOSTIC (bypasses circuit breakers for critical error recovery) |

Circuit breakers: max 50 messages, max 3 revision rounds, 300s timeout.

### Security Layer (`ai_tech_stack/ralph_core/security/`)

All agent outputs must pass `checkpoint()` before user exposure:
- towers.py - Audit logging with structured JSON
- dogs.py - Malware/secret detection (hardcoded patterns)
- guards.py - Input/output validation
- gate_out.py - Final output approval
- checkpoint.py - Orchestrates all security checks

### Silicon-Native Hardware Integration

Three specialized GPU engines for hardware-optimized inference:

| Engine | Hardware | Purpose | Models |
|--------|----------|---------|--------|
| iGPU | Intel Xe-LPG | ASIC micro-tasks | qwen2.5-coder:1.5b, tinyllama:1.1b |
| NPU | Intel AI Boost | Wheeler dynamics, memory consolidation | 64x64 frame processing |
| dGPU | AMD via Ollama | Primary reasoning | deepseek-r1:14b, qwen2.5-coder:14b |

### Tool System (`ai_tech_stack/ralph_core/tool_system/`)

- registry.py - Tool definitions with category, permission_level, side_effects
- dispatcher.py - Routes tool requests to handlers
- handler.py - Executes tools and captures results
- Permission levels: ALLOW, WARN, DENY (for generated code safety)

## Key Files

- `ai_tech_stack/ralph_loop.sh` - Main execution loop
- `ai_tech_stack/ralph_core/runner.py` - Iteration handler (connects shell to swarm)
- `ai_tech_stack/ralph_core/swarm.py` - Agent interface exports
- `ai_tech_stack/ralph_core/memory.py` - Context and vector DB
- `ai_tech_stack/ralph_core/forklift.py` - Selective memory loading
- `ai_tech_stack/ralph_daemon.py` - Background daemon with REM sleep

## Memory System

Three memory tiers:

| Tier | Storage | Purpose |
|------|---------|---------|
| Short-term | `context.json` | Working memory for current task |
| Long-term | ChromaDB (`ralph_local_memory`, `ralph_global_memory`) | Vector DB with nomic-embed-text embeddings |
| Wheeler Bridge | `~/.wheeler_memory/` | Spatial dynamics (text -> 64x64 grid -> attractors) |

Lessons stored in `~/.ralph/global_memory/lessons.json`.

The Forklift protocol (`forklift.py`) selectively loads memories by scope: "minimal", "standard", or "comprehensive".

### REM Sleep & Consolidation

The daemon (`ralph_daemon.py`) triggers REM sleep cycles after 60s idle:
- Consolidator clusters lessons using semantic similarity (0.80 threshold, min 3 lessons)
- Synthesizes higher-level guidelines from clusters
- Sleeper agent initiates cycles and updates global memory

## Wheeler Memory (Neural Net)

Located in `wheeler_ai_training/`, this implements a text autoencoder with 2D spatial latent space:

```
Text → Encoder → 2D Grid (64x64) → Decoder → Text
```

The grid is compatible with Wheeler Memory dynamics for spatial reasoning.

### Training (AMD GPU with ROCm)

```bash
cd wheeler_ai_training
source venv/bin/activate
python train.py --batch_size 64 --grid_size 64 --d_model 256 --epochs 10 --fp16
```

## Environment

- **CPU**: Intel Core Ultra 7 265K (20 cores)
- **GPU**: AMD Radeon RX 9070 XT (RDNA4) with ROCm
- **RAM**: 32GB DDR5
- **OS**: CachyOS Linux

For ROCm setup:
```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.0
export PYTORCH_ROCM_ARCH="gfx1200"
```

## Conductor System (`ai_tech_stack/conductor/`)

Project management framework for development tracks:
- `workflow.md` - Strict TDD-based task workflow with phase completion protocols
- `tech-stack.md` - Hardware architecture documentation
- `tracks.md` - Registry of development initiatives
- `tracks/` - Individual track directories (e.g., `ralph_loop_dgpu_20260131/`)

**Phase-Based TDD Workflow**:
1. Pick task from plan.md (sequential order)
2. Mark in-progress: `[ ]` -> `[~]`
3. Write failing tests (RED)
4. Implement minimum code (GREEN)
5. Refactor with test safety (BLUE)
6. Verify >80% coverage
7. Document deviations in tech-stack.md
8. Commit with clear message
9. Attach git notes with task summary
10. Update plan with commit SHA (first 7 chars)
11. Mark task complete: `[x]`

## Executor & Sandbox

`executor.py` provides:
- Sandbox mode: wraps commands in `docker exec ralph_sandbox /bin/bash -c <command>`
- Verification: `run_tests()`, `run_lint()`, `run_typecheck()`
- 30 second default timeout

## Development Notes

- V2 mode: Set `RALPH_V2=1` or use `--v2` flag
- Completion signal: `<promise>COMPLETE</promise>` in output
- Git is treated as memory - commit frequently
- Code output format: Use ` ```python:filename.py ` for file saves
- Command execution: Use `<execute>command</execute>` tags
- Test coverage target: >80% for new code
- CI/CD: `.github/workflows/` contains ci.yml, codeql.yml, and Gemini-based automation

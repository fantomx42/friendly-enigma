# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ralph AI is a hierarchical autonomous agent swarm system that uses local LLMs (via Ollama) to accomplish tasks iteratively. The system follows the "Ralph Wiggum Method": iterate until `<promise>COMPLETE</promise>` is detected.

**Repository components:**
1. **ai_tech_stack/** - Core Ralph AI swarm system (Python)
2. **ralph-vscode/** - Void Editor extension (TypeScript, VS Code API)
3. **ai_tech_stack/ralph_ui/** - Web dashboard (FastAPI + vanilla JS)
4. **ai_tech_stack/conductor/** - Project management framework (tracks, workflows, TDD)
5. **wheeler_ai_training/** - Wheeler Memory neural network (text-to-2D-grid autoencoder)

## Key Commands

```bash
# All commands from ai_tech_stack/ directory
cd ai_tech_stack

# Run Ralph with an objective (single-model, qwen3:8b)
./ralph_loop.sh "Your objective here"

# Override model or context window
RALPH_MODEL=qwen2.5-coder:14b ./ralph_loop.sh "Your objective"
RALPH_NUM_CTX=16384 ./ralph_loop.sh "Your objective"  # Smaller context to save VRAM

# Legacy multi-agent mode (V2 message bus)
PYTHONPATH=. venv/bin/python3 ralph_core/runner.py "objective" 1

# Void Editor extension
cd ralph-vscode && npm run compile         # Build the extension
npm run package                            # Create .vsix package

# Web UI
./start_ui.sh                              # Port 8000 (terminal) + 8001 (neural dashboard)

# Background services
python ralph_daemon.py                     # Monitors queue, REM sleep consolidation
python ralph_voice.py                      # Voice interface

# Testing (activate venv first, set PYTHONPATH)
source venv/bin/activate
PYTHONPATH=. pytest ralph_core/tests/ -v                        # All tests
PYTHONPATH=. pytest ralph_core/tests/test_bus.py -v             # Single file
PYTHONPATH=. pytest ralph_core/tests/test_bus.py::test_name -v  # Single test
PYTHONPATH=. pytest --cov=ralph_core --cov-report=html          # Coverage

# Linting (runs in CI)
flake8 ralph_core/ --select=E9,F63,F7,F82  # Syntax errors only
flake8 ralph_core/ --max-line-length=127    # Full lint
```

## Void Editor Extension (`ralph-vscode/`)

VS Code/Void Editor extension providing integrated Ralph AI control. Installs into the Void Editor sidebar.

**Key source files:**
- `src/extension.ts` - Entry point: activate/deactivate, command registration
- `src/types.ts` - Enums (RalphState, LlamaServerState), interfaces
- `src/config.ts` - Settings accessor (reads VS Code workspace config)
- `src/ralph/ralphRunner.ts` - Spawns ralph_loop.sh, manages lifecycle, emits events
- `src/ralph/outputParser.ts` - State machine parsing stdout signals into typed events
- `src/ralph/llamaServer.ts` - llama-server start/stop/health polling
- `src/views/sidebarProvider.ts` - TreeDataProvider: objective, controls, iteration, Wheeler status
- `src/views/statusBar.ts` - Status bar item: idle/running/paused/complete/error
- `src/views/wheelerWebview.ts` - WebviewViewProvider: memory list from ~/.wheeler_memory/

**Build:** `cd ralph-vscode && npm run compile`
**Package:** `cd ralph-vscode && npm run package` (creates `.vsix`)
**Dev launch:** Open Void with `--extensionDevelopmentPath=./ralph-vscode`

**Features:**
- Activity bar icon opening Ralph AI sidebar
- Sidebar: objective, run state, iteration progress, model, Wheeler Memory count, llama-server status
- Commands: Start, Stop, Pause/Resume, Set Objective, Show Output, llama-server management
- Status bar: state with icon (click to show output channel)
- Output channel: real-time ralph_loop.sh stdout streaming
- Wheeler Memory webview: reads `~/.wheeler_memory/memory.json`, shows last 10 entries
- Completion notification when `<promise>COMPLETE</promise>` detected
- llama-server auto-start (optional, configurable)

## Web UI (`ralph_ui/`)

Two FastAPI servers:
- `backend/main.py` (port 8000) - Terminal UI, serves `frontend/index.html`, WebSocket streams logs from `ralph.log`
- `backend/dashboard.py` (port 8001) - Neural dashboard, serves `frontend/neural.html`, D3.js agent visualization + Wheeler Memory canvas

## Architecture

### Current: Single-Model (v3.0)

```
Human Input → [qwen3-coder-next] → Wheeler Memory ↔ Stability Scoring → Output
               All cognitive roles    (hit_count, frame_persistence, compression_survival)
```

**Model:** `qwen3:8b` (8B dense, 32K context, thinking + tool use)
**Entry point:** `ralph_simple.py` via `ralph_loop.sh`
**VRAM budget:** ~8GB model weights + ~8GB KV cache (32K ctx) on 16GB GPU

### SCM Foundation

The Symbolic Collapse Model (see `/docs/SCM_AXIOMS.md`) provides theoretical grounding:
- Meaning is what survives symbolic pressure
- Wheeler stability scoring weights context by pattern survival

### Legacy: Multi-Agent Hierarchy

The previous multi-agent system is preserved in `ralph_core/agents/` but not used by default:

| Agent | Model | Purpose |
|-------|-------|---------|
| Translator | phi3:mini | Human input → TaskSpec |
| Orchestrator | deepseek-r1:14b | Strategic planning |
| Engineer | qwen2.5-coder:14b | Code generation |
| Designer | mistral-nemo:12b | Code review |

To use legacy mode, run `ralph_core/runner.py` directly instead of `ralph_simple.py`.

### ASICs (`ralph_core/asic/`)

Ultra-small specialist models for micro-tasks (legacy, not used in v3): regex, json, sql, test, fix, doc, tiny_code, sm_code.

### Message Bus (`ralph_core/protocols/`)

V2 mode uses typed messages with circuit breakers (max 50 messages, 3 revision rounds, 300s timeout):

| Category | Messages |
|----------|----------|
| Work | WORK_REQUEST, CODE_OUTPUT, REVISION_REQUEST, COMPLETE, ERROR |
| ASIC | ASIC_REQUEST, ASIC_RESPONSE |
| Memory | FORKLIFT_REQUEST, FORKLIFT_RESPONSE |
| Tools | TOOL_REQUEST, TOOL_RESPONSE, TOOL_CONFIRM |
| Sleep | REM_SLEEP_START, REM_SLEEP_COMPLETE |
| Debug | DIAGNOSTIC (bypasses circuit breakers) |

### Security Layer (`ralph_core/security/`)

All agent outputs pass `checkpoint()` before user exposure:
- `checkpoint.py` - Orchestrates all checks
- `towers.py` - Audit logging (structured JSON)
- `dogs.py` - Malware/secret detection
- `guards.py` - Input/output validation
- `gate_out.py` - Final output approval

### Tool System (`ralph_core/tool_system/`)

- `registry.py` - Tool definitions (category, permission_level, side_effects)
- `dispatcher.py` - Routes tool requests
- `handler.py` - Executes tools, captures results
- Permission levels: ALLOW, WARN, DENY

### Hardware Integration

| Engine | Hardware | Purpose |
|--------|----------|---------|
| iGPU | Intel Xe-LPG | ASIC micro-tasks |
| NPU | Intel AI Boost | Wheeler dynamics, memory consolidation |
| dGPU | AMD via Ollama | Primary reasoning |

## Key Files

| File | Purpose |
|------|---------|
| `ralph_loop.sh` | Entry point (delegates to ralph_simple.py) |
| `ralph_simple.py` | Single-model loop with Wheeler integration |
| `ralph_core/wheeler.py` | Wheeler Memory bridge with `recall_with_stability()` |
| `wheeler_recall.py` | CLI for stability-weighted memory recall |
| `ralph_core/runner.py` | Legacy multi-agent iteration handler |
| `ralph_core/memory.py` | Context and vector DB |
| `ralph_core/forklift.py` | Selective memory loading |
| `ralph_core/executor.py` | Command execution with sandbox support (30s timeout) |
| `ralph_daemon.py` | Background daemon with REM sleep |

## Memory System

| Tier | Storage | Purpose |
|------|---------|---------|
| Short-term | `context.json` | Working memory for current task |
| Long-term | ChromaDB (`ralph_local_memory`, `ralph_global_memory`) | Vector DB with nomic-embed-text |
| Wheeler Bridge | `~/.wheeler_memory/` | Spatial dynamics (text → 128x128 grid → attractors) |

### Wheeler Stability Scoring

`recall_with_stability()` scores memories 0.0-1.0:
| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency (sigmoid normalized) |
| frame_persistence | 30% | Re-encode matches stored frame |
| compression_survival | 30% | Pattern survives 10-tick dynamics |

Higher stability = more context budget. Implements SCM Axiom 2: "Meaning is the remainder after compression."

- Lessons stored in `~/.ralph/global_memory/lessons.json`
- Forklift protocol loads memories by scope: "minimal", "standard", "comprehensive"
- REM sleep triggers after 60s idle, clusters lessons (0.80 similarity threshold, min 3)

## Conductor System (`conductor/`)

TDD-based project management. See `conductor/workflow.md` for full protocol.

**Task lifecycle:** select from `plan.md` → mark `[~]` → RED → GREEN → BLUE → verify >80% coverage → commit with git notes → mark `[x]` with SHA.

## Development Notes

- **V2 mode**: `RALPH_V2=1` or `--v2` flag
- **Completion signal**: `<promise>COMPLETE</promise>` in output
- **Code output format**: ` ```python:filename.py ` for file saves
- **Command execution**: `<execute>command</execute>` tags
- **Sandbox mode**: Wraps commands in `docker exec ralph_sandbox /bin/bash -c`
- **CI**: flake8 syntax checks, structure validation, Trivy security scan
- **Python venv**: Always `source ai_tech_stack/venv/bin/activate` and set `PYTHONPATH=.` before running tests

## ROCm Setup (AMD GPU)

```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.0
export PYTORCH_ROCM_ARCH="gfx1200"
```

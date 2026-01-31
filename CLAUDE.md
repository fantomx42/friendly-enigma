# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ralph AI is a hierarchical autonomous agent swarm system that uses local LLMs (via Ollama) to accomplish tasks iteratively. The system follows the "Ralph Wiggum Method": iterate until `<promise>COMPLETE</promise>` is detected.

The repository contains two main components:
1. **ai_tech_stack/** - The core Ralph AI swarm system
2. **wheeler_ai_training/** - Wheeler Memory neural network for text-to-2D-grid encoding

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

V2 mode uses a message-driven pipeline where agents communicate via typed messages: WORK_REQUEST, CODE_OUTPUT, REVISION_REQUEST, ASIC_REQUEST, ASIC_RESPONSE, COMPLETE, ERROR.

### Security Layer (`ai_tech_stack/ralph_core/security/`)

All outputs pass through security checkpoint with:
- towers.py - Audit logging
- dogs.py - Malware/secret detection
- guards.py - Validation layer
- gate_out.py - Final output gate

## Key Files

- `ai_tech_stack/ralph_loop.sh` - Main execution loop
- `ai_tech_stack/ralph_core/runner.py` - Iteration handler (connects shell to swarm)
- `ai_tech_stack/ralph_core/swarm.py` - Agent interface exports
- `ai_tech_stack/ralph_core/memory.py` - Context and vector DB
- `ai_tech_stack/ralph_core/forklift.py` - Selective memory loading
- `ai_tech_stack/ralph_daemon.py` - Background daemon with REM sleep

## Memory System

- Short-term context: `context.json`
- Long-term lessons: `~/.ralph/global_memory/lessons.json`
- Vector DB for semantic recall

The Forklift protocol selectively loads relevant memories based on current objective.

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

## Development Notes

- V2 mode: Set `RALPH_V2=1` or use `--v2` flag
- Completion signal: `<promise>COMPLETE</promise>` in output
- Git is treated as memory - commit frequently
- Code output format: Use ` ```python:filename.py ` for file saves
- Command execution: Use `<execute>command</execute>` tags

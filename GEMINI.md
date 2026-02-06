# Ralph AI

## Project Overview

Ralph AI is a hierarchical autonomous agent system that uses local LLMs (via Ollama) to accomplish tasks iteratively. The system follows the "Ralph Wiggum Method": iterate until `<promise>COMPLETE</promise>` is detected.

**Current Architecture:** Single-model (v3.0) using `qwen3-coder-next` with Wheeler Memory integration.

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `ai_tech_stack/` | Core Ralph AI system (Python) |
| `ai_tech_stack/ralph_gui/` | Native desktop GUI (Rust/egui) |
| `ai_tech_stack/ralph_ui/` | Web dashboard (FastAPI + vanilla JS) |
| `ai_tech_stack/conductor/` | Project management framework (TDD workflows) |
| `wheeler_ai_training/` | Wheeler Memory neural network (text-to-2D-grid autoencoder) |

## Key Commands

```bash
cd ai_tech_stack

# Run Ralph with an objective
./ralph_loop.sh "Your objective here"

# Native GUI
cd ralph_gui && cargo build --release
./target/release/ralph_gui

# Web UI
./start_ui.sh  # Port 8000 (terminal) + 8001 (neural dashboard)

# Testing
source venv/bin/activate
PYTHONPATH=. pytest ralph_core/tests/ -v
```

## Architecture

### Current: Single-Model (v3.0)

```
Human Input -> [qwen3-coder-next] -> Wheeler Memory <-> Stability Scoring -> Output
```

- **Model:** qwen3-coder-next (80B MoE, 3B active per pass, 256K context)
- **Entry point:** `ralph_simple.py` via `ralph_loop.sh`

### Wheeler Memory

Text-to-spatial-grid autoencoder providing stability-weighted context:

| Metric | Weight | Meaning |
|--------|--------|---------|
| hit_count | 40% | Activation frequency |
| frame_persistence | 30% | Re-encode matches stored frame |
| compression_survival | 30% | Pattern survives dynamics |

### Legacy Multi-Agent (V2)

Preserved in `ralph_core/agents/` but not used by default. Includes Translator, Orchestrator, Engineer, and Designer agents.

## Key Files

| File | Purpose |
|------|---------|
| `ralph_loop.sh` | Entry point |
| `ralph_simple.py` | Single-model loop with Wheeler integration |
| `ralph_core/wheeler.py` | Wheeler Memory bridge |
| `ralph_core/runner.py` | Legacy multi-agent handler |
| `ralph_daemon.py` | Background daemon with REM sleep |

## Technical Stack

- **Languages:** Python 3.8+ (Core), Rust (GUI), JavaScript (Web UI)
- **AI/ML:** Ollama (Inference), PyTorch (Training), ChromaDB (Vector Store)
- **Hardware:** AMD GPU via ROCm/Ollama, Intel NPU for Wheeler dynamics

## Development Notes

- **Completion signal:** `<promise>COMPLETE</promise>` in output
- **Code output format:** ` ```python:filename.py ` for file saves
- **Command execution:** `<execute>command</execute>` tags
- **Local first:** All models run locally via Ollama

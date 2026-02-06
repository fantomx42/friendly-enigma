# Ralph AI Project Context

## Overview

Ralph AI is an autonomous agent system using local LLMs (Ollama) to accomplish tasks iteratively. It follows the "Ralph Wiggum Method": iterate until `<promise>COMPLETE</promise>` is detected.

## Current Architecture (v3.0)

Single-model system using `qwen3-coder-next` with Wheeler Memory integration:

```
Human Input -> [qwen3-coder-next] -> Wheeler Memory -> Output
```

Entry point: `ai_tech_stack/ralph_loop.sh`

## Repository Layout

- `ai_tech_stack/` - Core Python system
- `ai_tech_stack/ralph_gui/` - Rust/egui desktop GUI
- `ai_tech_stack/ralph_ui/` - FastAPI + JS web dashboard
- `wheeler_ai_training/` - Wheeler Memory neural network

## Key Commands

```bash
cd ai_tech_stack

# Run Ralph
./ralph_loop.sh "Your objective"

# Build GUI
cd ralph_gui && cargo build --release

# Run tests
source venv/bin/activate
PYTHONPATH=. pytest ralph_core/tests/ -v
```

## Important Files

- `ralph_simple.py` - Main single-model loop
- `ralph_core/wheeler.py` - Wheeler Memory bridge
- `ralph_core/memory.py` - Context and vector DB
- `ralph_daemon.py` - Background daemon

## Conventions

- Local-first: All models run via Ollama
- Completion signal: `<promise>COMPLETE</promise>`
- Code blocks: ` ```python:filename.py `
- Commands: `<execute>command</execute>` tags

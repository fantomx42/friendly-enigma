# Implementation Plan: Ralph Loop & dGPU Orchestration

## Phase 1: VRAM & Model Management [checkpoint: bd5e329]
- [x] Task: Implement `OllamaClient` with model loading control [commit: 5ccb4c5]
    - [x] Write Tests: Verify API connection and basic inference
    - [x] Implement Feature: Create a wrapper that explicitly manages model residency (load/unload)
- [x] Task: VRAM Sentinel [commit: 5ccb4c5]
    - [x] Write Tests: Verify `rocm-smi` integration
    - [x] Implement Feature: Create a monitoring utility to check available VRAM before model loads
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Cognitive Loop implementation
- [x] Task: Create `RalphLoop` controller
    - [x] Write Tests: Mock the agent phases and verify transition logic
    - [x] Implement Feature: The main state machine (TRANSLATE -> PLAN -> CODE -> VERIFY -> REFLECT)
- [x] Task: Message Bus Integration
    - [x] Write Tests: Verify asynchronous message delivery between agents
    - [x] Implement Feature: Connect `RalphLoop` to the internal message bus
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Agent Integration (dGPU)
- [x] Task: Orchestrator Implementation (`deepseek-r1:14b`)
    - [x] Write Tests: Verify structured output (JSON) for task decomposition
    - [x] Implement Feature: Connect the agent to Ollama/dGPU
- [x] Task: Engineer Implementation (`qwen2.5-coder:14b`)
    - [x] Write Tests: Verify code generation and file writing integration
    - [x] Implement Feature: Connect the agent to Ollama/dGPU
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Full System Verification
- [ ] Task: Update `ralph_loop.sh`
    - [ ] Implement Feature: Replace legacy loop with call to `ai_tech_stack/ralph_core/runner.py`
- [ ] Task: End-to-End "Hello World" Task
    - [ ] Write Tests: Run a full loop for a simple task (e.g., "Create a sum function")
    - [ ] Implement Feature: Verify successful completion and Wheeler memory storage
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

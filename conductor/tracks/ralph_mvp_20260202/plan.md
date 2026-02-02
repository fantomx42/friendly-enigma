# Implementation Plan - Finalize Ralph AI MVP

This plan focuses on the critical path to MVP: Dependencies -> Pipeline Glue -> Model Check -> End-to-End Verification.

## Phase 1: Environment & Dependencies [checkpoint: 08e4b6a]
- [x] Task: Install Python dependencies 11a9ddc
    - [x] Install dependencies using `pip install -r ai_tech_stack/requirements.txt`.
    - [x] Verify installation of critical packages (chromadb, openvino, ollama).
- [x] Task: Verify and Pull Ollama Models 613b6ba
    - [x] Check for existence of `phi3:mini`, `deepseek-r1:14b`, `qwen2.5-coder:14b`, and `mistral-nemo:12b`.
    - [x] Create a script or command to pull any missing models.
- [x] Task: Conductor - User Manual Verification 'Environment & Dependencies' (Protocol in workflow.md)

## Phase 2: Pipeline Completion (Shell & Runner)
- [ ] Task: Finalize `ralph_loop.sh` logic
    - [ ] Review `ai_tech_stack/ralph_loop.sh` for "Phase 4" TODOs.
    - [ ] Implement the call to `ai_tech_stack/ralph_core/runner.py`.
    - [ ] Ensure proper argument passing (user objective) and exit code handling.
- [ ] Task: Verify `runner.py` Entry Point
    - [ ] Ensure `runner.py` can be invoked from the shell and correctly initializes the V2 pipeline.
- [ ] Task: Conductor - User Manual Verification 'Pipeline Completion' (Protocol in workflow.md)

## Phase 3: Integration Testing (The "Hello World")
- [ ] Task: Create End-to-End Test Script
    - [ ] Create `ai_tech_stack/tests/test_hello_world.py` (or shell equivalent).
    - [ ] Test should invoke `ralph_loop.sh` with a simple objective (e.g., "Write a poem about coding").
    - [ ] Assert that the process completes with a success signal.
- [ ] Task: Fix Import/Runtime Errors
    - [ ] Run the test and iteratively fix any `ImportError` or runtime exceptions that arise during the first full execution.
- [ ] Task: Conductor - User Manual Verification 'Integration Testing' (Protocol in workflow.md)
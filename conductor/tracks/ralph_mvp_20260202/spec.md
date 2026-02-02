# Track Specification: Finalize Ralph AI MVP

## Objective
To resolve critical blockers and finalize the Minimum Viable Product (MVP) for Ralph AI. This involves ensuring all dependencies are installed, the execution pipeline (`ralph_loop.sh` -> `runner.py`) is seamless, local Ollama models are verified, and a "Hello World" end-to-end integration test passes successfully.

## Context
The codebase is 95% complete but lacks the final "glue" to operate as a cohesive system. Critical issues include:
- Missing Python dependencies.
- Incomplete shell script integration for the V2 pipeline.
- Unverified local model availability.
- Absence of a comprehensive integration test.

## Requirements

### 1. Dependency Management
- **Action:** Install all dependencies listed in `ai_tech_stack/requirements.txt`.
- **Validation:** Verify successful installation of key packages (`chromadb`, `openvino`, `ollama`).

### 2. Pipeline Integration (`ralph_loop.sh`)
- **Action:** Complete "Phase 4" of the `ralph_loop.sh` script.
- **Requirement:** Ensure it correctly invokes the Python `runner.py` with the user's objective and handles exit codes properly.

### 3. Model Verification
- **Action:** Verify that the required Ollama models (`phi3:mini`, `deepseek-r1:14b`, `qwen2.5-coder:14b`, `mistral-nemo:12b`) are pulled and available.
- **Fallback:** Provide a script or command to pull missing models automatically.

### 4. Integration Testing
- **Action:** Create and run an end-to-end "Hello World" test.
- **Goal:** Prove that an objective flows from the CLI -> `ralph_loop.sh` -> `runner.py` -> Agents -> Completion.

## Acceptance Criteria
- [ ] `pip install` completes without conflict.
- [ ] `ralph_loop.sh` successfully launches the Python core.
- [ ] All required Ollama models are present.
- [ ] The "Hello World" integration test passes, demonstrating a full autonomous cycle.

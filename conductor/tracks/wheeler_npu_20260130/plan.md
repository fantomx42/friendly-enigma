# Implementation Plan: Wheeler Memory NPU Bridge

## Phase 1: Environment & Discovery [checkpoint: 8fbf24d]
- [x] Task: Verify OpenVINO NPU visibility and drivers
    - [x] Run `python -c "import openvino; print(openvino.Core().available_devices)"`
    - [x] Ensure 'NPU' is listed in the output
- [x] Task: Analyze current `WheelerAI` implementation in `wheeler_ai_training/`
    - [x] Identify core math kernels in `dynamics.py` (or equivalent)
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: NPU Kernel Development
- [x] Task: Implement Wheeler Attractor Dynamics in OpenVINO
    - [x] Write Tests: Define expected output for a single dynamic tick
    - [x] Implement Feature: Create OpenVINO model/kernel for spatial updates
- [x] Task: Create `NPUWheelerEngine` class
    - [x] Write Tests: Verify initialization and device loading
    - [x] Implement Feature: Create the wrapper class for OpenVINO inference
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Bridge Integration
- [x] Task: Update `ai_tech_stack/ralph_core/wheeler.py` to support NPU offload
    - [x] Write Tests: Test the detection and switching logic (CPU vs NPU)
    - [x] Implement Feature: Integrate `NPUWheelerEngine` into `WheelerMemoryBridge`
- [x] Task: Verify "Dreaming" (Autonomic) offload
    - [x] Write Tests: Verify background thread performance
    - [x] Implement Feature: Ensure `start_autonomic()` uses the NPU engine
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Verification & Benchmarking
- [x] Task: Comparative Accuracy Test
    - [x] Write Tests: Compare NPU final frames with CPU reference frames
    - [x] Implement Feature: Run 100 random probe tests and assert similarity > 0.99
- [x] Task: Resource Usage Audit
    - [x] Write Tests: Verify 0% dGPU usage during memory updates
    - [x] Implement Feature: Log VRAM and NPU power usage during a 5-minute run
- [x] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

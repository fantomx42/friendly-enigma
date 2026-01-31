# Specification: Ralph Loop & dGPU Orchestration

## Overview
This track focuses on implementing the core cognitive loop of Ralph AI, optimized for the AMD Radeon RX 9070 XT (16GB VRAM) using ROCm and Ollama. It includes the iterative "Ralph Wiggum" logic and the VRAM-aware model swapping mechanism.

## Objectives
- Implement the "Ralph Loop": a Python-managed cycle of Plan -> Execute -> Verify -> Learn.
- Configure `Orchestrator` and `Engineer` agents to use `Ollama` with ROCm acceleration.
- Implement a VRAM Management system to swap 14B models (DeepSeek-R1, Qwen2.5-Coder) within the 16GB limit.
- Update the legacy `ralph_loop.sh` to trigger the new Python-based orchestration.

## Technical Details
- **Hardware:** AMD RX 9070 XT (ROCm 6.2+)
- **Inference Engine:** Ollama
- **Models:** 
    - `deepseek-r1:14b` (Planning)
    - `qwen2.5-coder:14b` (Coding)
- **VRAM Logic:** Ensure only one 14B model is loaded at a time to prevent OOM or performance degradation.
- **Protocol:** Message Bus (internal JSON) for agent communication.

## Success Criteria
- [ ] A task can be completed from natural language input to verified code output.
- [ ] Models swap automatically between planning and coding phases.
- [ ] VRAM usage stays within the 16GB limit of the RX 9070 XT.
- [ ] Each iteration correctly records context into the NPU-accelerated Wheeler Memory.

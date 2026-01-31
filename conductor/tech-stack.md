# Ralph AI: Technology Stack

## Core Language & Runtime
- **Language:** Python 3.8+
- **Execution Environment:** Linux (CachyOS/Arch)
- **Isolation:** Docker (Sandbox mode for untrusted execution)

## Silicon-Native Swarm Architecture
Ralph AI utilizes a hybrid hardware strategy to maximize local performance:

### 1. The Heavy Thinkers (dGPU - RX 9070 XT)
- **Role:** Deep reasoning, planning, and complex code generation.
- **Engine:** Ollama (ROCm).
- **Models:** `deepseek-r1:14b` (Orchestrator) and `qwen2.5-coder:14b` (Engineer).
- **Strategy:** Models swap in/out of the 16GB VRAM buffer.

### 2. The ASIC Specialists (iGPU - Intel Xe-LPG)
- **Role:** Micro-tasks, linting, regex, and syntax fixing.
- **Engine:** OpenVINO (DP4a).
- **Models:** `qwen2.5-coder:1.5b` and `tinyllama:1.1b`.
- **Strategy:** Runs on iGPU using System RAM to keep dGPU VRAM clear.

### 3. The Wheeler Memory Engine (NPU - Intel AI Boost)
- **Role:** Associative memory dynamics and spatial attractor updates.
- **Engine:** Custom OpenVINO Kernels.
- **Logic:** Uses NPU SHAVE DSPs for continuous, low-power background memory consolidation ("Dreaming").

### 4. The System Bus (CPU - Core Ultra 7 265K)
- **Role:** Human-to-machine translation and message routing.
- **Engine:** Ollama (CPU mode).
- **Model:** `phi3:mini`.
- **Strategy:** Leveraging 20 high-performance cores for always-on responsiveness.

## Memory & Knowledge
- **Semantic Memory:** ChromaDB (Vector DB) with `nomic-embed-text`.
- **Associative Memory:** Custom "Wheeler" Spatial Memory (NPU-accelerated).
- **Persistence:** JSONL logging & Git-based context snapshots.

## Interface & Communication
- **API Backend:** FastAPI with Uvicorn.
- **Neural Dashboard:** Vanilla JavaScript (Browser) - Visualizes silicon usage (NPU/iGPU/dGPU).
- **Terminal UI (TUI):** Python + Textual.

## Development Tools
- **Version Control:** Git
- **Testing:** Pytest
- **Linting:** Flake8 / MyPy
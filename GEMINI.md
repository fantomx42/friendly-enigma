# Ralph AI

## Project Overview
Ralph AI is a hierarchical, autonomous multi-agent system designed to execute tasks using local LLMs (via Ollama). It operates on the "Ralph Wiggum Protocol," which embraces iterative failure and refinement to achieve success. The system is designed to run entirely locally, leveraging high-end consumer hardware (specifically AMD ROCm for GPUs).

## Architecture & Components
The project is organized into several key modules:

### Core System (`ai_tech_stack/`)
The primary logic resides here, implementing the multi-agent swarm architecture.
*   **Orchestrator**: Strategic planning (DeepSeek-R1).
*   **Translator**: Input processing (Phi-3).
*   **Engineer/Designer**: Execution and verification (Qwen/Mistral).
*   **Memory**: Vector database (ChromaDB) and "Wheeler Memory" (2D spatial representation).

### Interfaces
*   **CLI**: The primary entry point via `ralph_loop.sh`.
*   **Native GUI (`ai_tech_stack/ralph_gui/`)**: A Rust-based immediate mode GUI using `eframe`/`egui` for real-time monitoring and control.
*   **Web UI (`ai_tech_stack/ralph_ui/`)**: A FastAPI backend with a Vanilla JS frontend, including Playwright E2E tests.

### Experimental (`wheeler_ai_training/`)
Dedicated training environment for the "Wheeler" model—a text-to-spatial-grid autoencoder designed for efficient memory representation.
*   **Stack**: PyTorch (ROCm), Transformers, Datasets.

## Technical Stack
*   **Languages**: Python 3.8+ (Core), Rust (GUI), JavaScript (Web UI).
*   **AI/ML**: Ollama (Inference), PyTorch (Training), ChromaDB (Vector Store).
*   **Hardware Optimization**: Optimized for **AMD Radeon RX 9070 XT** (ROCm/HIP) and **Intel Core Ultra 7 265K**.
*   **OS**: CachyOS (Linux).

## Key Files & Directories
*   `README.md`: Root project documentation.
*   `ai_tech_stack/ralph_loop.sh`: Main CLI execution script.
*   `ai_tech_stack/ralph_daemon.py`: Background service handling periodic tasks (like REM sleep).
*   `ai_tech_stack/ralph_core/`: Python source code for agents, tools, and memory.
*   `ai_tech_stack/ralph_gui/Cargo.toml`: Rust GUI configuration.

## Development & Usage

### Running Ralph
```bash
# Basic Task
./ai_tech_stack/ralph_loop.sh "Your objective here"

# Launch Web UI
./ai_tech_stack/start_ui.sh

# Run Background Daemon
python ai_tech_stack/ralph_daemon.py
```

### Wheeler Training
Training the Wheeler autoencoder requires a specific ROCm environment:
```bash
cd wheeler_ai_training
source venv/bin/activate
python train.py --fp16 --batch_size 64
```

### Rust GUI
```bash
cd ai_tech_stack/ralph_gui
cargo run --release
```

## Conventions
*   **Local First**: All dependencies and models should run locally. No external APIs.
*   **Iterative Design**: The system is built to loop and refine. Failures are expected and handled.
*   **Hardware Awareness**: Scripts often include specific optimizations for the user's AMD GPU and Intel CPU.

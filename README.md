# Ralph AI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-3.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-yellow)
![Rust](https://img.shields.io/badge/rust-1.75+-orange)

> **Ralph is a local, hierarchical autonomous agent system that pairs a single LLM with "Wheeler Memory" — a cellular automata-based spatial memory system — to achieve iterative task completion and true epistemological independence.**

---

## Description

Ralph AI is an experimental autonomous agent designed to run entirely on local hardware (AMD GPUs, Intel NPUs). Unlike traditional agents that rely on vector databases (RAG) for memory, Ralph uses **Wheeler Memory**, a neural-spatial system that encodes text into 2D grid patterns (attractors).

The system follows the **"Ralph Wiggum Protocol"**: a persistent autonomous loop where the agent iterates, reasons, and executes tools until it self-verifies completion with a strict `<promise>COMPLETE</promise>` signal.

### Key Features

*   **Tri-Brid Architecture**: Optimized for heterogenous compute.
    *   **Chalmers (GPU)**: Heavy reasoning and code generation (e.g., Qwen3).
    *   **Wiggum (CPU)**: Reflexive responses and speculative decoding.
    *   **Librarian (NPU)**: Wheeler Memory dynamics and context management.
*   **Wheeler Memory**: A non-RAG memory system where text collapses into stable 2D attractors via cellular automata dynamics. Meaning is defined by what "survives" the dynamics.
*   **Epistemological Independence**: Ralph tracks confidence metrics to form its own "beliefs," preventing sycophancy (blindly agreeing with the user) and allowing for genuine disagreement based on established knowledge.
*   **Local-First Design**: Built for privacy and performance on consumer hardware (CachyOS, ROCm, Ollama).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

*   **OS**: Linux (CachyOS/Arch recommended)
*   **Python**: 3.10 or higher
*   **Rust**: Latest stable (for GUI)
*   **Ollama**: Running locally with `qwen3` models pulled
*   **Hardware**: GPU recommended (AMD ROCm or NVIDIA CUDA)

### Step-by-Step

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ralph.git
    cd ralph
    ```

2.  **Set up the Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install Rust dependencies (optional, for GUI):**
    ```bash
    cd ai_tech_stack/ralph_gui
    cargo build --release
    ```

4.  **Verify Ollama:**
    Ensure Ollama is running and pull the necessary models:
    ```bash
    ollama pull qwen3:8b
    ollama pull nomic-embed-text
    ```

## Quick Start

### CLI Loop (Primary)

The fastest way to use Ralph is via the terminal loop script.

```bash
cd ai_tech_stack
./ralph_loop.sh "Create a snake game in Python using Pygame"
```

### Web UI

For a visual dashboard of the neural memory and task progress:

```bash
./start_ui.sh
```
*   **Dashboard**: http://localhost:8000
*   **Neural View**: http://localhost:8001

## Architecture

Ralph executes tasks through a cyclic process:

1.  **Input**: User defines an objective.
2.  **Wheeler Encoding**: Input is converted to a 128x128 2D frame.
3.  **Dynamics**: The frame evolves via cellular automata until it settles into a stable "attractor."
4.  **Recall**: Stable patterns trigger associated memories (weighted by hit count, persistence, and compression survival).
5.  **Reasoning**: The LLM (Qwen3) receives the weighted context and generates a response/action.
6.  **Feedback**: The result acts as new sensory input, reinforcing or modifying the memory grid.

### Directory Structure

*   `ai_tech_stack/`: Core Python logic, Ollama client, and loop scripts.
*   `wheeler_ai_training/`: The neural network and cellular automata logic for memory.
*   `conductor/`: Project management and TDD workflow tracking.
*   `ralph_gui/`: Native Rust-based desktop interface.

## Configuration

Configuration is primarily handled via environment variables passed to the execution scripts.

**Common Options:**

| Variable | Description | Default |
| :--- | :--- | :--- |
| `RALPH_MODEL` | The Ollama model to use | `qwen3:8b` |
| `RALPH_NUM_CTX` | Context window size | `32768` |
| `WHEELER_DEVICE` | Compute device for memory | `cpu` (or `npu`/`cuda`) |

**Example:**
```bash
RALPH_MODEL=qwen2.5-coder:14b RALPH_NUM_CTX=16384 ./ralph_loop.sh "Analyze this code"
```

## Examples

**Task**: "Fix the bug in the login function."

1.  **Ralph analyzes** the file structure.
2.  **Ralph reads** `login.py`.
3.  **Wheeler Memory** recalls similar debugging patterns or previous context about `login.py`.
4.  **Ralph writes** a test case to reproduce the bug.
5.  **Ralph implements** the fix.
6.  **Ralph verifies** with the test.
7.  **Ralph signals** `<promise>COMPLETE</promise>`.

## Development

### Running Tests

To ensure core functionality:

```bash
source venv/bin/activate
PYTHONPATH=. pytest ai_tech_stack/ralph_core/tests/ -v
```

### Training Wheeler Memory

If you are modifying the memory dynamics:

```bash
cd wheeler_ai_training
python wheeler_ai.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please review `conductor/product-guidelines.md` for coding standards.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors & Acknowledgments

*   **Tristan** - *Lead Developer*
*   **Ralph Team** - *Architecture & Design*

Special thanks to the open-source community behind **Ollama**, **CachyOS**, and **Rust**.

## Support

For issues, please open a ticket in the GitHub Issue Tracker.
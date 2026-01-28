# Ralph AI: Autonomous Local Swarm System

## Project Overview
Ralph AI is a **3-tier hierarchical swarm system** designed for autonomous task execution using **local LLMs** (via Ollama). It implements the "Ralph Wiggum Method": executing tasks through iterative failure, analysis, and refinement until a `<promise>COMPLETE</promise>` signal is achieved.

- **Core Philosophy:** Embrace failure as diagnostic data. Iterate until success.
- **Architecture:** Translator (Input) → Orchestrator (Plan) → Engineer/Designer (Execute/Verify) → ASICs (Micro-tasks).
- **Tech Stack:** Python 3.8+, Ollama, ChromaDB, FastAPI, Shell.
- **Environment:** Optimized for Linux (CachyOS/Arch) with high-end hardware (32GB+ RAM, GPU).

## Architecture & Agents

| Tier | Role | Agent Name | Model | Path |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Translator** | `Translator` | `phi3:mini` | `ralph_core/agents/translator/` |
| **1** | **Orchestrator** | `Ralph` | `deepseek-r1:14b` | `ralph_core/agents/orchestrator/` |
| **2** | **Engineer** | `Qwen` | `qwen2.5-coder:14b` | `ralph_core/agents/engineer/` |
| **2** | **Designer** | `Mistral` | `mistral-nemo:12b` | `ralph_core/agents/designer/` |
| **3** | **ASICs** | *Various* | *Small (1b-3b)* | `ralph_core/asic/` |

**Directory Structure:**
- `ralph_core/`: Core logic (memory, bus, tools).
- `ralph_loop.sh`: **Primary CLI Entry Point.**
- `ralph_daemon.py`: Background task runner.
- `ralph_ui/`: FastAPI backend + Vanilla JS frontend.
- `sandbox/`: Docker isolation environment.

## Operational Commands

### Execution
*   **Run a Task:**
    ```bash
    ./ralph_loop.sh "Your objective here"
    ```
*   **Run in Sandbox (Docker):**
    ```bash
    ./ralph_loop.sh --sandbox "Your objective here"
    ```
*   **Start Background Daemon:**
    ```bash
    python ralph_daemon.py
    ```
*   **Start Web UI:**
    ```bash
    ./start_ui.sh
    # Access at http://localhost:8000
    ```

### Development & Maintenance
*   **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
*   **Pull Required Models:**
    See `README.md` for the full list of `ollama pull` commands.
*   **Run Tests:**
    ```bash
    pytest
    ```

## Development Conventions

*   **Code Style:** Python 3.8+, PEP 8.
*   **Documentation:** Update `STATUS.md` if conflicting info is found; it is the "Source of Truth".
*   **Safety:** Use the sandbox for destructive tasks.
*   **Logging:** Execution logs are stored in `ralph_core/logs/` (often JSONL format).
*   **Memory:** Context is persisted in `memory/` and `~/.ralph/`.

## Critical Context for the Agent
*   **Iterative Process:** Do not expect perfection on the first try. The system is designed to loop. If you (the agent) are debugging Ralph, look for the "Reflector" logs to see what it learned from previous failures.
*   **Hardware:** The user is running on **CachyOS (Linux)** with a **Ryzen 5800X3D** and **RX 9070 XT**. Assume GPU availability.
*   **Tool Usage:** Always consult `TOOLS.md` before implementing new shell integrations.

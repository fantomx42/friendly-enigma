# GEMINI CLI CHECKPOINT - PROJECT RALPH

## 1. System Context
- **Project Name:** Project Ralph (The Autonomous Swarm)
- **Repository:** https://github.com/fantomx42/friendly-enigma
- **Local Path:** `/home/tristan/Documents/Ralph Ai/ai_tech_stack`
- **Hardware:** AMD Ryzen 7 5800X3D / AMD Radeon RX 9070 XT (16GB VRAM)
- **OS:** CachyOS (Arch Linux)

## 2. The Tech Stack (100% Local FOSS)
- **Orchestrator:** `deepseek-r1:14b` (Reasoning/Planning)
- **Engineer:** `qwen2.5-coder:14b` (Coding)
- **Designer:** `mistral-nemo:12b` (Critique/Safety)
- **Backend:** FastAPI + Uvicorn + WebSockets (Python)
- **Frontend:** HTML/JS "Cyberpunk Terminal" (No framework build step required)
- **Database:** `memory/` (Markdown files) + `context.json`

## 3. Operational Capabilities
- **The Ralph Protocol:** A persistent loop (`ralph_loop.sh`) where the agent iterates until it writes `<promise>COMPLETE</promise>`.
- **Autonomous Execution:** Ralph can write code AND execute it using `<execute>command</execute>` tags.
- **Self-Correction:** If execution fails, Ralph sees the error and fixes it in the next loop iteration.
- **Web Interface:** Accessible at `http://ralph.ai` (mapped to 127.0.0.1 in `/etc/hosts`).
    - **Features:** Real-time log streaming, "Compute Time" tracking, latency heartbeat.
    - **Start Command:** `sudo ./start_ui.sh`

## 4. Key Files & Structure
- `ralph_loop.sh`: The entry point for the autonomous agent.
- `ralph_core/`: Contains the brain logic.
    - `swarm.py`: Interface to Ollama models.
    - `runner.py`: The main logic loop (Plan -> Code -> Execute -> Review).
    - `memory.py`: Handles persistent state.
    - `executor.py`: Safe shell command execution.
- `ralph_ui/`: The Web UI code.
    - `backend/main.py`: FastAPI server.
    - `frontend/index.html`: The Terminal interface.
- `start_ui.sh`: Launch script for the UI.

## 5. How to Resume
1. **Start the UI:** `sudo ./start_ui.sh`
2. **Access:** Go to `http://ralph.ai`
3. **Engage:** Type a command in the web UI to trigger `ralph_loop.sh`.
4. **Develop:** Codebase is Git-versioned. Push changes to `origin main`.

## 6. Recent Achievements
- Built fully autonomous loop with verified execution.
- Integrated Memory Module (Short & Long term).
- Deployed Web UI with real-time feedback loop.
- Migrated from generic "bridge" script to robust `ralph_core` package.

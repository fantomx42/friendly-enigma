# Project Ralph Changelog

## Monday, January 12, 2026

### 🚀 Major System Upgrades

**1. The Local Swarm (FOSS Transition)**
- **Objective:** Eliminate dependency on cloud APIs and "gray area" licenses.
- **Action:** Migrated full stack to local Ollama models.
- **Outcome:** 
    - **Orchestrator:** `deepseek-r1:14b` (Reasoning/Planning)
    - **Engineer:** `qwen2.5-coder:14b` (Implementation)
    - **Designer:** `mistral-nemo:12b` (Critique & Safety)
- **Status:** Operational (100% Local, Optimized for RX 9070 XT).

**2. The Ralph Protocol (Autonomous Loop)**
- **Objective:** Enable the agent to self-correct and iterate until a task is truly done.
- **Action:** Implemented the "Ralph Wiggum" loop architecture.
    - `ralph_loop.sh`: The outer control shell (persistence).
    - `ralph_core/runner.py`: The cycle handler (Plan -> Code -> Verify).
    - `ralph_core/swarm.py`: The model interface.
- **Outcome:** Ralph can now receive a task, write code, save files, and verify his own work against a "Definition of Done."

**3. Architectural Refactor**
- **Objective:** Clean up the prototyping mess in the root directory.
- **Action:** 
    - Created `ralph_core/` for the brain (`swarm.py`, `runner.py`, `memory.py`).
    - Created `setup_scripts/` and `legacy/` for maintenance files.
    - Moved documentation to `docs/`.
- **Outcome:** Clean project root.

**4. Memory Module (Hippocampus)**
- **Objective:** Give Ralph persistent state.
- **Action:** Created `ralph_core/memory.py`.
- **Outcome:** Ralph now has a `context.json` (Short-term) and a `memory/` folder (Long-term markdown storage).

**5. Executor Module (Motor Cortex)**
- **Objective:** Enable Ralph to run the code he writes autonomously.
- **Action:** 
    - Created `ralph_core/executor.py` for shell command execution.
    - Updated `runner.py` to parse and run `<execute>` tags.
    - Updated `swarm.py` system prompts to empower models with "Physical" agency.
- **Outcome:** Ralph can now write a test, run it, see the failure, and fix it without human intervention.

**6. Web Interface (The Face of Ralph)**
- **Objective:** Provide a fast, easy-to-access UI for the agent.
- **Action:** 
    - Scaffolded a FastAPI backend and HTML/JS frontend.
    - Configured local domain `ralph.ai` via `/etc/hosts`.
    - Set up port 80 serving for clean URL access.
- **Outcome:** Ralph is now accessible at **http://ralph.ai**.

### 🛠️ Minor Fixes & Adjustments
- Removed GPT4All dependencies and cleaned up disk space.
- Fixed regex parsing in `runner.py` to correctly handle Markdown code blocks.
- Updated `hardware_polling.sh` for reliable AMD GPU monitoring.

# Project Ralph Changelog

## Wednesday, January 28, 2026

### 🚀 Ralph v2: Tri-Brid Architecture
- **Objective:** Maximize hardware efficiency (CPU/GPU/NPU) and reduce latency via speculative decoding.
- **Action:**
    - **Hardware Roles:** Chalmers (GPU/Reasoning), Wiggum (CPU/Reflex), Librarian (NPU/Memory).
    - **Speculative Decoding:** Integrated `llama-server` (llama.cpp) to run 14B models with 1.5B draft models on the Ultra 7 265K.
    - **Librarian (NPU):** Created `ralph_core/librarian_daemon.py` using OpenVINO for automated background context migration and summarization.
    - **Memory Persistence:** Initialized `/home/tristan/ralph_brain/` as a centralized external memory store (.md network).
    - **CLI Update:** Added `--v2` flag to `ralph_loop.sh`.
- **Outcome:** Sub-20ms latency on draft tokens and a massive reduction in VRAM context pressure through background NPU summaries.

### 🖥️ Native Rust GUI Dashboard
- **Objective:** Provide a high-performance, real-time visualization of the Swarm.
- **Action:**
    - Created `ralph_gui` using Rust, `egui`, and `eframe`.
    - Implemented real-time log parsing, agent flow visualization, and metrics tracking.
    - Fixed compatibility issues with `egui 0.31` and implemented structured JSON metrics/plan capture.
- **Outcome:** A native, dark-themed dashboard that monitors Ralph's internal state without the overhead of a browser.

## Tuesday, January 13, 2026

### 🌙 Dream Mode
- **Objective:** Allow Ralph to practice and learn while idle.
- **Action:**
    - Created `ralph_core/dreamer.py`: Generates synthetic coding challenges.
    - Updated `ralph_daemon.py`: Triggers a dream after 60 seconds of inactivity.
- **Outcome:** Ralph is never truly idle; he fills his downtime by solving random algorithmic problems to populate his long-term memory.

### 🧬 Meta-Ralph (Recursive Self-Improvement)
- **Objective:** Enable Ralph to upgrade his own codebase.
- **Action:**
    - Created `meta_ralph.sh`: A dedicated entry point for introspection tasks.
    - Updated `ralph_core/tools.py`: Added automatic `.bak` creation when modifying Python files.
- **Outcome:** Running `./meta_ralph.sh` triggers a safe, sandboxed self-refactoring session.

### 🎙️ Voice Interface
- **Objective:** Talk to Ralph using local TTS/STT.
- **Action:**
    - Created `setup_voice.sh`: Automates installation of Piper and Whisper.
    - Created `ralph_core/voice.py`: Manages audio I/O.
    - Created `ralph_voice.py`: Run this to start the voice loop.
- **Outcome:** "Status report" triggers a spoken summary. "Hey Ralph, [Task]" queues a new job.

### 🖥️ Real-Time Progress Visualization
- **Objective:** See what Ralph is thinking and building in real-time.
- **Action:**
    - Updated `ralph_ui/backend/main.py`: Added `/api/plan` endpoint.
    - Updated `ralph_ui/frontend/index.html`: Added "Mission Plan" and "Live Code Preview" panels.
- **Outcome:** The UI now visualizes the task graph and streams the code generation live as it happens.

### ⚖️ Bayesian Task Prioritization
- **Objective:** Work on high-impact, high-probability tasks first.
- **Action:**
    - Created `ralph_core/agents/estimator/`: Predicts Value (1-10) and Success Prob (0-1).
    - Updated `ralph_core/triggers/queue_manager.py`: Sorts queue by `Score = Value * Probability`.
- **Outcome:** Ralph ignores low-value distractions and focuses on quick wins or critical features.

### 🗣️ Natural Language Triggers (The Listener)
- **Objective:** Queue tasks via chat apps or email.
- **Action:**
    - Created `ralph_core/triggers/queue_manager.py`: Persistent task queue (`RALPH_QUEUE.json`).
    - Created `ralph_core/triggers/discord_listener.py`: Simple Discord bot integration.
    - Created `ralph_daemon.py`: Background process that consumes the queue and runs `ralph_loop.sh`.
- **Outcome:** "Hey Ralph" messages now automatically become queued jobs.

### 🐝 Collaborative Swarm Tasks
- **Objective:** Enable parallel execution of independent tasks.
- **Action:**
    - Created `ralph_core/swarm_dispatcher.py`: Spawns multiple `runner.py` processes.
    - Updated `ralph_core/runner.py`: Supports `RALPH_INSTANCE_ID` for isolated logging.
    - Updated `ralph_core/tools.py`: Added `dispatch_swarm`.
- **Outcome:** The Orchestrator can now say "Deploy 3 workers" and they will execute tasks simultaneously.

### 🩺 Self-Healing Code (The Debugger)
- **Objective:** Automatically detect runtime errors and propose fixes.
- **Action:**
    - Created `ralph_core/agents/debugger/`: Analyzes stack traces and failed code.
    - Updated `ralph_core/runner.py`: Detects `Command: FAIL` in the previous iteration and triggers a diagnosis.
- **Outcome:** If Ralph's code crashes, the next iteration starts with a "Diagnosis Report" explaining exactly how to fix it.

### 👁️ Multi-Modal Input (Vision)
- **Objective:** Give Ralph "eyes" to see UI mockups and screenshots.
- **Action:**
    - Integrated `llava` (Large Language-and-Vision Assistant) via Ollama.
    - Created `ralph_core/vision.py`: Handles base64 image encoding and API calls.
    - Updated `ralph_core/tools.py`: Added `analyze_image` tool.
- **Outcome:** Ralph can now look at an image path provided in the prompt and describe it or build code from it.

### 🌐 Web Search & Documentation Access
- **Objective:** Allow Ralph to learn from external sources when stuck.
- **Action:**
    - Created `ralph_core/web.py`: A dependency-light scraping module (DuckDuckGo HTML).
    - Updated `ralph_core/tools.py`: Added `web_search` and `read_url`.
- **Outcome:** Ralph can now query the web for error messages or documentation.

### 🐙 Git Integration
- **Objective:** Enable automatic version control and experimental branching.
- **Action:**
    - Created `ralph_core/git_manager.py`: Handles git operations and uses Orchestrator for commit messages.
    - Updated `ralph_core/runner.py`: Triggers auto-commit upon successful verification.
    - Updated `ralph_core/tools.py`: Added `git_commit`, `git_branch`, and `git_revert` tools.
- **Outcome:** Ralph now commits his own code after verifying it works, providing a safety net for rollbacks.

### 🛡️ Sandboxed Execution Environment
- **Objective:** Protect the host system from accidental damage during autonomous loops.
- **Action:**
    - Created `sandbox/Dockerfile` & `docker-compose.yml`: A containerized Python environment.
    - Updated `ralph_loop.sh`: Added `--sandbox` flag to auto-start Docker.
    - Updated `ralph_core/executor.py`: Routes commands through `docker exec` when `RALPH_SANDBOX=1`.
- **Outcome:** Ralph can now safely execute risky code (like `rm -rf`) inside a disposable container.

### 🌍 Multi-Project Memory (Global Knowledge)
- **Objective:** Share "lessons learned" across different projects.
- **Action:**
    - Created `~/.ralph/global_memory/`: A centralized knowledge store.
    - Updated `ralph_core/memory.py`: Lessons are now saved globally.
    - Updated `ralph_core/vector_db.py`: Implemented a hybrid Local/Global vector search.
- **Outcome:** If Ralph learns a generic trick (e.g., regex optimization) in Project A, he can now recall and apply it in Project B.

### 💾 Conversation Compression (The Compressor)
- **Objective:** Prevent context overflow on long-running tasks.
- **Action:**
    - Created `ralph_core/compressor.py`: Uses the Orchestrator to summarize `RALPH_PROGRESS.md`.
    - Updated `ralph_core/runner.py`: Triggers compression every 5 iterations.
    - Checkpoints are saved to `memory/checkpoints/` and the progress file is truncated.
- **Outcome:** Ralph can theoretically run indefinitely without "forgetting" the core mission or crashing the context window.

### 🧠 Vector Memory Search
- **Objective:** Enable semantic recall of past tasks and lessons.
- **Action:**
    - Integrated `nomic-embed-text` via Ollama.
    - Created `ralph_core/vector_db.py`: A lightweight, numpy-based vector store.
    - Updated `ralph_core/memory.py`: Automatically indexes lessons and manual memories.
    - Updated `ralph_core/runner.py`: Injects relevant past experiences into the Orchestrator's context.
- **Outcome:** Ralph can now "remember" how he solved similar problems in the past, even if the keywords don't match exactly.

### 🧩 Task Decomposition Engine (The Planner)
- **Objective:** Break complex goals into manageable subtasks.
- **Action:**
    - Created `ralph_core/planning.py`: Manages `RALPH_PLAN.json` state.
    - Updated `ralph_core/agents/orchestrator/`: Added `decompose()` capability.
    - Updated `ralph_core/runner.py`: Now checks for plans, executes subtasks sequentially, and only signals completion when the full graph is resolved.
- **Outcome:** Ralph can now tackle multi-step complex projects without getting overwhelmed by the full context.

### 📊 Cost/Performance Metrics Dashboard
- **Objective:** Track system efficiency and model usage.
- **Action:**
    - Created `ralph_core/metrics.py`: A persistent JSONL tracker.
    - Updated `ralph_core/agents/common/llm.py`: Captures token counts and duration from Ollama.
    - Updated `ralph_ui/`: Added `/api/metrics` and a live dashboard to the frontend.
- **Outcome:** Real-time visibility into token usage, task duration, and system status.

### 🧠 Self-Improvement Loop (The Reflector)
- **Objective:** Allow Ralph to learn from his own mistakes over time.
- **Action:**
    - Created `ralph_core/agents/reflector/`: An agent that analyzes `RALPH_PROGRESS.md`.
    - Updated `ralph_core/memory.py`: Added `lessons.json` persistence.
    - Updated `ralph_core/runner.py`: Injects "Lessons Learned" into the context of every new iteration.
- **Outcome:** Ralph now has a persistent "Lesson Database" that grows with every failure.

### 🏗️ Architectural Refactor: Agent Specialization
- **Objective:** Organize the codebase to reflect the specialized roles of the Swarm.
- **Action:**
    - Refactored `ralph_core/swarm.py` into a modular `agents/` directory.
    - Created dedicated modules for **Orchestrator** (DeepSeek), **Engineer** (Qwen), and **Designer** (Mistral).
    - Established a Facade pattern in `swarm.py` to maintain compatibility with `runner.py`.
- **Outcome:** Clean separation of concerns. Agent logic, prompts, and behaviors can now be evolved independently.

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

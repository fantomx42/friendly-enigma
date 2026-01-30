# Technology Stack

## Core Languages
- **Python (3.8+)**: Primary language for agent logic, message bus, and swarm orchestration.
- **Rust**: Used for the high-performance native GUI dashboard (`ralph_gui`) via the `egui` framework.
- **JavaScript/HTML/CSS**: Powers the accessible web interface (`ralph_ui`).

## LLM Orchestration & Inference
- **Ollama**: Local inference engine for running all swarm models.
- **Hierarchical Model Swarm**:
    - **Strategic (Tier 1)**: `phi3:mini` (Translator), `deepseek-r1:14b` (Orchestrator).
    - **Management (Tier 2)**: `qwen2.5-coder:14b` (Engineer), `mistral-nemo:12b` (Designer).
    - **Specialists (Tier 3/ASICs)**: `tinyllama`, `deepseek-coder`, `qwen2.5-coder:1.5b`.
- **Embeddings**: `nomic-embed-text` via Ollama for semantic search and vector memory.

## Memory & Data Persistence
- **ChromaDB**: Vector database for long-term semantic memory and context retrieval.
- **Wheeler Memory (Experimental)**: Spatial dynamics-based memory system using cellular automata.
- **JSONL**: Used for high-frequency metrics, execution tracking, and agent communication logs.

## Backend & Communication
- **FastAPI**: Backend framework for the web UI and external API endpoints.
- **WebSockets**: Real-time bidirectional communication between the swarm and the UI.
- **Message Bus (V2)**: Internal async I/O based bus with circuit breakers and high-priority diagnostic routing for the Wiggum Protocol.

## Infrastructure & Hardware Acceleration
- **Docker**: Provides isolated sandboxing for tool execution and code running.
- **OpenVINO / OpenVINO-GenAI**: Leverages Intel NPU for background memory management (The Librarian).
- **ROCm**: AMD GPU compute stack support (optimized for RX 9070 XT).

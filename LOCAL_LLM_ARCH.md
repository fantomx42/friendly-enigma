# System Architecture: Autonomous Local AI (Ralph/Wheeler)

> **Design Goal**: A fully offline, token-free autonomous agent system leveraging high-end consumer hardware (AMD RX 9070 XT, 64GB RAM) and a custom "Wheeler" memory architecture.

## 1. Requirements Clarification

### Functional Requirements
- **Core Function**: Autonomous task execution (Coding, Research, System Management) without cloud dependencies.
- **Memory Integration**: Deep integration of "Wheeler Memory" (Spatial/Attractor dynamics) and Vector Search for long-term context.
- **Cost Efficiency**: Zero reliance on paid APIs (OpenAI/Anthropic).
- **Interface**: Unified CLI and Web Dashboard.

### Non-Functional Requirements
- **Privacy**: 100% Local processing. Data never leaves the machine.
- **Latency**: Interactive speeds for chat (<50ms TTFT), reasonable batch processing for complex tasks.
- **Hardware Utilization**: Maximize usage of AMD GPU (via ROCm) and NPU (if available/supported).
- **Extensibility**: Modular agent system (Swarm architecture).

## 2. Capacity Estimation (Hardware Constraints)

### Hardware Profile (User: Tristan)
- **GPU**: AMD Radeon RX 9070 XT (High VRAM bandwidth, ROCm support).
- **RAM**: 64GB DDR5 (Allows loading large quantized models, e.g., Llama-3-70B-Q4 or multiple smaller experts).
- **CPU**: Intel Core Ultra 7 265K (High single-core performance for orchestration).

### Throughput Targets
- **Inference Speed**: Aim for >50 tokens/s for 7B models, >10 tokens/s for 30B+ models.
- **Context Length**: Support for 32k-128k context windows (model dependent).
- **Memory Overhead**:
    - *System*: ~4GB
    - *Vector DB*: ~2GB (Scales with memories)
    - *Wheeler Dynamics*: ~1GB
    - *Models*: ~24GB - 48GB (Reserved for VRAM/RAM offload).

## 3. High-Level Architecture

```
                    ┌────────────────────────┐
                    │     User Interface     │
                    │   (CLI / Web / Voice)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Orchestration Layer  │
                    │ (Ralph Loop / FastAPI) │
                    └──────┬────┬────┬───────┘
                           │    │    │
          ┌────────────────▼──┐ │ ┌──▼───────────────────┐
          │  Inference Engine │ │ │    Memory System     │
          │  (Ollama / ROCm)  │ │ │ (The Hippocampus)    │
          └───────────────────┘ │ └──────────┬───────────┘
                                │            │
                                │   ┌────────▼─────────┐
                                │   │  Wheeler Bridge  │
                                │   │ (Spatial/Dynamic)│
                                │   └────────┬─────────┘
                                │            │
                      ┌─────────▼────────────▼────────┐
                      │      Knowledge Storage        │
                      │ (Vector DB + JSON + Markdown) │
                      └───────────────────────────────┘
```

## 4. Component Design

### A. Inference Engine (The "Brain")
- **Technology**: **Ollama** (Server mode).
- **Backend**: **Llama.cpp** with **ROCm** acceleration for AMD GPUs.
- **Model Strategy**:
    - *Planner*: DeepSeek-R1 (High reasoning, slower).
    - *Worker*: Llama-3 / Mistral (Fast, reliable).
    - *Specialist*: CodeLlama / Phil-3 (Task specific).

### B. Memory System (The "Soul")
This is the custom differentiator. It uses a dual-pathway approach:
1.  **Vector Memory (Semantic)**:
    -   *Tech*: Local embedding model (`nomic-embed-text`) + JSON/Numpy store (`vector_db.py`).
    -   *Role*: Retrieval of exact facts and similar documents.
2.  **Wheeler Memory (Episodic/Spatial)**:
    -   *Tech*: Custom Python Module (`ralph_core/wheeler.py`).
    -   *Mechanism*: 2D Spatial Attractor Dynamics. Inputs are "encoded" into a spatial map; recall happens by settling into "basins of attraction" (stable states).
    -   *Role*: Associative memory, "dreaming" (consolidation during idle time), and habit formation.

### C. Orchestration Layer (Ralph)
-   **Loop**: `ralph_loop.sh` triggers the cycle.
-   **Agent**: Python scripts (`ralph_core/agents/`) that manage the prompt construction, tool execution, and result verification.
-   **Tool Use**: Direct shell execution, file manipulation, and web browsing (via local headless browser).

## 5. Data Flow

### The "Recall-Act-Learn" Cycle

1.  **Input**: User provides a task (e.g., "Refactor memory.py").
2.  **Recall Phase**:
    -   Orchestrator queries **Vector DB**: "Find code related to memory.py".
    -   Orchestrator queries **Wheeler**: "What happened last time we touched memory?" (Associative recall).
    -   *Context Fusion*: Retrieved text is prepended to the system prompt.
3.  **Inference Phase**:
    -   Prompt sent to Ollama (Localhost:11434).
    -   Model generates plan/code.
4.  **Execution Phase**:
    -   Orchestrator executes shell commands/python scripts.
    -   Captures `stdout`/`stderr`.
5.  **Learning Phase**:
    -   **Short-term**: Execution result appended to chat context.
    -   **Long-term**:
        -   Result is embedded and stored in **Vector DB**.
        -   Result is fed into **Wheeler** dynamics to reinforce the "pathway" used.

## 6. Scaling Strategy (Local)

Since vertical scaling (buying more hardware) is expensive, we use **Optimization**:
-   **Quantization**: Use 4-bit (Q4_K_M) or 5-bit models to fit larger parameter counts in VRAM.
-   **Context Caching**: Cache the system prompt and static context to speed up subsequent turns.
-   **Speculative Decoding**: Use a small model (e.g., TinyLlama) to draft tokens for the large model to verify (if VRAM permits).

## 7. Technology Stack

-   **Runtime**: Python 3.10+
-   **LLM Server**: Ollama (Linux/ROCm build)
-   **Vector Search**: NumPy (Cosine Similarity) + Local Embeddings (No cloud vector DBs like Pinecone).
-   **Application**: FastAPI (Backend) + React (Frontend - Optional).
-   **Shell**: Bash (for low-level control).

## 8. Deployment Strategy

1.  **Environment**: Containerized (Docker) or Virtual Environment (venv) on CachyOS.
2.  **Service Management**: `systemd` user units to manage:
    -   `ollama.service` (Inference)
    -   `ralph_daemon.service` (Background memory consolidation)
    -   `ralph_ui.service` (Interface)

## 9. Trade-offs

| Decision | Pro | Con |
| :--- | :--- | :--- |
| **Local Hosting** | 100% Privacy, No subscription costs. | High hardware cost, maintenance burden. |
| **Custom Memory** | Tailored to specific needs, "Organic" behavior. | Complexity, debugging "hallucinations" in memory dynamics. |
| **ROCm (AMD)** | High perf/dollar (vs Nvidia). | Software ecosystem (PyTorch/Bitsandbytes) can be finicky compared to CUDA. |


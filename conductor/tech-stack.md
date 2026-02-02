# Technology Stack

## Backend & Core
- **Python 3.8+:** Primary language for agent logic, memory management, and orchestration.
- **FastAPI:** Web framework for the UI backend and local API endpoints.
- **Pydantic:** Data validation and settings management.

## Frontend & GUI
- **Rust (eframe/egui):** Native desktop GUI for real-time monitoring and control.
- **JavaScript (Vanilla):** Frontend for the web-based Neural Dashboard.

## AI & Machine Learning
- **Ollama:** Primary engine for local LLM inference.
- **PyTorch (ROCm):** Used for training the Wheeler AI autoencoder on AMD hardware.
- **OpenVINO:** Optimized execution for NPU and Intel CPU components.
- **ROCm:** Native acceleration for AMD Radeon GPUs (RDNA4/9070 XT).

## Memory & Data
- **ChromaDB:** Vector database for associative memory and retrieval-augmented generation (RAG).
- **Sentence-Transformers:** Local embedding generation for memory vectors.

## Testing & Automation
- **Pytest:** Unit and integration testing for Python components.
- **Playwright:** End-to-end testing for the Web UI.

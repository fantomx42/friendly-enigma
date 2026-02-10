# Tech Stack: TWAI

## Core Orchestration & Backend: Rust (Axum)
- **Rationale:** Rust provides the memory safety and high-concurrency performance required for real-time system mapping and orchestration of local LLM tasks.
- **Framework:** Axum for a fast, modular, and type-safe web backend.

## Frontend Dashboard: Rust (Leptos/Yew)
- **Rationale:** An "All-Rust" stack ensures high performance for the visual dashboard and seamless type sharing between the frontend and backend.
- **Visuals:** Integration with Mermaid.js or D3.js for interactive architectural maps.

## Local LLM Orchestration: Ollama (via ROCm)
- **Rationale:** Ollama provides the widest variety of models and the easiest management layer, while natively supporting AMD GPU acceleration through ROCm.
- **Interface:** REST API integration for model swapping and inference.

## Multi-Model Persistence: SurrealDB
- **Rationale:** A Rust-native, multi-model database that handles Documents, Graphs (for architectural relationships), and Vectors (for codebase search) in a single high-performance engine.

## Infrastructure
- **GPU Acceleration:** AMD ROCm (optimized for RX 9070 XT).
- **Environment:** Local-only, 1000ffline-capable.

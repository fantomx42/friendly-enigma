# Specification: Core Orchestration & Mission Control MVP

## Overview
TWAI is a local, high-performance alternative to Claude Code, optimized for AMD hardware (ROCm) and designed for visual systems thinkers. This MVP focuses on establishing the core communication loop between the user, the filesystem, and the local LLM (Ollama).

## Functional Requirements
- **Ollama Chat Interface:** Real-time chat with local LLMs via Ollama's API.
- **Live Codebase Map:** Interactive visualization of the project's directory structure.
- **File System Interaction:** Read and write access to the local codebase.
- **System Telemetry:** Display basic hardware stats (CPU/GPU usage) if feasible.

## Technical Architecture
- **Backend:** Rust (Axum) handling LLM orchestration, filesystem operations, and telemetry gathering.
- **Frontend:** Rust (Leptos or Yew) compiled to WebAssembly for a high-performance, single-page dashboard.
- **Persistence:** SurrealDB for graph-based relationship tracking and metadata storage.
- **LLM Engine:** Ollama with ROCm acceleration.

## Success Criteria
- User can chat with a local model and receive valid responses.
- User can see a live tree or graph map of the project files.
- System is 100% offline and runs on AMD hardware.

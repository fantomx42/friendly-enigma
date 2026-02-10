# Implementation Plan: Core Orchestration & Mission Control MVP

## Phase 1: Project Scaffolding ## Phase 1: Project Scaffolding & Backend Setup Backend Setup [checkpoint: 84221be]
- [x] Task: Initialize Rust workspace and Axum backend structure e4af11b
- [x] Task: Implement basic health check and telemetry endpoints 048dd57
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Project Scaffolding & Backend Setup' (Protocol in workflow.md)

## Phase 2: Local LLM Integration (Ollama) [checkpoint: fc367bb]
- [x] Task: Implement Ollama API client in Rust 2969b05
- [x] Task: Create chat endpoint with streaming support 8940724
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Local LLM Integration (Ollama)' (Protocol in workflow.md)

## Phase 3: Filesystem ## Phase 3: Filesystem & Persistence (SurrealDB) Persistence (SurrealDB) [checkpoint: d522b69]
- [x] Task: Set up SurrealDB and implement basic project indexing bf2b8c1
- [x] Task: Implement filesystem service for reading and writing files b942b15
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Filesystem & Persistence (SurrealDB)' (Protocol in workflow.md)

## Phase 4: Frontend Dashboard (Leptos/Yew)
- [ ] Task: Scaffold Leptos/Yew frontend with a grid-based Bauhaus layout
- [ ] Task: Implement chat component and integrate with backend
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Frontend Dashboard (Leptos/Yew)' (Protocol in workflow.md)

## Phase 5: Visual Systems Mapping
- [ ] Task: Implement interactive file tree/map using D3.js or Mermaid.js
- [ ] Task: Connect visual map to the filesystem backend service
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Visual Systems Mapping' (Protocol in workflow.md)

## Phase 6: Final Integration & Polishing
- [ ] Task: Wire all components together and perform end-to-end testing
- [ ] Task: Final performance tuning for AMD hardware
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Final Integration & Polishing' (Protocol in workflow.md)

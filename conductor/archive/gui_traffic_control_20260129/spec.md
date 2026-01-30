# Track Specification: Native GUI Dashboard (ralph_gui) - Traffic & Control

## Overview
This track focuses on developing the core visual and interactive layers of the `ralph_gui` (Rust/egui) application. The goal is to provide a real-time, high-performance dashboard for monitoring agent swarm dynamics and controlling task execution.

## Functional Requirements
- **Force-Directed Agent Traffic Graph**:
    - Implement a dynamic, organic node-link graph visualization.
    - Agents (Translator, Orchestrator, Engineers, ASICs) are represented as nodes.
    - Active communication between agents is visualized as reactive, moving links.
    - Nodes should cluster and move based on communication frequency and force-directed logic.
- **Integrated Task Control Center**:
    - **Command Input**: A text area for submitting natural language objectives.
    - **Lifecycle Management**: Buttons for Start, Pause, Stop, and Emergency Flush (to clear the diagnostic bus).
    - **Agent Configuration**: Toggles for enabling/disabling specific agents and adjusting key model parameters (e.g., temperature).
    - **Sandbox Mode Toggle**: A switch to choose between standard local execution and Docker-sandboxed execution.

## Non-Functional Requirements
- **Performance**: The visualization must maintain a high frame rate (60 FPS+) even during high agent activity, leveraging Rust's performance and `egui`'s efficiency.
- **Responsiveness**: UI controls must reflect the current state of the backend swarm immediately via the message bus.
- **Aesthetic**: Modern, "alive" feel with reactive animations for the graph.

## Acceptance Criteria
- [ ] The `ralph_gui` launches and displays a force-directed graph.
- [ ] Simulated agent messages trigger visual "pulses" or links in the graph.
- [ ] Submitting a command via the Input Box correctly routes the task to the Orchestrator.
- [ ] Toggling the "Sandbox" switch updates the swarm's execution mode.
- [ ] The "Stop" button successfully sends a termination signal to all active agent processes.

## Out of Scope
- Detailed hardware metric monitoring (GPU/VRAM) is reserved for a future track.
- Persistent task history viewer/logs within the GUI (handled by CLI or basic stdout for now).

# Implementation Plan: Native GUI Dashboard (ralph_gui) - Traffic & Control

## Phase 1: Foundation and Message Bus Integration
- [x] Task: Project Structure and Egui Setup [4eab179]
    - [ ] Initialize Rust project in `ralph_gui/` if not already present
    - [ ] Configure `egui` and `eframe` dependencies in `Cargo.toml`
    - [ ] Implement a basic boilerplate window with placeholder areas for the graph and control center
- [x] Task: Message Bus Subscriber Interface [0559fec]
    - [ ] Define Rust structs for Agent Status and Communication Messages (matching Python backend)
    - [ ] Implement an async message bus listener using `tokio` or similar to receive real-time data
    - [ ] Create a thread-safe state store for the GUI to consume received messages
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation'
## Phase 2: Force-Directed Graph Visualization
- [x] Task: Graph Data Structure and Layout Logic [bd681c9]
    - [ ] Implement a `Node` and `Edge` data structure within the GUI state
    - [ ] Integrate or implement a basic force-directed layout algorithm (attraction/repulsion)
- [x] Task: Visualization Rendering [1036de2]
    - [ ] Write tests for node positioning logic
    - [ ] Implement `egui` painting commands to render nodes as tiered circles (Translator, Orchestrator, etc.)
    - [ ] Implement reactive link rendering that "pulses" when communication is detected
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Graph Visualization' (Protocol in workflow.md)

## Phase 3: Task Control Center Implementation
- [ ] Task: UI Control Widgets
    - [ ] Write unit tests for control state management
    - [ ] Implement the Command Input text area and Submit logic
    - [ ] Create buttons for Start, Pause, Stop, and Emergency Flush
    - [ ] Implement the Sandbox Toggle and Agent Configuration switches
- [ ] Task: Backend Command Routing
    - [ ] Implement the publisher side of the message bus to send commands back to the swarm
    - [ ] Verify that UI actions (e.g., clicking Stop) generate the correct message bus packets
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Control Center' (Protocol in workflow.md)

## Phase 4: Integration and Polishing
- [ ] Task: End-to-End Swarm Synchronization
    - [ ] Run the Python swarm alongside the GUI to verify real-time state synchronization
    - [ ] Implement graceful handling for backend disconnection/reconnection
- [ ] Task: UI/UX Refinement
    - [ ] Refine the force-directed parameters for a smoother "organic" feel
    - [ ] Ensure the UI remains responsive (60 FPS) during high-frequency message bursts
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration' (Protocol in workflow.md)

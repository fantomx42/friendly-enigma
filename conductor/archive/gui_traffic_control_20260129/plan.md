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
- [x] Task: Conductor - User Manual Verification 'Phase 2: Graph Visualization' [1036de2]

## Phase 3: Task Control Center Implementation
- [x] Task: UI Control Widgets [0cae05c]
    - [ ] Write unit tests for control state management
    - [ ] Implement the Command Input text area and Submit logic
    - [ ] Create buttons for Start, Pause, Stop, and Emergency Flush
    - [ ] Implement the Sandbox Toggle and Agent Configuration switches
- [x] Task: Backend Command Routing [86f8166]
    - [ ] Implement the publisher side of the message bus to send commands back to the swarm
    - [ ] Verify that UI actions (e.g., clicking Stop) generate the correct message bus packets
- [x] Task: Conductor - User Manual Verification 'Phase 3: Control Center' [86f8166]

## Phase 4: Integration and Polishing
- [x] Task: End-to-End Swarm Synchronization [3829a30]
    - [ ] Run the Python swarm alongside the GUI to verify real-time state synchronization
    - [ ] Implement graceful handling for backend disconnection/reconnection
- [x] Task: UI/UX Refinement [e0814dc]
    - [ ] Refine the force-directed parameters for a smoother "organic" feel
    - [ ] Ensure the UI remains responsive (60 FPS) during high-frequency message bursts
- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration' [e0814dc]

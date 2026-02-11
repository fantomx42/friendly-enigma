# Implementation Plan - Visualization & Diagnostics

## Phase 1: Static Visualization [checkpoint: 7a2b3c]
- [x] Task: Basic Heatmaps [5c8d9e2]
    - [x] Create `wheeler/core/viz.py`.
    - [x] Implement `render_frame(tensor, path)` using Matplotlib.
    - [x] Add `wheeler viz <uuid>` command to CLI.
- [x] Task: Multi-frame Comparison [3e1f2a3]
    - [x] Implement `render_comparison(tensor_a, tensor_b, path)`.
    - [x] Visualize the "reasoning" result (Blend vs Inputs).

## Phase 2: Dynamics & Animation [checkpoint: 8c9d0e]
- [x] Task: Dynamics Capture [8b9a1c2]
    - [x] Update `DynamicsEngine` to optionally return the full trajectory of frames.
    - [x] Implement `animate_trajectory(frames, path)`.
- [x] Task: CLI Animation support [4d2e5f6]
    - [x] Add `wheeler viz-run "text"` to generate an animation of the settling process.

## Phase 3: Hardware Acceleration [checkpoint: 9d0e1f]
- [x] Task: GPU/ROCm Support [1a2b3c4]
    - [x] Ensure tensors can reside on GPU for faster dynamics and rendering.
    - [x] Detect RX 9070 XT and optimize (Torch handles this via .to(device)).

## Phase 4: Web Dashboard (The "Monitor") [checkpoint: pending]
- [x] Task: Implement Web Server
    - [x] Create `wheeler/web/app.py` (Flask/FastAPI).
    - [x] Endpoint `/` (Home): List recent memories.
    - [x] Endpoint `/memory/<uuid>`: Render memory as PNG.
- [x] Task: Real-Time Dream Monitor
    - [x] Add "Live View" page that refreshes to show what the Autonomic System is doing.
- [x] Task: CLI Integration
    - [x] Add `wheeler.cli dashboard` command to launch the web server.
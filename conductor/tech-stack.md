# Technology Stack

## Core Engine
- **Language:** Python 3.10+
- **Acceleration:** PyTorch (ROCm) for AMD RX 9070 XT optimization.
- **Custom Kernels:** Triton / OpenCL for highly optimized cellular automata update steps.
- **Math/Data:** NumPy for CPU fallback and general tensor manipulation.

## Persistence Layer (Hybrid)
- **Metadata:** SQLite (via `aiosqlite` for async access) to store timestamps, stability scores, associations, and tags.
- **Data Blobs:** Local Filesystem (`.npy` format) for storing the raw 128x128 spatial frames (attractors).

## Visualization & Inspection
- **Static:** Matplotlib for generating high-quality heatmaps and stability graphs for reports/debugging.
- **Real-time:** Vispy (OpenGL) for high-performance, live rendering of the reaction-diffusion dynamics.

## Distribution & Tooling
- **Package Manager:** Poetry
- **Testing:** Pytest (with `hypothesis` for property-based testing of the deterministic rules).
- **Linting/Formatting:** Ruff

## Hardware Targets
- **Primary:** AMD Radeon RX 9070 XT (via ROCm).
- **Secondary:** CUDA-capable NVIDIA GPUs (via standard PyTorch).
- **Fallback:** CPU (via NumPy/PyTorch CPU).
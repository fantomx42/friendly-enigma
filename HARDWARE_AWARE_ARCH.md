# System Architecture: Hardware-Aware Auto-Configuration Subsystem

> **Design Goal**: A robust initialization subsystem that detects the host hardware (GPU, RAM, CPU) on first boot and dynamically configures the Inference Engine (Ollama/Llama.cpp) for maximum performance without user intervention.

## 1. Requirements Clarification

### Functional Requirements
-   **Hardware Detection**: Automatically identify GPU vendor (AMD/NVIDIA/Intel), VRAM amount, System RAM, and CPU instruction sets (AVX2/AVX512).
-   **Profile Selection**: Map hardware capabilities to optimal runtime configurations (Context Window, Quantization Level, GPU Layers).
-   **Dependency Check**: Verify drivers (ROCm, CUDA) and library availability.
-   **Fallback Mechanism**: Gracefully degrade to CPU-only mode if acceleration fails.

### Non-Functional Requirements
-   **Reliability**: Must not crash on obscure hardware; default to safe/slow settings.
-   **Transparency**: Log all detection decisions clearly for debugging.
-   **Speed**: Detection should take < 5 seconds.

## 2. High-Level Architecture

```
┌─────────────────────────────────────┐
│          Bootstrapper               │
│        (ralph_loop.sh)              │
└──────────────────┬──────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Hardware Probe   │ ◄──── System APIs (lspci, /proc, SMI)
         │ (hardware_scan.py)│
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │   Config Logic    │ ◄──── Hardware Profiles (JSON)
         │  (configurator.py)│
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ Runtime Environ   │ ────► Writes .env / service config
         │    (Setup)        │
         └───────────────────┘
```

## 3. Component Design

### A. Hardware Probe (`ralph_core/hardware/scan.py`)
Uses `psutil`, `subprocess` (for SMI tools), and file reading to gather metrics.

*   **GPU Detection**:
    *   Check `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA).
    *   Fallback: Parse `lspci | grep -i vga`.
    *   *Metric*: VRAM total, Compute Unit count.
*   **Memory Detection**:
    *   `psutil.virtual_memory().total`
*   **CPU Detection**:
    *   `platform.processor()`, check `/proc/cpuinfo` for AVX flags.

### B. Configuration Logic (`ralph_core/hardware/profiles.py`)
A rules engine that selects the best configuration.

*   **Rule Example (Tristan's Rig)**:
    *   *Condition*: `Vendor=AMD` AND `VRAM > 16GB` AND `ROCm=Found`.
    *   *Action*:
        *   `OLLAMA_LLM_LIBRARY="rocm"`
        *   `GPU_LAYERS="all"`
        *   `KV_CACHE_TYPE="f16"` (if VRAM allows)
*   **Rule Example (Generic Laptop)**:
    *   *Condition*: `Vendor=Intel` AND `RAM < 16GB`.
    *   *Action*:
        *   `OLLAMA_LLM_LIBRARY="cpu"`
        *   `MODEL_QUANT="q4_k_m"`
        *   `CONTEXT_WINDOW=4096`

### C. Persistent Configuration Store
Stores the result of the scan so it doesn't need to run every time (unless hardware changes).
*   File: `.ralph_hardware_config.json`
*   Format:
    ```json
    {
      "detected_at": "2026-01-30T10:00:00",
      "hardware": {
        "gpu": "AMD Radeon RX 9070 XT",
        "vram_gb": 16,
        "ram_gb": 64
      },
      "runtime_config": {
        "OLLAMA_NUM_GPU": 99,
        "OLLAMA_FLASH_ATTENTION": 1
      }
    }
    ```

## 4. Data Flow (First Boot)

1.  **Launch**: User runs `ralph_loop.sh`.
2.  **Check**: Script checks for `.ralph_hardware_config.json`.
3.  **Probe**: If missing, runs `scan.py`.
    *   *Probe* detects RX 9070 XT + 64GB RAM.
4.  **Match**: `profiles.py` matches "High-End AMD Workstation" profile.
5.  **Write**: Generates `.env` file with optimization flags.
6.  **Start**: Ollama starts with `HSA_OVERRIDE_GFX_VERSION` (if needed for RDNA3/4) and optimal thread count.

## 5. Deployment Strategy

*   **Init Script**: A bash script `init_hardware.sh` that wraps the python modules.
*   **Docker Integration**: If running in Docker, map `/dev/kfd` and `/dev/dri` automatically based on detection.

## 6. Trade-offs

| Decision | Pro | Con |
| :--- | :--- | :--- |
| **Python Probing** | Cross-platform, rich libraries (psutil). | Requires Python environment to be set up first. |
| **Static Profiles** | predictable, easy to test. | Might miss edge-case hardware combos. |
| **Env Var Config** | Standard for containers/services. | Requires restarting services to apply changes. |


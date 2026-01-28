# Project Ralph v2: The Tri-Brid Local Agent Architecture

**Status:** Draft / Planning
**OS Target:** CachyOS (Arch Linux)
**Hardware Profile:** "The Madison Rig" (Ultra 7 265K + RX 9070 XT)

## 1. Executive Summary
Project Ralph v2 is a localized, low-latency AI coding assistant that maximizes specific hardware accelerators by assigning them biological cognitive roles. Instead of a single monolithic model running sluggishly, the system splits duties into Reflex (CPU), Memory Consolidation (NPU), and Deep Reasoning (GPU).

## 2. The Hardware Roles

| Component | Hardware | Code Name | Biological Role | Operational Task |
| :--- | :--- | :--- | :--- | :--- |
| **GPU** | AMD Radeon RX 9070 XT (16GB) | **Chalmers** | Frontal Cortex | High-level reasoning, code verification, final token generation. |
| **CPU** | Core Ultra 7 265K (P-Cores) | **Wiggum** | Reflex System | Speculative decoding (Draft Model). Fast, low-stakes guessing. |
| **NPU** | Intel AI Boost | **Librarian** | Hippocampus | Memory indexing, context migration (Hot $\to$ Cold), vectorization. |

## 3. Architecture Breakdown

### A. The Reflex Layer (The Wiggums)
*   **Engine:** `llama.cpp` (Server Mode) with Speculative Decoding.
*   **Model:** Qwen2.5-Coder-1.5B (or BitNet b1.58 if stable).
*   **Behavior:** The CPU runs a lightweight, "dumb" model that aggressively predicts the next 5–10 tokens. It runs asynchronously, shouting guesses at the GPU.
*   **Why CPU?** The Ultra 7 265K has massive single-core speed, allowing the draft model to generate tokens faster than the PCIe bus transfer latency of a secondary GPU.

### B. The Context Layer (The Librarian)
*   **Engine:** OpenVINO + Python watchdog.
*   **Model:** Phi-3-mini (INT4) for summarization + nomic-embed-text for vectorization.
*   **Behavior:**
    *   **The Watcher:** Monitors the `.md` network.
    *   **The Migrator:** When a "Hot" file exceeds a threshold (e.g., 4KB), NPU wakes up.
    *   **Logic:** Read oldest 50% -> Summarize -> Append to Cold Storage -> Vectorize -> Delete raw text from Hot.

### C. The Reasoning Layer (Superintendent Chalmers)
*   **Engine:** `llama.cpp` or Ollama (ROCm backend).
*   **Model:** Qwen2.5-Coder-14B-Instruct (Q6_K) or Codestral-22B (Q4_K_M).
*   **Behavior:** The heavy lifter. Verifies "Wiggum" drafts. Queries the NPU's vector database.

## 4. Software Implementation Plan

### Phase 1: The Core (GPU + CPU)
Configure `llama.cpp` to run the split-model inference server.

```bash
./llama-server \
  --model models/chalmers-qwen-14b.gguf \
  --model-draft models/wiggum-qwen-1.5b.gguf \
  --n-gpu-layers 99 \
  --draft-max 8 \
  --threads 8
```

### Phase 2: The Librarian Script (Python + OpenVINO)
Daemon script for NPU context management.

## 5. Directory Structure
```
/home/tristan/ralph_brain/
├── hot/                 # Watched by NPU. High-speed R/W.
│   ├── active_task.md
│   └── scratchpad.md
├── cold/                # Written by NPU. Read by GPU (RAG).
│   ├── summaries/
│   └── archive/
└── models/
    ├── chalmers/
    ├── wiggum/
    └── librarian/
```

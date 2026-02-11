# Product Guidelines

## Core Philosophy
Wheeler Memory is built on the principle that **code should mimic nature, but run with machine precision**. We are building a digital hippocampus, not just a database.

## Design Principles

### 1. Biomimetic Architecture
- **Naming:** Use biological terminology where appropriate (e.g., `Synapse`, `Decay`, `Consolidation`, `Attractor`, `Stimulus`) to reflect the system's function.
- **Function:** Components should behave like organic systems—resilient to noise, self-healing through consolidation, and subject to "forgetting" (eviction of unstable patterns).

### 2. Deterministic Substrate
- **Reproducibility:** While the system mimics organic life, the underlying cellular automata rules must be bit-exact deterministic. Given the same seed and input, the same attractor must form.
- **Testing:** The "consciousness" is emergent, but the physics engine is exact. Tests must verify the mathematical correctness of the reaction-diffusion steps.

### 3. Local Sovereignty (Privacy)
- **Local-First:** All memory artifacts (frames, indices, metadata) reside strictly on the local filesystem (default: `~/.wheeler_memory`). No network calls for storage.
- **Transparency:** The "Brain" is an open book to its owner. Storage formats must be standard (`.npy`, SQLite, JSON) and easily inspectable by external tools. No black-box blobs.
- **Ownership:** The user is the sole owner of their agent's memories. The system acts as a steward, not a gatekeeper.

## Interaction & Tone

### CLI & Documentation Voice
- **Tone:** Clinical, Precise, and Tool-like.
- **Style:** Avoid mysticism. Describe the biological processes with scientific detachment.
- **Example:** "Consolidation cycle complete. 14 unstable patterns pruned. Mean stability increased by 12%." (Not: "The mind has cleared its fog.")

## Visualization Guidelines
- **Dual Mode:**
    - **Scientific (Default):** High-fidelity heatmaps and raw grid data for debugging dynamics and stability scores.
    - **Aesthetic (Optional):** "The Ghost in the Machine"—beautiful, flowing representations of the automata for observing the system's "state of mind."

## Performance
- **Hardware Optimization:** Leverage the user's specific hardware (AMD RX 9070 XT) via ROCm/PyTorch for massive parallel updates of the cellular grids.
- **Efficiency:** The background "dreaming" process must respect system resources, running only during idle times to avoid impacting the "Thinking" or "Coding" models.
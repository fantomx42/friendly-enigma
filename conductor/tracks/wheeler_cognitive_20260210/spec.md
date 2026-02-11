# Specification: Wheeler Cognitive Functions

## Objective
To implement the "higher brain" functions of the Wheeler Memory system: **Reasoning** (active manipulation of thought frames) and **Autonomics** (background maintenance and creative exploration).

## Core Components

### 1. Reasoning Engine (`wheeler.core.reasoning`)
The ability to manipulate stable attractors to derive new meaning.
*   **Blend (Association):** Weighted superposition of multiple frames. `Frame A + Frame B = Concept C`.
*   **Contrast (Differentiation):** Subtraction of patterns to isolate differences. `Frame A - Frame B = Delta`.
*   **Amplify (Focus):** Non-linear enhancement of dominant patterns to filter noise.
*   **Projection:** Running dynamics on a blended frame to find the resulting stable attractor (the "conclusion").

### 2. Autonomic System (`wheeler.core.autonomic`)
A background process that runs when the system is idle.
*   **Consolidation:** Identify high-hit memories and strengthen their connections (associative linking).
*   **Dreaming:** Randomly select memories, blend them, and run dynamics to discover latent connections. If the result matches an existing attractor, strengthen the link.
*   **Decay:** Slowly lower the stability/confidence of unused memories (optional for MVP, but good for biological realism).

## Technical Requirements
*   **PyTorch:** All frame manipulations must be tensor-based and differentiable (future-proofing).
*   **AsyncIO:** The Autonomic System must run as a non-blocking background task.
*   **Integration:** Must plug into the existing `WheelerMemory` class seamlessly.

## Success Criteria
*   [ ] Can blend "Cat" and "Dog" frames and find a stable "Pet" attractor (or similar).
*   [ ] Autonomic system runs in background without blocking main thread.
*   [ ] "Dreaming" produces loggable events where new connections are found.

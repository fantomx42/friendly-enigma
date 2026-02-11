# Track Specification: Core Engine & Persistence MVP

## Objective
Build the foundational `wheeler_memory` library, enabling the storage and retrieval of text-based memories using cellular automata dynamics and a hybrid SQLite/Filesystem persistence layer.

## Core Components
1.  **Dynamics Engine (`wheeler.core.engine`)**
    -   Input: 128x128 float32 grid.
    -   Process: Reaction-Diffusion cellular automata (PyTorch optimized).
    -   Output: Stable 128x128 "Attractor" grid.
    -   Must be deterministic.

2.  **Persistence Layer (`wheeler.core.storage`)**
    -   **SQLite:** Stores metadata (`id`, `timestamp`, `stability_score`, `text_hash`).
    -   **Filesystem:** Stores raw `.npy` arrays (filenames derived from ID).
    -   API: `save_memory(frame, meta)`, `load_memory(id)`, `find_similar(frame)`.

3.  **Public API (`wheeler.api`)**
    -   `WheelerMemory` class.
    -   `store(text: str) -> MemoryID`: Encodes text -> runs dynamics -> saves result.
    -   `recall(query: str, top_k: int) -> List[Memory]`: Encodes query -> runs dynamics -> finds nearest neighbors.

4.  **CLI (`wheeler.cli`)**
    -   Simple wrapper around the API for testing.
    -   `wheeler store "text"`
    -   `wheeler recall "query"`

## Constraints
-   **Strict Separation:** The engine must not depend on the storage, and vice versa.
-   **Performance:** Tensor operations must be vectorized (PyTorch).
-   **Testing:** 80% coverage required. Dynamics must be tested for determinism.

# Implementation Plan - Core Engine & Persistence MVP

## Phase 1: Project Skeleton & Configuration
- [x] Task: Initialize Poetry project structure [ef5aff0]
    - [x] Initialize `pyproject.toml` with dependencies (pytorch, numpy, aiosqlite, click).
    - [x] Create directory structure (`wheeler/core`, `wheeler/cli`, `tests`).
- [x] Task: Configure Code Quality Tools [3197fa5]
    - [x] Configure `ruff` and `mypy`.
    - [x] Create `pytest.ini`.

## Phase 2: Dynamics Engine (The "Physics") [checkpoint: 9a31742]
- [x] Task: Implement Text Codec [24bfe98]
    - [x] Write Tests: Ensure string <-> grid conversion is deterministic.
    - [x] Implement `TextCodec` (128x128 grid encoding).
- [x] Task: Implement Cellular Automata Engine [d8d9cf7]
    - [x] Write Tests: Verify determinism (Same Input + Same Seed = Same Output).
    - [x] Implement `DynamicsEngine` using PyTorch.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Dynamics Engine' (Protocol in workflow.md)
    *Note: Verification pending user confirmation.*

## Phase 3: Persistence Layer (The "Hard Drive") [checkpoint: 3e723e5]
- [x] Task: Implement SQLite Metadata Store [8c7a2b1]
    - [x] Write Tests: CRUD operations for memory metadata.
    - [x] Implement `MetadataStore` (aiosqlite).
- [x] Task: Implement Blob Store [3f9e2d4]
    - [x] Write Tests: Save/Load `.npy` files.
    - [x] Implement `BlobStore` (filesystem operations).
- [x] Task: Implement Hybrid Storage Controller [d5e6f7a]
    - [x] Write Tests: Integration test ensuring metadata and blobs stay in sync.
    - [x] Implement `StorageController` combining Metadata and Blob stores.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Persistence Layer' (Protocol in workflow.md)

## Phase 4: Public API & CLI [checkpoint: 0adebc0]
- [x] Task: Implement WheelerMemory Main Class [e9b8f1a]
    - [x] Write Tests: End-to-end `store` and `recall` flow.
    - [x] Implement `WheelerMemory` class (wiring Engine + Storage).
- [x] Task: Implement CLI [4d2c8b9]
    - [x] Write Tests: Verify CLI commands invoke API correctly.
    - [x] Implement Click-based CLI (`store`, `recall`).
- [x] Task: Conductor - User Manual Verification 'Phase 4: Public API & CLI' (Protocol in workflow.md)

## Phase 5: Stability Scoring & Ralph Integration [checkpoint: beeaf43]
- [x] Task: Implement Advanced Stability Scoring [3c8b9a1]
    - [x] Update `WheelerMemory.recall` to calculate stability (hit_count, persistence, survival).
    - [x] Add tests for stability weighting in recall.
- [x] Task: Integrate with Ralph Runtime [e5f6a7b]
    - [x] Create `ai_tech_stack/ralph_core/wheeler_bridge.py` using the new library.
    - [x] Update `wheeler_recall.py` and `wheeler_store.py` to use the bridge.
- [x] Task: Verify Integration in `ralph_simple.py` [9d1e2f3]
    - [x] Run a test loop of Ralph using the new Wheeler Memory.

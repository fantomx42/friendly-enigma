# Implementation Plan - Core Engine & Persistence MVP

## Phase 1: Project Skeleton & Configuration
- [x] Task: Initialize Poetry project structure [ef5aff0]
    - [x] Initialize `pyproject.toml` with dependencies (pytorch, numpy, aiosqlite, click).
    - [x] Create directory structure (`wheeler/core`, `wheeler/cli`, `tests`).
- [x] Task: Configure Code Quality Tools [3197fa5]
    - [x] Configure `ruff` and `mypy`.
    - [x] Create `pytest.ini`.

## Phase 2: Dynamics Engine (The "Physics")
- [x] Task: Implement Text Codec [24bfe98]
    - [x] Write Tests: Ensure string <-> grid conversion is deterministic.
    - [x] Implement `TextCodec` (128x128 grid encoding).
- [x] Task: Implement Cellular Automata Engine [d8d9cf7]
    - [x] Write Tests: Verify determinism (Same Input + Same Seed = Same Output).
    - [x] Implement `DynamicsEngine` using PyTorch.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dynamics Engine' (Protocol in workflow.md)

## Phase 3: Persistence Layer (The "Hard Drive")
- [ ] Task: Implement SQLite Metadata Store
    - [ ] Write Tests: CRUD operations for memory metadata.
    - [ ] Implement `MetadataStore` (aiosqlite).
- [ ] Task: Implement Blob Store
    - [ ] Write Tests: Save/Load `.npy` files.
    - [ ] Implement `BlobStore` (filesystem operations).
- [ ] Task: Implement Hybrid Storage Controller
    - [ ] Write Tests: Integration test ensuring metadata and blobs stay in sync.
    - [ ] Implement `StorageController` combining Metadata and Blob stores.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Persistence Layer' (Protocol in workflow.md)

## Phase 4: Public API & CLI
- [ ] Task: Implement WheelerMemory Main Class
    - [ ] Write Tests: End-to-end `store` and `recall` flow.
    - [ ] Implement `WheelerMemory` class (wiring Engine + Storage).
- [ ] Task: Implement CLI
    - [ ] Write Tests: Verify CLI commands invoke API correctly.
    - [ ] Implement Click-based CLI (`store`, `recall`).
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Public API & CLI' (Protocol in workflow.md)

# Implementation Plan - Wheeler Cognitive Functions

## Phase 1: Foundation Upgrades (The "Smarter" Codec)
- [ ] Task: Implement Position-Invariant TextCodec
    - [ ] Update `TextCodec` to use a position-independent encoding (e.g., distributed hash summation or cyclic binding).
    - [ ] Goal: "permanent memory" should have high similarity to "This is my first permanent memory".
- [ ] Task: Verify Improved Similarity Scores
    - [ ] Write a script to compare similarity of "A B" vs "B A" vs "A ... B".

## Phase 2: The Reasoning Engine
- [x] Task: Implement Tensor Operations [5c8d9e2]
    - [x] Create `wheeler/core/reasoning.py`.
    - [x] Implement `blend(frames, weights)`.
    - [x] Implement `contrast(frame_a, frame_b)`.
    - [x] Implement `amplify(frame, strength)`.
    - [x] Write Tests: Verify mathematical correctness of tensor ops.
- [x] Task: Integrate Reasoning with Memory [3e1f2a3]
    - [x] Update `WheelerMemory` to include `ReasoningEngine`.
    - [x] Add `infer(text_a, text_b)` method to `WheelerMemory` (blend -> dynamics -> recall).

## Phase 3: The Autonomic System
- [x] Task: Implement Background Loop [8b9a1c2]
    - [x] Create `wheeler/core/autonomic.py`.
    - [x] Implement `AutonomicSystem` class with `start()`, `stop()`, and `_loop()`.
    - [x] Write Tests: Ensure it runs async without blocking.
- [x] Task: Implement Consolidation & Dreaming [4d2e5f6]
    - [x] Implement `_consolidate()`: Link frequent memories.
    - [x] Implement `_dream()`: Random blends + stability check.
    - [x] Write Tests: Mock storage to verify "dreams" occur.

## Phase 4: CLI & Verification
- [x] Task: Update CLI [1a2b3c4]
    - [x] Add `wheeler.cli dream` command.
    - [x] Add `wheeler.cli reason` command.
- [ ] Task: Conductor - User Manual Verification 'Reasoning & Dreaming'
    - [ ] Verify `reason` command outputs logical blends.
    - [ ] Verify `dream` command strengthens memories over time.
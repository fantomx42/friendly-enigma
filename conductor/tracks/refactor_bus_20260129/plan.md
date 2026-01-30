# Implementation Plan: Message Bus Refactor

This plan follows the Test-Driven Development (TDD) workflow.

## Phase 1: Enhanced Message Schema [checkpoint: 3aa71d4]
- [x] Task: Create tests for `DiagnosticMessage`. [2f4b3c7]
    - [x] Create `ralph_core/tests/test_messages_diagnostic.py`.
    - [x] Define test cases for schema validation (required vs optional fields).
- [x] Task: Implement `DiagnosticMessage` class. [7b7b5b7]
    - [x] Update `ralph_core/protocols/messages.py`.
    - [x] Add `DiagnosticMessage` class inheriting from base `Message`.
    - [x] Implement validation logic.
- [x] Task: Conductor - User Manual Verification 'Enhanced Message Schema' (Protocol in workflow.md) [3aa71d4]

## Phase 2: Diagnostic Channel & Routing [checkpoint: ee1bd56]
- [x] Task: Create tests for Bus Routing. [48cd46f]
    - [x] Update `ralph_core/tests/test_bus.py`.
    - [x] Add test case: `test_publish_diagnostic_message`.
    - [x] Add test case: `test_diagnostic_subscriber_receives_event`.
- [x] Task: Implement Routing Logic. [3b74ed2]
    - [x] Modify `ralph_core/protocols/bus.py`.
    - [x] Add logic to handle `events/diagnostic` topic specifically.
    - [x] Ensure high-priority dispatch (bypass standard queue if applicable).
- [x] Task: Conductor - User Manual Verification 'Diagnostic Channel & Routing' (Protocol in workflow.md) [ee1bd56]

## Phase 3: Error Interception (The "Wiggum" Logic)
- [ ] Task: Create Integration Test for Error Handling.
    - [ ] Create `ralph_core/tests/test_runner_error.py`.
    - [ ] Mock a failing agent.
    - [ ] Assert that running the failing agent produces a `DiagnosticMessage` on the bus.
- [ ] Task: Implement Exception Catching in Runner.
    - [ ] Modify `ralph_core/runner.py` (execute method).
    - [ ] Wrap execution in try/except block.
    - [ ] On exception:
        - [ ] Capture traceback.
        - [ ] Create `DiagnosticMessage`.
        - [ ] Publish to bus.
- [ ] Task: Conductor - User Manual Verification 'Error Interception' (Protocol in workflow.md)

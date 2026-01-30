# Implementation Plan: Message Bus Refactor

This plan follows the Test-Driven Development (TDD) workflow.

## Phase 1: Enhanced Message Schema
- [x] Task: Create tests for `DiagnosticMessage`. [2f4b3c7]
    - [x] Create `ralph_core/tests/test_messages_diagnostic.py`.
    - [x] Define test cases for schema validation (required vs optional fields).
- [ ] Task: Implement `DiagnosticMessage` class.
    - [ ] Update `ralph_core/protocols/messages.py`.
    - [ ] Add `DiagnosticMessage` class inheriting from base `Message`.
    - [ ] Implement validation logic.
- [ ] Task: Conductor - User Manual Verification 'Enhanced Message Schema' (Protocol in workflow.md)

## Phase 2: Diagnostic Channel & Routing
- [ ] Task: Create tests for Bus Routing.
    - [ ] Update `ralph_core/tests/test_bus.py`.
    - [ ] Add test case: `test_publish_diagnostic_message`.
    - [ ] Add test case: `test_diagnostic_subscriber_receives_event`.
- [ ] Task: Implement Routing Logic.
    - [ ] Modify `ralph_core/protocols/bus.py`.
    - [ ] Add logic to handle `events/diagnostic` topic specifically.
    - [ ] Ensure high-priority dispatch (bypass standard queue if applicable).
- [ ] Task: Conductor - User Manual Verification 'Diagnostic Channel & Routing' (Protocol in workflow.md)

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

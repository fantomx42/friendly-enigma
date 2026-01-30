# Specification: Message Bus Refactor for Wiggum Protocol

## Objective
Enhance the existing `ralph_core/protocols/bus.py` and `messages.py` to support "Diagnostic Events." When an agent fails or an exception occurs, the bus must capture detailed context (stack trace, agent state, input) and route it to a new specialized channel that the `Reflector` agent can subscribe to. This enables the "Wiggum Protocol" (Iterate until success).

## Requirements

### 1. Enhanced Message Schema
- Update `Message` class in `ralph_core/protocols/messages.py`.
- Add optional fields for `error_context`:
    - `traceback`: str
    - `agent_state`: dict
    - `attempt_count`: int

### 2. Diagnostic Channel
- Implement a dedicated pub/sub channel topic: `events/diagnostic`.
- Ensure high-priority routing for these events (they should not be dropped by circuit breakers if possible).

### 3. Error Interception
- Modify `ralph_core/runner.py` (or the agent base class) to catch unhandled exceptions.
- Instead of crashing, wrap the exception in a `DiagnosticMessage` and publish it to the bus.

### 4. Circuit Breaker Integration
- Ensure the existing circuit breaker logic allows for "partial failure" (i.e., one agent down doesn't kill the bus, but triggers a diagnostic event).

## Acceptance Criteria
- [ ] A new `DiagnosticMessage` type is defined and validated.
- [ ] An agent throwing an exception results in a `DiagnosticMessage` appearing on the bus.
- [ ] The `Reflector` (mock or real) can subscribe to and receive these messages.
- [ ] Existing unit tests for the bus pass.
- [ ] New unit tests cover the error interception flow.

"""
Tests for Diagnostic Messages

Validates the schema and constructor for the Wiggum Protocol diagnostic events.
"""

import pytest
from ralph_core.protocols.messages import MessageType, Message

def test_diagnostic_message_type_exists():
    """Verify that DIAGNOSTIC message type is defined."""
    # This should fail initially as DIAGNOSTIC is not in MessageType
    assert hasattr(MessageType, "DIAGNOSTIC")
    assert MessageType.DIAGNOSTIC.value == "diagnostic"

def test_diagnostic_message_schema():
    """Verify the schema for DiagnosticMessage."""
    # This assumes a new class DiagnosticMessage will be created
    # or the base Message will be used with specific payload requirements.
    # We'll test for a specialized constructor.
    from ralph_core.protocols.messages import diagnostic_message
    
    error_text = "Something went wrong"
    traceback_text = "Traceback (most recent call last):\n  File \"...\", line 1, in <module>"
    agent_state = {"current_task": "test", "vram_usage": "low"}
    
    msg = diagnostic_message(
        error=error_text,
        traceback=traceback_text,
        agent_state=agent_state,
        attempt_count=2,
        sender="engineer"
    )
    
    assert msg.type == MessageType.DIAGNOSTIC
    assert msg.sender == "engineer"
    assert msg.receiver == "orchestrator" # Default receiver for diagnostics
    assert msg.payload["error"] == error_text
    assert msg.payload["error_context"]["traceback"] == traceback_text
    assert msg.payload["error_context"]["agent_state"] == agent_state
    assert msg.payload["error_context"]["attempt_count"] == 2

def test_diagnostic_message_serialization():
    """Verify that diagnostic messages serialize/deserialize correctly."""
    from ralph_core.protocols.messages import diagnostic_message
    
    msg = diagnostic_message(
        error="Fail",
        traceback="...",
        agent_state={},
        attempt_count=1
    )
    
    data = msg.to_dict()
    new_msg = Message.from_dict(data)
    
    assert new_msg.type == msg.type
    assert new_msg.payload == msg.payload
    assert new_msg.id == msg.id

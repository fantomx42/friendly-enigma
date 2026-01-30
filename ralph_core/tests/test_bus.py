"""
Tests for Message Bus Routing

Validates diagnostic channel routing and subscriber behavior.
"""

import pytest
from ralph_core.protocols.bus import MessageBus, BusConfig
from ralph_core.protocols.messages import Message, MessageType, diagnostic_message

def test_publish_diagnostic_message():
    """Verify that a diagnostic message can be sent through the bus."""
    bus = MessageBus(BusConfig(enable_logging=False))
    
    msg = diagnostic_message(
        error="Test",
        traceback="...",
        agent_state={},
        sender="engineer"
    )
    
    success = bus.send(msg)
    assert success is True
    assert bus.has_messages("orchestrator") # Default receiver
    
    received = bus.receive("orchestrator")
    assert received.type == MessageType.DIAGNOSTIC
    assert received.payload["error"] == "Test"

def test_diagnostic_subscriber_receives_event():
    """Verify that a subscriber to the diagnostic channel receives the event."""
    bus = MessageBus(BusConfig(enable_logging=False))
    
    received_messages = []
    def subscriber(msg):
        received_messages.append(msg)
    
    # This is the 'Red' part: the bus doesn't support topic-based handlers yet
    # Or if it does via MessageType, we need to verify if it works for DIAGNOSTIC
    bus.register_handler(MessageType.DIAGNOSTIC, subscriber)
    
    msg = diagnostic_message(
        error="Failure",
        traceback="...",
        agent_state={},
        sender="engineer"
    )
    
    bus.send(msg)
    
    assert len(received_messages) == 1
    assert received_messages[0].type == MessageType.DIAGNOSTIC
    assert received_messages[0].payload["error"] == "Failure"

def test_diagnostic_channel_priority():
    """
    Verify that diagnostic messages bypass standard circuit breakers if possible.
    (Requirement 2 in Spec)
    """
    # Create a bus at the message limit
    config = BusConfig(max_messages=1, enable_logging=False)
    bus = MessageBus(config)
    
    # Send one message to hit the limit
    bus.send(Message(MessageType.STATUS, "a", "b", {}))
    assert len(bus._history) == 1
    
    # A standard message should now fail (circuit breaker)
    assert bus.send(Message(MessageType.STATUS, "a", "b", {})) is False
    
    # A diagnostic message SHOULD still succeed (Priority Routing)
    msg = diagnostic_message("Critical", "...", {})
    success = bus.send(msg)
    
    assert success is True # This will fail in Red phase
    assert bus.has_messages("orchestrator")

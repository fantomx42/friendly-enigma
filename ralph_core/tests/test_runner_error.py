"""
Integration Tests for Runner Error Interception

Validates that agent exceptions are caught by the runner and 
converted into DiagnosticMessages on the bus.
"""

import pytest
from unittest.mock import MagicMock, patch
from ralph_core.runner import run_v2_pipeline
from ralph_core.protocols.messages import Message, MessageType, DiagnosticMessage

@patch("ralph_core.runner.get_bus")
@patch("ralph_core.runner.translate")
@patch("ralph_core.runner.think")
@patch("ralph_core.runner.Memory")
def test_runner_catches_agent_exception(mock_memory, mock_think, mock_translate, mock_get_bus):
    """
    Verify that an exception in an agent handler results in a DiagnosticMessage.
    """
    # Setup Bus
    bus = MagicMock()
    mock_get_bus.return_value = bus
    
    # Setup mocks to allow the pipeline to reach the message loop
    mock_translate.return_value = MagicMock(to_dict=lambda: {}, to_prompt_context=lambda: "")
    mock_think.return_value = "Test Plan"
    
    # Simulate a work request being sent and then processed
    # We'll mock the handler to throw an error
    def failing_handler(msg):
        raise ValueError("Simulated Agent Crash")
    
    # We need to mock the AGENT_HANDLERS which is local to run_v2_pipeline
    # Instead, we can mock the actual handler functions imported into runner.py
    with patch("ralph_core.runner.engineer_handle_message", side_effect=failing_handler):
        # We need to make sure bus.has_messages returns True once for engineer
        bus.has_messages.side_effect = lambda agent: agent == "engineer"
        # And bus.receive returns a valid message
        bus.receive.return_value = Message(MessageType.WORK_REQUEST, "orchestrator", "engineer", {})
        
        # We need to prevent infinite loop in the test
        # Let's make bus.has_messages return False after the first call
        has_messages_calls = [0]
        def has_messages_mock(agent):
            if agent == "engineer" and has_messages_calls[0] == 0:
                has_messages_calls[0] += 1
                return True
            return False
        bus.has_messages.side_effect = has_messages_mock

        # Run the pipeline (V2)
        # Note: We need to set RALPH_V2=1 or mock USE_V2
        with patch("ralph_core.runner.USE_V2", True):
            run_v2_pipeline("test objective", "context", "plan", mock_memory())
    
    # Assertions
    # Check if bus.send was called with a DiagnosticMessage
    # The first call is the initial work request, subsequent should be diagnostic
    sent_messages = [call.args[0] for call in bus.send.call_args_list]
    
    diagnostic_msgs = [m for m in sent_messages if isinstance(m, DiagnosticMessage)]
    
    assert len(diagnostic_msgs) > 0, "No DiagnosticMessage was sent to the bus"
    assert diagnostic_msgs[0].payload["error"] == "Simulated Agent Crash"
    assert "ValueError" in diagnostic_msgs[0].payload["error_context"]["traceback"]

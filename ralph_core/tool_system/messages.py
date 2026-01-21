"""
Tool Protocol Messages

Message constructors for tool request/response through the message bus.
"""

import sys
import os

# Ensure protocols is importable
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from typing import Any, Optional
from protocols.messages import Message, MessageType


def tool_request(
    tool_name: str,
    arguments: dict,
    requester: str = "engineer",
    return_to: str = "engineer",
    correlation_id: Optional[str] = None,
) -> Message:
    """
    Create a TOOL_REQUEST message.

    Args:
        tool_name: Name of tool to invoke
        arguments: Dictionary of tool arguments
        requester: Agent making the request
        return_to: Agent to send response to
        correlation_id: Optional ID for request tracking

    Returns:
        Message ready to send to tool dispatcher
    """
    return Message(
        type=MessageType.TOOL_REQUEST,
        sender=requester,
        receiver="tool_dispatcher",
        payload={
            "tool_name": tool_name,
            "arguments": arguments,
        },
        metadata={
            "return_to": return_to,
        },
        correlation_id=correlation_id,
    )


def tool_response(
    tool_name: str,
    success: bool,
    result: Any,
    return_to: str,
    execution_time_ms: int = 0,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Message:
    """
    Create a TOOL_RESPONSE message.

    Args:
        tool_name: Name of tool that was invoked
        success: Whether execution succeeded
        result: Tool execution result
        return_to: Agent to send response to
        execution_time_ms: Execution time in milliseconds
        error: Error message if failed
        correlation_id: ID linking to original request

    Returns:
        Message with tool results
    """
    return Message(
        type=MessageType.TOOL_RESPONSE,
        sender="tool_dispatcher",
        receiver=return_to,
        payload={
            "tool_name": tool_name,
            "success": success,
            "result": result,
            "execution_time_ms": execution_time_ms,
            "error": error,
        },
        correlation_id=correlation_id,
    )


def tool_confirm(
    tool_name: str,
    request_id: str,
    confirmed: bool,
    responder: str = "user",
) -> Message:
    """
    Create a TOOL_CONFIRM message for pending confirmations.

    Args:
        tool_name: Name of tool awaiting confirmation
        request_id: ID of the pending request
        confirmed: Whether user confirmed
        responder: Who confirmed (usually "user")

    Returns:
        Message to complete or cancel pending tool execution
    """
    return Message(
        type=MessageType.TOOL_CONFIRM,
        sender=responder,
        receiver="tool_dispatcher",
        payload={
            "tool_name": tool_name,
            "request_id": request_id,
            "confirmed": confirmed,
        },
    )

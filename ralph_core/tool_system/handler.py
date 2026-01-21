"""
Tool Message Handler - Message Bus Integration

Handles TOOL_REQUEST, TOOL_CONFIRM messages and produces TOOL_RESPONSE.
Integrates with the message bus routing system.
"""

import sys
import os
from typing import Optional

# Ensure ralph_core is importable
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from .dispatcher import dispatcher
from .messages import tool_response


def handle_tool_request(message: "Message") -> Optional["Message"]:
    """
    Handle a TOOL_REQUEST message.

    Args:
        message: Incoming TOOL_REQUEST

    Returns:
        TOOL_RESPONSE message (or None if confirmation needed)
    """
    from protocols.messages import Message, MessageType

    payload = message.payload
    tool_name = payload.get("tool_name", "")
    arguments = payload.get("arguments", {})
    requester = message.sender
    return_to = message.metadata.get("return_to", requester)
    request_id = message.id

    # Try to audit the request
    try:
        from security.towers import audit, AuditLevel
        audit.log(
            event_type="tool_request",
            level=AuditLevel.INFO,
            output_type="tool",
            content_summary=f"{tool_name}({', '.join(arguments.keys())})",
            metadata={
                "requester": requester,
                "tool_name": tool_name,
                "arguments_keys": list(arguments.keys()),
            }
        )
    except ImportError:
        pass  # Audit not available

    # Dispatch the tool
    result = dispatcher.dispatch(
        tool_name=tool_name,
        arguments=arguments,
        requester=requester,
        request_id=request_id,
    )

    # Try to audit the result
    try:
        from security.towers import audit, AuditLevel
        audit.log(
            event_type="tool_result",
            level=AuditLevel.INFO if result.success else AuditLevel.WARNING,
            output_type="tool",
            content_summary=f"{tool_name} -> {result.security_decision}",
            metadata={
                "success": result.success,
                "security_decision": result.security_decision,
                "execution_time_ms": result.execution_time_ms,
            }
        )
    except ImportError:
        pass

    if result.requires_confirmation:
        # Don't send response yet - awaiting confirmation
        print(f"[ToolHandler] Awaiting confirmation for {tool_name}")
        return None

    return tool_response(
        tool_name=tool_name,
        success=result.success,
        result=result.result,
        return_to=return_to,
        execution_time_ms=result.execution_time_ms,
        error=result.blocked_reason,
        correlation_id=message.id,
    )


def handle_tool_confirm(message: "Message") -> Optional["Message"]:
    """
    Handle a TOOL_CONFIRM message.

    Args:
        message: Incoming TOOL_CONFIRM

    Returns:
        TOOL_RESPONSE message with execution result
    """
    payload = message.payload
    tool_name = payload.get("tool_name", "")
    request_id = payload.get("request_id", "")
    confirmed = payload.get("confirmed", False)

    if not confirmed:
        # User declined - return failure
        return tool_response(
            tool_name=tool_name,
            success=False,
            result=None,
            return_to="engineer",  # Default
            error="User declined confirmation",
            correlation_id=request_id,
        )

    # Execute confirmed request
    result = dispatcher.confirm_pending(request_id)

    if result is None:
        return tool_response(
            tool_name=tool_name,
            success=False,
            result=None,
            return_to="engineer",
            error="Confirmation expired or not found",
            correlation_id=request_id,
        )

    return tool_response(
        tool_name=tool_name,
        success=result.success,
        result=result.result,
        return_to="engineer",
        execution_time_ms=result.execution_time_ms,
        error=result.blocked_reason,
        correlation_id=request_id,
    )


def handle_message(message: "Message") -> Optional["Message"]:
    """
    Main message handler for tool system.
    """
    from protocols.messages import MessageType

    msg_type = message.type

    if msg_type == MessageType.TOOL_REQUEST:
        return handle_tool_request(message)
    elif msg_type == MessageType.TOOL_CONFIRM:
        return handle_tool_confirm(message)
    else:
        from protocols.messages import error_message
        return error_message(
            error=f"Tool handler cannot process: {msg_type}",
            recoverable=True,
        )


# Handler registry for message bus
TOOL_HANDLERS = {
    "tool_dispatcher": handle_message,
}

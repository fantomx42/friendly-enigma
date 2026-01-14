"""
ASIC Message Handler - Message Bus Integration

Handles ASIC_REQUEST messages and spawns specialists to generate options.
Returns ASIC_RESPONSE with multiple candidate solutions.

Message Flow:
    Engineer --[ASIC_REQUEST]--> ASICHandler --[ASIC_RESPONSE]--> Engineer/Designer
"""

import sys
import os

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from typing import Optional
from .spawner import spawn_asic, spawn_asic_parallel
from .registry import get_asic_config, list_available_asics, get_model_with_fallback


def handle_asic_request(message: "Message") -> Optional["Message"]:
    """
    Handle an ASIC_REQUEST message and spawn a specialist.

    Args:
        message: Incoming ASIC_REQUEST message

    Returns:
        ASIC_RESPONSE message with options, or ERROR message
    """
    from protocols.messages import (
        Message, MessageType,
        asic_response, error_message,
    )

    payload = message.payload
    task_type = payload.get("task_type", "")
    prompt = payload.get("prompt", "")
    parallel = payload.get("parallel", False)
    return_to = message.metadata.get("return_to", "engineer")

    print(f"[ASIC:{task_type}] Processing request...")

    # Validate task type
    if not task_type:
        return error_message(
            error="ASIC_REQUEST missing task_type",
            context=f"Payload: {payload}",
            recoverable=True,
        )

    # Get config to report model used
    config = get_asic_config(task_type)
    model_used = ""
    if config:
        model_used = get_model_with_fallback(config)
        print(f"[ASIC:{task_type}] Using model: {model_used}")
    else:
        print(f"[ASIC:{task_type}] Unknown type, using small_code fallback")

    # Spawn the ASIC
    try:
        if parallel:
            options = spawn_asic_parallel(task_type, prompt)
        else:
            options = spawn_asic(task_type, prompt)

        # Filter out error options
        valid_options = [opt for opt in options if not opt.startswith("Error:") and not opt.startswith("ASIC ERROR:")]

        if not valid_options:
            print(f"[ASIC:{task_type}] All options failed, returning errors")
            valid_options = options  # Return errors so caller knows what happened

        print(f"[ASIC:{task_type}] Generated {len(valid_options)} options")

        return asic_response(
            task_type=task_type,
            options=valid_options,
            model_used=model_used,
            return_to=return_to,
        )

    except Exception as e:
        print(f"[ASIC:{task_type}] Error: {e}")
        return error_message(
            error=f"ASIC spawn failed: {str(e)}",
            context=f"task_type={task_type}, prompt={prompt[:100]}...",
            recoverable=True,
        )


def handle_message(message: "Message") -> Optional["Message"]:
    """
    Main message handler for ASIC system.

    Routes based on message type and task_type in receiver.
    """
    from protocols.messages import MessageType, error_message

    msg_type = message.type

    if msg_type == MessageType.ASIC_REQUEST:
        return handle_asic_request(message)

    else:
        return error_message(
            error=f"ASIC handler cannot process message type: {msg_type}",
            recoverable=True,
        )


def get_asic_handler_for_type(task_type: str):
    """
    Get a handler function bound to a specific ASIC type.

    Useful for registering multiple ASIC handlers in the message bus.
    """
    def handler(message: "Message") -> Optional["Message"]:
        # Override task_type in payload if not set
        if "task_type" not in message.payload:
            message.payload["task_type"] = task_type
        return handle_asic_request(message)

    return handler


# Pre-built handlers for common ASIC types
ASIC_HANDLERS = {
    f"asic:{task_type}": get_asic_handler_for_type(task_type)
    for task_type in list_available_asics()
}

"""
Protocols - Message Bus for Agent Communication

This module provides the communication infrastructure for bidirectional
agent collaboration. Instead of a linear pipeline (Orchestrator → Engineer → Designer),
agents can now send messages back and forth:

- Designer can request revisions from Engineer
- Engineer can spawn ASICs and collect options
- Orchestrator can monitor progress and intervene

Message Flow:
    Orchestrator --[WORK_REQUEST]--> Engineer
    Engineer --[CODE_OUTPUT]--> Designer
    Designer --[REVISION_REQUEST]--> Engineer (if quality insufficient)
    Designer --[COMPLETE]--> Orchestrator (when satisfied)
"""

from .messages import (
    Message, MessageType,
    work_request, code_output, revision_request,
    options_message, complete_message, error_message,
    # ASIC messages
    asic_request, asic_response,
)
from .bus import MessageBus, get_bus, reset_bus, BusConfig

__all__ = [
    # Core types
    "Message", "MessageType", "MessageBus", "BusConfig",

    # Bus helpers
    "get_bus", "reset_bus",

    # Message constructors
    "work_request", "code_output", "revision_request",
    "options_message", "complete_message", "error_message",

    # ASIC message constructors
    "asic_request", "asic_response",
]

"""
tools/ - Enhanced Tool Registry with Security

Provides:
- ToolSpec: Full tool specifications with metadata
- EnhancedToolRegistry: Registry with type info and policies
- ToolDispatcher: Secure execution through message bus
- Message constructors: tool_request, tool_response, tool_confirm
"""

from .spec import (
    ToolSpec, ToolCategory, SideEffect, PermissionLevel, ArgumentSpec
)
from .registry import (
    EnhancedToolRegistry, ToolInvocationResult, enhanced_registry
)
from .dispatcher import (
    ToolDispatcher, ToolDispatchResult, dispatcher, dispatch_tool
)
from .messages import tool_request, tool_response, tool_confirm
from .handler import handle_message, TOOL_HANDLERS

__all__ = [
    # Specifications
    "ToolSpec", "ToolCategory", "SideEffect", "PermissionLevel", "ArgumentSpec",
    # Registry
    "EnhancedToolRegistry", "ToolInvocationResult", "enhanced_registry",
    # Dispatcher
    "ToolDispatcher", "ToolDispatchResult", "dispatcher", "dispatch_tool",
    # Messages
    "tool_request", "tool_response", "tool_confirm",
    # Handler
    "handle_message", "TOOL_HANDLERS",
]

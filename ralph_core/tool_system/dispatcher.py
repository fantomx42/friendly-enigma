"""
Tool Dispatcher - Message-driven tool execution with security.

Receives TOOL_REQUEST messages, validates through Security Checkpoint,
executes tools, and returns TOOL_RESPONSE messages.
"""

import time
from typing import Optional, Any
from dataclasses import dataclass

from .registry import enhanced_registry
from .spec import PermissionLevel, ToolCategory


@dataclass
class ToolDispatchResult:
    """Result from dispatch attempt."""
    success: bool
    result: Any
    tool_name: str
    execution_time_ms: int
    security_decision: str  # "PASS", "BLOCK", "CONFIRM"
    blocked_reason: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None


class ToolDispatcher:
    """
    Dispatches tool requests with security validation.

    All tool execution goes through:
    1. Registry lookup
    2. Argument validation
    3. Rate limit check
    4. Security checkpoint
    5. Execution
    6. Result packaging
    """

    def __init__(self):
        self.registry = enhanced_registry
        self._pending_confirmations: dict[str, dict] = {}  # request_id -> request_data

        # Import security at runtime to avoid circular imports
        self._checkpoint = None
        self._is_safe_to_output = None

    def _get_security(self):
        """Lazy load security modules."""
        if self._checkpoint is None:
            try:
                from security import checkpoint, is_safe_to_output
                self._checkpoint = checkpoint
                self._is_safe_to_output = is_safe_to_output
            except ImportError:
                self._checkpoint = None
                self._is_safe_to_output = None
        return self._checkpoint, self._is_safe_to_output

    def dispatch(
        self,
        tool_name: str,
        arguments: dict,
        requester: str = "unknown",
        request_id: Optional[str] = None,
    ) -> ToolDispatchResult:
        """
        Dispatch a tool request with full security validation.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Dictionary of arguments
            requester: Name of requesting agent
            request_id: Optional correlation ID

        Returns:
            ToolDispatchResult with execution results
        """
        start_time = time.time()

        # 1. Registry lookup
        spec = self.registry.get_tool(tool_name)
        if not spec:
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=tool_name,
                execution_time_ms=0,
                security_decision="BLOCK",
                blocked_reason=f"Unknown tool: {tool_name}",
            )

        # 2. Argument validation
        args_valid, issues = spec.validate_args(arguments)
        if not args_valid:
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=tool_name,
                execution_time_ms=0,
                security_decision="BLOCK",
                blocked_reason=f"Invalid arguments: {', '.join(issues)}",
            )

        # 3. Rate limit check
        if not self.registry.check_rate_limit(tool_name):
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=tool_name,
                execution_time_ms=0,
                security_decision="BLOCK",
                blocked_reason=f"Rate limit exceeded for {tool_name}",
            )

        # 4. Security checkpoint
        checkpoint, is_safe_to_output = self._get_security()
        if is_safe_to_output:
            # Map tool category to output type
            output_type = self._category_to_output_type(spec.category)

            # Build content summary for security check
            content = self._build_security_content(spec, arguments)

            safe, reason = is_safe_to_output(content, output_type)

            if not safe:
                # Check if this should be ASK vs BLOCK
                if spec.permission_level == PermissionLevel.ASK:
                    # Store for later confirmation
                    self._pending_confirmations[request_id or tool_name] = {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "requester": requester,
                        "confirmation_message": reason,
                    }
                    return ToolDispatchResult(
                        success=False,
                        result=None,
                        tool_name=tool_name,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        security_decision="CONFIRM",
                        requires_confirmation=True,
                        confirmation_message=reason,
                    )
                else:
                    return ToolDispatchResult(
                        success=False,
                        result=None,
                        tool_name=tool_name,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        security_decision="BLOCK",
                        blocked_reason=reason,
                    )

        # Check permission level before execution
        if spec.permission_level == PermissionLevel.DENY:
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=tool_name,
                execution_time_ms=0,
                security_decision="BLOCK",
                blocked_reason=f"Tool {tool_name} is denied by policy",
            )

        # 5. Execute tool
        try:
            result = spec.func(**arguments)
            self.registry.record_call(tool_name)

            execution_time = int((time.time() - start_time) * 1000)

            return ToolDispatchResult(
                success=True,
                result=result,
                tool_name=tool_name,
                execution_time_ms=execution_time,
                security_decision="PASS",
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=tool_name,
                execution_time_ms=execution_time,
                security_decision="PASS",
                blocked_reason=f"Execution error: {str(e)}",
            )

    def confirm_pending(self, request_id: str) -> Optional[ToolDispatchResult]:
        """Execute a pending confirmed request."""
        pending = self._pending_confirmations.pop(request_id, None)
        if not pending:
            return None

        # Re-dispatch without security check (already confirmed)
        spec = self.registry.get_tool(pending["tool_name"])
        if not spec:
            return None

        start_time = time.time()
        try:
            result = spec.func(**pending["arguments"])
            self.registry.record_call(pending["tool_name"])

            return ToolDispatchResult(
                success=True,
                result=result,
                tool_name=pending["tool_name"],
                execution_time_ms=int((time.time() - start_time) * 1000),
                security_decision="CONFIRMED",
            )
        except Exception as e:
            return ToolDispatchResult(
                success=False,
                result=None,
                tool_name=pending["tool_name"],
                execution_time_ms=int((time.time() - start_time) * 1000),
                security_decision="CONFIRMED",
                blocked_reason=f"Execution error: {str(e)}",
            )

    def _category_to_output_type(self, category: ToolCategory) -> str:
        """Map tool category to security output type."""
        mapping = {
            ToolCategory.FILE_READ: "terminal",
            ToolCategory.FILE_WRITE: "file",
            ToolCategory.SHELL: "shell",
            ToolCategory.MEMORY: "terminal",
            ToolCategory.GIT: "git",
            ToolCategory.WEB: "api",
            ToolCategory.VISION: "terminal",
            ToolCategory.SWARM: "shell",
        }
        return mapping.get(category, "terminal")

    def _build_security_content(self, spec, arguments: dict) -> str:
        """Build content string for security check."""
        if spec.category == ToolCategory.SHELL:
            return arguments.get("command", "")
        elif spec.category == ToolCategory.FILE_WRITE:
            return arguments.get("content", "")
        elif spec.category == ToolCategory.GIT:
            return arguments.get("message", "") or arguments.get("branch", "")
        else:
            # For other tools, just summarize the call
            return f"{spec.name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"


# Global dispatcher
dispatcher = ToolDispatcher()


def dispatch_tool(
    tool_name: str,
    arguments: dict,
    requester: str = "unknown",
) -> ToolDispatchResult:
    """Convenience function for tool dispatch."""
    return dispatcher.dispatch(tool_name, arguments, requester)

"""
Enhanced Tool Registry with metadata and security integration.

Backward compatible with existing tools.py while adding:
- Full argument/return type specs
- Side effect declarations
- Per-tool permission policies
- Security checkpoint integration
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

from .spec import ToolSpec, ToolCategory, SideEffect, PermissionLevel, ArgumentSpec


@dataclass
class ToolInvocationResult:
    """Result of a tool invocation."""
    success: bool
    result: Any
    error: Optional[str] = None
    tool_name: str = ""
    execution_time_ms: int = 0
    security_passed: bool = True
    blocked_reason: Optional[str] = None


class EnhancedToolRegistry:
    """
    Enhanced Tool Registry with security and metadata.

    Maintains backward compatibility with simple register(name, func, desc)
    while supporting full ToolSpec registration.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._category_index: Dict[ToolCategory, List[str]] = defaultdict(list)

    def register_spec(self, spec: ToolSpec) -> None:
        """Register a tool with full specification."""
        self._tools[spec.name] = spec
        self._category_index[spec.category].append(spec.name)

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        category: ToolCategory = ToolCategory.FILE_READ,
        side_effects: Optional[List[SideEffect]] = None,
        permission_level: PermissionLevel = PermissionLevel.ALLOW,
    ) -> None:
        """
        Simple registration (backward compatible).

        Creates a ToolSpec with inferred argument info.
        """
        # Infer arguments from function signature
        sig = inspect.signature(func)
        arguments = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            # Get type annotation or default to str
            arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            # Handle special type annotations
            if arg_type == inspect.Parameter.empty:
                arg_type = str
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            arguments.append(ArgumentSpec(
                name=param_name,
                type=arg_type,
                required=required,
                default=default,
            ))

        spec = ToolSpec(
            name=name,
            description=description,
            func=func,
            category=category,
            arguments=arguments,
            side_effects=side_effects or [],
            permission_level=permission_level,
        )
        self.register_spec(spec)

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get tool specification by name."""
        return self._tools.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        """Get tool function (backward compatible)."""
        spec = self._tools.get(name)
        return spec.func if spec else None

    def list_tools(self) -> str:
        """Return formatted list of tools (backward compatible)."""
        return "\n".join(
            f"- {name}: {spec.description}"
            for name, spec in self._tools.items()
        )

    def list_tools_for_prompt(self) -> str:
        """Generate tool descriptions for LLM prompts."""
        return "\n".join(
            spec.to_prompt_description()
            for spec in self._tools.values()
        )

    def get_by_category(self, category: ToolCategory) -> List[ToolSpec]:
        """Get all tools in a category."""
        return [self._tools[name] for name in self._category_index.get(category, [])]

    def check_rate_limit(self, name: str) -> bool:
        """Check if tool is within rate limit."""
        spec = self._tools.get(name)
        if not spec or spec.max_calls_per_session == 0:
            return True
        return self._call_counts[name] < spec.max_calls_per_session

    def record_call(self, name: str) -> None:
        """Record a tool call for rate limiting."""
        self._call_counts[name] += 1

    def reset_counts(self) -> None:
        """Reset call counts (for new session)."""
        self._call_counts.clear()

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools


# Global enhanced registry
enhanced_registry = EnhancedToolRegistry()

"""
Tool Specifications - Metadata for all registered tools.

Provides argument schemas, return types, side effects, and permission requirements
for safe tool invocation through the message bus.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Type


class ToolCategory(Enum):
    """Categories of tools for grouping and policy application."""
    FILE_READ = "file_read"      # read_file, list_dir
    FILE_WRITE = "file_write"    # write_file
    SHELL = "shell"              # run_shell
    MEMORY = "memory"            # memory_save, memory_get, memory_search
    GIT = "git"                  # git_commit, git_branch, git_revert
    WEB = "web"                  # web_search, read_url
    VISION = "vision"            # analyze_image
    SWARM = "swarm"              # dispatch_swarm


class SideEffect(Enum):
    """Possible side effects of tool execution."""
    NONE = "none"                # Pure read operation
    FILE_CREATE = "file_create"  # Creates new file(s)
    FILE_MODIFY = "file_modify"  # Modifies existing file(s)
    FILE_DELETE = "file_delete"  # Deletes file(s)
    PROCESS = "process"          # Spawns subprocess
    NETWORK = "network"          # Network I/O
    STATE = "state"              # Modifies internal state


class PermissionLevel(Enum):
    """Permission levels for tool execution."""
    ALLOW = "allow"              # Execute without confirmation
    WARN = "warn"                # Log warning but execute
    ASK = "ask"                  # Require user confirmation
    DENY = "deny"                # Block execution entirely


@dataclass
class ArgumentSpec:
    """Specification for a tool argument."""
    name: str
    type: Type                   # Python type (str, int, List[str], etc.)
    required: bool = True
    default: Any = None
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None  # Custom validation


@dataclass
class ToolSpec:
    """Complete specification for a tool."""
    name: str
    description: str
    func: Callable
    category: ToolCategory
    arguments: List[ArgumentSpec] = field(default_factory=list)
    return_type: Type = str
    side_effects: List[SideEffect] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.ALLOW
    required_capabilities: List[str] = field(default_factory=list)  # e.g., ["sandbox"]
    max_calls_per_session: int = 0  # 0 = unlimited
    timeout_seconds: int = 30

    def validate_args(self, args: dict) -> tuple[bool, List[str]]:
        """Validate arguments against spec."""
        issues = []
        for arg_spec in self.arguments:
            if arg_spec.name not in args:
                if arg_spec.required:
                    issues.append(f"Missing required argument: {arg_spec.name}")
            else:
                value = args[arg_spec.name]
                # Type check (handle None for optional args)
                if value is not None and not isinstance(value, arg_spec.type):
                    issues.append(
                        f"Argument {arg_spec.name} should be {arg_spec.type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                # Custom validation
                if arg_spec.validator and value is not None and not arg_spec.validator(value):
                    issues.append(f"Argument {arg_spec.name} failed validation")
        return len(issues) == 0, issues

    def to_prompt_description(self) -> str:
        """Generate description suitable for LLM prompts."""
        args_desc = ", ".join(
            f"{a.name}: {a.type.__name__}" + ("?" if not a.required else "")
            for a in self.arguments
        )
        effects = ", ".join(e.value for e in self.side_effects) if self.side_effects else "none"
        return (
            f"{self.name}({args_desc}) -> {self.return_type.__name__}\n"
            f"  Description: {self.description}\n"
            f"  Side effects: {effects}\n"
        )

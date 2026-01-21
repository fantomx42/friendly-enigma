"""
tools.py - The Tool Registry for Ralph.
Provides a structured set of safe, importable capabilities for generated code.

This module wraps the enhanced tools/ package for backward compatibility.
"""

import os
import sys
from typing import List, Dict, Any, Callable

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.abspath(__file__))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from executor import Executor
from git_manager import git
from web import web
from vision import vision
from swarm_dispatcher import dispatcher as swarm_dispatcher
from memory import Memory

# Import enhanced registry components
from tool_system import (
    enhanced_registry, ToolCategory, SideEffect, PermissionLevel,
    dispatch_tool, TOOL_HANDLERS
)


class ToolRegistry:
    """
    Backward-compatible Tool Registry.

    Registers tools in both the legacy format and the enhanced registry
    for full compatibility with existing code and new tool protocol.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self.executor = Executor()
        self.memory = Memory()

        # Reference to enhanced registry
        self._enhanced = enhanced_registry

        # Register core tools with full metadata
        self._register_core_tools()
        self._register_memory_tools()
        self._register_git_tools()
        self._register_web_tools()
        self._register_vision_tools()
        self._register_swarm_tools()

    def _register_core_tools(self):
        """Register file and shell tools with full metadata."""
        # read_file - safe read operation
        self._register_enhanced(
            name="read_file",
            func=self.read_file,
            description="Reads content of a file.",
            category=ToolCategory.FILE_READ,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )

        # write_file - modifies filesystem
        self._register_enhanced(
            name="write_file",
            func=self.write_file,
            description="Writes content to a file.",
            category=ToolCategory.FILE_WRITE,
            side_effects=[SideEffect.FILE_CREATE, SideEffect.FILE_MODIFY],
            permission_level=PermissionLevel.WARN,
        )

        # list_dir - safe read operation
        self._register_enhanced(
            name="list_dir",
            func=self.list_dir,
            description="Lists files in a directory.",
            category=ToolCategory.FILE_READ,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )

        # run_shell - spawns subprocess
        self._register_enhanced(
            name="run_shell",
            func=self.run_shell,
            description="Executes a shell command.",
            category=ToolCategory.SHELL,
            side_effects=[SideEffect.PROCESS],
            permission_level=PermissionLevel.WARN,
        )

    def _register_memory_tools(self):
        """Register memory tools."""
        self._register_enhanced(
            name="memory_save",
            func=self.memory.save,
            description="Saves a fact to memory. Args: fact (str), tag (str, optional)",
            category=ToolCategory.MEMORY,
            side_effects=[SideEffect.STATE],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="memory_get",
            func=self.memory.get,
            description="Retrieves facts by tag. Args: tag (str)",
            category=ToolCategory.MEMORY,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="memory_search",
            func=self.memory.search,
            description="Semantic search for similar facts. Args: query (str)",
            category=ToolCategory.MEMORY,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="memory_remember",
            func=self.memory.remember,
            description="Saves a fact with tag. Args: fact (str), tag (str)",
            category=ToolCategory.MEMORY,
            side_effects=[SideEffect.STATE],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="memory_recall",
            func=self.memory.recall,
            description="Retrieves facts by tag. Args: tag (str)",
            category=ToolCategory.MEMORY,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )

    def _register_git_tools(self):
        """Register git tools."""
        self._register_enhanced(
            name="git_commit",
            func=git.commit_all,
            description="Stages and commits all changes.",
            category=ToolCategory.GIT,
            side_effects=[SideEffect.STATE],
            permission_level=PermissionLevel.WARN,
        )
        self._register_enhanced(
            name="git_branch",
            func=git.create_branch,
            description="Creates a new branch.",
            category=ToolCategory.GIT,
            side_effects=[SideEffect.STATE],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="git_revert",
            func=git.revert_last_commit,
            description="Reverts the last commit.",
            category=ToolCategory.GIT,
            side_effects=[SideEffect.STATE],
            permission_level=PermissionLevel.ASK,
        )

    def _register_web_tools(self):
        """Register web tools."""
        self._register_enhanced(
            name="web_search",
            func=web.search,
            description="Searches the web for a query.",
            category=ToolCategory.WEB,
            side_effects=[SideEffect.NETWORK],
            permission_level=PermissionLevel.ALLOW,
        )
        self._register_enhanced(
            name="read_url",
            func=web.fetch_page,
            description="Reads the content of a URL.",
            category=ToolCategory.WEB,
            side_effects=[SideEffect.NETWORK],
            permission_level=PermissionLevel.ALLOW,
        )

    def _register_vision_tools(self):
        """Register vision tools."""
        self._register_enhanced(
            name="analyze_image",
            func=vision.analyze_image,
            description="Analyzes an image file. Args: image_path, prompt",
            category=ToolCategory.VISION,
            side_effects=[SideEffect.NONE],
            permission_level=PermissionLevel.ALLOW,
        )

    def _register_swarm_tools(self):
        """Register swarm tools."""
        self._register_enhanced(
            name="dispatch_swarm",
            func=swarm_dispatcher.dispatch,
            description="Spawns parallel Ralph instances for subtasks. Args: [list_of_tasks]",
            category=ToolCategory.SWARM,
            side_effects=[SideEffect.PROCESS],
            permission_level=PermissionLevel.ASK,
        )

    def _register_enhanced(
        self,
        name: str,
        func: Callable,
        description: str,
        category: ToolCategory,
        side_effects: List[SideEffect],
        permission_level: PermissionLevel,
    ):
        """Register a tool in both legacy and enhanced registries."""
        # Legacy registration
        self._tools[name] = func
        self._descriptions[name] = description

        # Enhanced registration
        self._enhanced.register(
            name=name,
            func=func,
            description=description,
            category=category,
            side_effects=side_effects,
            permission_level=permission_level,
        )

    def register(self, name: str, func: Callable, description: str):
        """Registers a new tool function (legacy interface)."""
        self._tools[name] = func
        self._descriptions[name] = description
        # Also register in enhanced registry with defaults
        self._enhanced.register(name, func, description)

    def get_tool(self, name: str) -> Callable:
        """Retrieves a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> str:
        """Returns a formatted list of available tools."""
        return "\n".join([f"- {name}: {desc}" for name, desc in self._descriptions.items()])

    def list_tools_for_prompt(self) -> str:
        """Returns tool descriptions formatted for LLM prompts."""
        return self._enhanced.list_tools_for_prompt()

    # --- Core Tool Implementations ---

    def read_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str) -> str:
        try:
            # Safety: If overwriting a core file, make a backup first
            if os.path.exists(file_path):
                # Check if it's a python file
                if file_path.endswith(".py"):
                    backup_path = file_path + ".bak"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original = f.read()
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original)
                    print(f"[Tools] Created backup: {backup_path}")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_dir(self, path: str = ".") -> List[str]:
        try:
            return os.listdir(path)
        except Exception as e:
            return [f"Error: {str(e)}"]

    def run_shell(self, command: str) -> Dict[str, Any]:
        return self.executor.run(command)


# Global Instance
registry = ToolRegistry()

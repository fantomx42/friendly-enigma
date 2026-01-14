"""
tools.py - The Tool Registry for Ralph.
Provides a structured set of safe, importable capabilities for generated code.
"""

import os
import glob
from typing import List, Dict, Any, Callable
from .executor import Executor
from .git_manager import git
from .web import web
from .vision import vision
from .swarm_dispatcher import dispatcher

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self.executor = Executor()
        
        # Register core tools
        self.register("read_file", self.read_file, "Reads content of a file.")
        self.register("write_file", self.write_file, "Writes content to a file.")
        self.register("list_dir", self.list_dir, "Lists files in a directory.")
        self.register("run_shell", self.run_shell, "Executes a shell command.")
        
        # Register Git tools
        self.register("git_commit", git.commit_all, "Stages and commits all changes.")
        self.register("git_branch", git.create_branch, "Creates a new branch.")
        self.register("git_revert", git.revert_last_commit, "Reverts the last commit.")
        
        # Register Web tools
        self.register("web_search", web.search, "Searches the web for a query.")
        self.register("read_url", web.fetch_page, "Reads the content of a URL.")
        
        # Register Vision tools
        self.register("analyze_image", vision.analyze_image, "Analyzes an image file. Args: image_path, prompt")
        
        # Register Swarm tools
        self.register("dispatch_swarm", dispatcher.dispatch, "Spawns parallel Ralph instances for subtasks. Args: [list_of_tasks]")

    def register(self, name: str, func: Callable, description: str):
        """Registers a new tool function."""
        self._tools[name] = func
        self._descriptions[name] = description

    def get_tool(self, name: str) -> Callable:
        """Retrieves a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> str:
        """Returns a formatted list of available tools."""
        return "\n".join([f"- {name}: {desc}" for name, desc in self._descriptions.items()])

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

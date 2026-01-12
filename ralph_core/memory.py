"""
memory.py - The Hippocampus of Ralph.
Handles persistence of state (Context) and knowledge (Facts).
"""

import json
import os
import glob
from typing import Dict, Optional

class Memory:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.memory_dir = os.path.join(root_dir, "memory")
        self.context_file = os.path.join(root_dir, "context.json")
        self.context: Dict = {}

        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)

    def load_context(self) -> None:
        """Loads short-term working memory."""
        if os.path.exists(self.context_file):
            try:
                with open(self.context_file, 'r') as f:
                    self.context = json.load(f)
            except json.JSONDecodeError:
                self.context = {}
        else:
            self.context = {}

    def save_context(self) -> None:
        """Saves short-term working memory."""
        with open(self.context_file, 'w') as f:
            json.dump(self.context, f, indent=2)

    def remember(self, fact: str, tag: str) -> None:
        """Saves a fact (markdown) to long-term memory."""
        filename = f"{tag}.md"
        # Sanitize filename
        filename = "".join([c for c in filename if c.isalnum() or c in "._-"])
        
        file_path = os.path.join(self.memory_dir, filename)
        
        # Append if exists, else create
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode) as f:
            f.write(f"\n{fact}\n")

    def recall(self, tag: str) -> str:
        """Retrieves facts matching a tag."""
        file_path = os.path.join(self.memory_dir, f"{tag}.md")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return f.read().strip()
        return ""

    def retrieve_full_state(self) -> str:
        """Returns a summary of current context and available memory tags."""
        self.load_context()
        
        # List available memories
        memories = [os.path.basename(f) for f in glob.glob(os.path.join(self.memory_dir, "*.md"))]
        
        summary = (
            f"=== CURRENT CONTEXT ===\n{json.dumps(self.context, indent=2)}\n\n"
            f"=== AVAILABLE MEMORIES ===\n{', '.join(memories)}"
        )
        return summary

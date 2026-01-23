"""
memory.py - The Hippocampus of Ralph.
Handles persistence of state (Context) and knowledge (Facts).
"""

import json
import os
import sys
import glob
from typing import Dict, Optional

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.abspath(__file__))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from vector_db import vector_memory

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
            
        # ALSO save to Vector DB
        vector_memory.add(fact, {"tag": tag, "source": "manual_memory"})

    def save(self, fact: str, tag: str = "general") -> None:
        """Alias for remember() - saves a fact to long-term memory.

        This alias exists because LLMs often guess 'save' as the method name.
        """
        self.remember(fact, tag)

    def get(self, tag: str) -> str:
        """Alias for recall() - retrieves facts by tag."""
        return self.recall(tag)

    def search(self, query: str) -> str:
        """Alias for recall_similar() - semantic search for facts."""
        return self.recall_similar(query)

    def recall(self, tag: str) -> str:
        """Retrieves facts matching a tag."""
        file_path = os.path.join(self.memory_dir, f"{tag}.md")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return f.read().strip()
        return ""
    
    def recall_similar(self, query: str) -> str:
        """Retrieves semantically similar facts/lessons."""
        results = vector_memory.search(query)
        if not results:
            return ""
        
        summary = "--- RELEVANT MEMORIES ---\n"
        for r in results:
            doc = r['document']
            summary += f"- [{r['score']:.2f}] {doc['text']} (Tag: {doc['metadata'].get('tag', 'unknown')})\n"
        return summary

    def retrieve_full_state(self) -> str:
        """Returns a summary of current context and available memory tags."""
        self.load_context()
        lessons = self.get_lessons()
        
        # List available memories
        memories = [os.path.basename(f) for f in glob.glob(os.path.join(self.memory_dir, "*.md"))]
        
        summary = (
            f"=== CURRENT CONTEXT ===\n{json.dumps(self.context, indent=2)}\n\n"
            f"=== LEARNED LESSONS ===\n{json.dumps(lessons, indent=2)}\n\n"
            f"=== AVAILABLE MEMORIES ===\n{', '.join(memories)}"
        )
        return summary

    def get_lessons(self) -> list:
        """Retrieves the list of learned lessons (Global)."""
        global_lessons_file = os.path.expanduser("~/.ralph/global_memory/lessons.json")
        if os.path.exists(global_lessons_file):
            try:
                with open(global_lessons_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def add_lesson(self, lesson: str) -> None:
        """Adds a new lesson to the global database."""
        lessons = self.get_lessons()
        if lesson not in lessons:
            lessons.append(lesson)
            
            global_lessons_file = os.path.expanduser("~/.ralph/global_memory/lessons.json")
            os.makedirs(os.path.dirname(global_lessons_file), exist_ok=True)
            
            with open(global_lessons_file, 'w') as f:
                json.dump(lessons, f, indent=2)
            
            # Index lesson in Vector DB (Global Scope)
            vector_memory.add(lesson, {"tag": "lesson", "source": "reflector"}, scope="global")

    def get_guidelines(self) -> list:
        """Retrieves the list of synthesized guidelines (Global).

        Guidelines are meta-rules created by the REM Sleep consolidation
        process from clusters of similar lessons.
        """
        global_guidelines_file = os.path.expanduser("~/.ralph/global_memory/global_guidelines.md")
        if not os.path.exists(global_guidelines_file):
            return []

        # Parse markdown to extract guidelines
        guidelines = []
        try:
            with open(global_guidelines_file, 'r') as f:
                content = f.read()

            # Extract guidelines (lines starting with "- **")
            for line in content.split('\n'):
                if line.strip().startswith('- **') and '**' in line[4:]:
                    # Extract text between ** markers
                    text = line.split('**')[1] if '**' in line else line
                    guidelines.append(text)
        except Exception as e:
            print(f"[Memory] Error loading guidelines: {e}")

        return guidelines

    def get_guidelines_detailed(self) -> list:
        """Retrieves guidelines with metadata from consolidation log."""
        log_file = os.path.expanduser("~/.ralph/global_memory/consolidation_log.json")
        if not os.path.exists(log_file):
            return []

        try:
            with open(log_file, 'r') as f:
                log = json.load(f)
            return log.get("events", [])
        except Exception:
            return []

    def add_guideline(self, guideline: str, category: str = "General", confidence: float = 0.8) -> None:
        """Adds a new guideline (typically called during REM Sleep).

        Note: This is a manual add - normally guidelines are created
        through the consolidation process in the sleeper agent.
        """
        global_guidelines_file = os.path.expanduser("~/.ralph/global_memory/global_guidelines.md")
        os.makedirs(os.path.dirname(global_guidelines_file), exist_ok=True)

        # Check if file exists, create header if not
        if not os.path.exists(global_guidelines_file):
            header = """# Ralph Global Guidelines

> Auto-generated meta-rules synthesized from learned lessons.
> Manual additions below.

"""
            with open(global_guidelines_file, 'w') as f:
                f.write(header)

        # Append new guideline
        from datetime import datetime
        entry = f"""
## {category}

- **{guideline}**
  - Source: manual
  - Confidence: {confidence}
  - Created: {datetime.now().isoformat()}

"""
        with open(global_guidelines_file, 'a') as f:
            f.write(entry)

        # Index in Vector DB
        vector_memory.add(
            guideline,
            {"tag": "guideline", "category": category, "source": "manual", "confidence": confidence},
            scope="global"
        )


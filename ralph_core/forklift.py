"""
Forklift Protocol - Intelligent Memory Loading for Ralph AI

The Forklift analyzes tasks and selectively retrieves relevant memories,
replacing the old "dump everything" approach with focused, efficient loading.

Part of the Ralph Compound architecture (Warehouse → Processing/HQ transfer).
"""

import os
import sys
import time
import glob
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.abspath(__file__))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from vector_db import vector_memory

if TYPE_CHECKING:
    from memory import Memory
    from protocols.messages import Message


@dataclass
class ForkliftConfig:
    """Configuration for Forklift retrieval."""
    max_lessons: int = 5
    max_facts: int = 3
    max_guidelines: int = 3  # NEW: Synthesized meta-rules from REM Sleep
    similarity_threshold: float = 0.6
    include_context_keys: list = field(default_factory=list)


# Preset configurations by scope
SCOPE_CONFIGS = {
    "minimal": ForkliftConfig(
        max_lessons=2,
        max_facts=1,
        max_guidelines=1,
        similarity_threshold=0.7,
    ),
    "standard": ForkliftConfig(
        max_lessons=5,
        max_facts=3,
        max_guidelines=3,
        similarity_threshold=0.6,
    ),
    "comprehensive": ForkliftConfig(
        max_lessons=10,
        max_facts=5,
        max_guidelines=5,
        similarity_threshold=0.4,
    ),
}

# Task type → relevant memory tags mapping
TASK_MEMORY_MAP = {
    "code": ["architecture", "patterns", "errors", "general"],
    "test": ["test_patterns", "coverage", "errors", "general"],
    "fix": ["errors", "debugging", "solutions", "general"],
    "refactor": ["architecture", "patterns", "performance", "general"],
    "regex": ["patterns", "regex_examples", "general"],
    "json": ["data_structures", "schemas", "general"],
    "sql": ["database", "queries", "schemas", "general"],
    "docs": ["documentation", "api", "architecture", "general"],
    "general": ["general", "lessons"],
}

# Context keys always included
ALWAYS_INCLUDE_CONTEXT = ["last_iteration", "last_objective", "last_verdict"]

# Task-specific context keys
TASK_CONTEXT_MAP = {
    "code": ["current_file", "language"],
    "test": ["test_coverage", "failing_tests"],
    "fix": ["error_message", "stack_trace"],
}


class Forklift:
    """
    Intelligent memory loader for Ralph agents.

    Replaces retrieve_full_state() with selective, task-aware loading.
    """

    def __init__(self, memory: "Memory"):
        self.memory = memory

    def lift(
        self,
        objective: str,
        task_type: str = "code",
        scope: str = "standard",
        hints: Optional[list] = None,
    ) -> dict:
        """
        Retrieve relevant memories for a task.

        Args:
            objective: The task/objective being worked on
            task_type: Type of task (code, test, fix, regex, etc.)
            scope: How much to retrieve (minimal, standard, comprehensive)
            hints: Optional hints about what's relevant (tags, files)

        Returns:
            Structured dict with lessons, facts, context, files
        """
        config = SCOPE_CONFIGS.get(scope, SCOPE_CONFIGS["standard"])

        result = {
            "lessons": [],
            "facts": [],
            "guidelines": [],  # NEW: Meta-rules from REM Sleep consolidation
            "context": {},
            "files": [],
        }

        # 1. Semantic search for relevant lessons
        result["lessons"] = self._get_relevant_lessons(objective, config)

        # 1b. NEW: Get relevant guidelines (higher-level rules from REM Sleep)
        result["guidelines"] = self._get_relevant_guidelines(objective, config)

        # 2. Get facts by relevant tags
        relevant_tags = TASK_MEMORY_MAP.get(task_type, TASK_MEMORY_MAP["general"])
        if hints:
            relevant_tags = list(set(relevant_tags + hints))
        result["facts"] = self._get_facts_by_tags(relevant_tags, config)

        # 3. Get relevant context fields
        self.memory.load_context()
        result["context"] = self._filter_context(self.memory.context, task_type)

        # 4. Suggest relevant files
        result["files"] = self._suggest_files(objective, task_type)

        return result

    def _get_relevant_lessons(self, objective: str, config: ForkliftConfig) -> list:
        """Retrieve semantically similar lessons from vector DB."""
        try:
            results = vector_memory.search(objective, top_k=config.max_lessons)
        except Exception:
            return []

        lessons = []
        for r in results:
            score = r.get("score", 0)
            if score >= config.similarity_threshold:
                doc = r.get("document", {})
                # Skip guidelines (handled separately)
                if doc.get("metadata", {}).get("tag") == "guideline":
                    continue
                lessons.append({
                    "text": doc.get("text", ""),
                    "score": round(score, 3),
                    "tag": doc.get("metadata", {}).get("tag", "unknown"),
                })
        return lessons

    def _get_relevant_guidelines(self, objective: str, config: ForkliftConfig) -> list:
        """
        Retrieve semantically similar guidelines from vector DB.

        Guidelines are meta-rules synthesized from lesson clusters during
        REM Sleep consolidation. They are higher-level than individual lessons.
        """
        try:
            # Search more broadly since guidelines are more general
            results = vector_memory.search(objective, top_k=config.max_guidelines * 2)
        except Exception:
            return []

        guidelines = []
        for r in results:
            score = r.get("score", 0)
            # Slightly lower threshold for guidelines since they're more general
            if score >= config.similarity_threshold - 0.1:
                doc = r.get("document", {})
                metadata = doc.get("metadata", {})

                # Only include actual guidelines
                if metadata.get("tag") != "guideline":
                    continue

                guidelines.append({
                    "text": doc.get("text", ""),
                    "score": round(score, 3),
                    "category": metadata.get("category", "General"),
                    "confidence": metadata.get("confidence", 0.8),
                })

                if len(guidelines) >= config.max_guidelines:
                    break

        return guidelines

    def _get_facts_by_tags(self, tags: list, config: ForkliftConfig) -> list:
        """Retrieve facts from memory/*.md files by relevant tags."""
        facts = []

        for tag in tags[:config.max_facts + 2]:  # Check a few extra in case some are empty
            if len(facts) >= config.max_facts:
                break

            content = self.memory.recall(tag)
            if content:
                facts.append({
                    "tag": tag,
                    "content": content[:500] if len(content) > 500 else content,
                })
        return facts

    def _filter_context(self, context: dict, task_type: str) -> dict:
        """Filter context.json to relevant fields only."""
        if not context:
            return {}

        # Combine always-include keys with task-specific keys
        relevant_keys = set(ALWAYS_INCLUDE_CONTEXT)
        relevant_keys.update(TASK_CONTEXT_MAP.get(task_type, []))

        return {k: v for k, v in context.items() if k in relevant_keys}

    def _suggest_files(self, objective: str, task_type: str) -> list:
        """Suggest relevant files based on task type and objective keywords."""
        suggestions = []
        obj_lower = objective.lower()

        # Task-type based suggestions
        if task_type == "test" or "test" in obj_lower:
            suggestions.extend(["tests/", "test_*.py", "*_test.py"])
        if task_type == "sql" or "database" in obj_lower:
            suggestions.extend(["models/", "db/", "migrations/"])
        if "api" in obj_lower:
            suggestions.extend(["api/", "routes/", "endpoints/"])
        if "auth" in obj_lower:
            suggestions.extend(["auth/", "security/"])

        return suggestions[:5]


def handle_forklift_request(message: "Message") -> "Message":
    """
    Handle FORKLIFT_REQUEST and return FORKLIFT_RESPONSE.

    For use with the MessageBus.
    """
    from memory import Memory
    from protocols.messages import MessageType, Message as Msg, forklift_response

    start_time = time.time()

    payload = message.payload
    objective = payload.get("objective", "")
    task_type = payload.get("task_type", "code")
    scope = payload.get("scope", "standard")
    hints = payload.get("hints", [])
    requester = payload.get("requester", "unknown")

    # Initialize forklift with memory
    memory = Memory()
    forklift = Forklift(memory)

    # Lift the relevant memories
    memories = forklift.lift(
        objective=objective,
        task_type=task_type,
        scope=scope,
        hints=hints,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Calculate aggregate scores
    lesson_scores = [l["score"] for l in memories["lessons"]]
    avg_lesson_score = sum(lesson_scores) / len(lesson_scores) if lesson_scores else 0

    return forklift_response(
        memories=memories,
        scores={
            "lessons": round(avg_lesson_score, 3),
            "facts": len(memories["facts"]) / 5,  # Normalized 0-1
        },
        retrieval_time_ms=elapsed_ms,
        requester=requester,
    )


def forklift_lift_sync(
    memory: "Memory",
    objective: str,
    task_type: str = "code",
    scope: str = "standard",
) -> dict:
    """
    Synchronous forklift call for non-message-bus code paths.

    Use this in runner.py or other places where you need forklift
    but aren't using the message bus.

    Args:
        memory: Memory instance
        objective: Task objective
        task_type: Type of task
        scope: Retrieval scope

    Returns:
        Structured memory dict
    """
    forklift = Forklift(memory)
    return forklift.lift(objective, task_type, scope)


def format_forklift_context(
    memories: dict,
    files: list,
    iteration: str,
    plan_summary: str,
    project_context: str = "",
    diagnosis_context: str = "",
) -> str:
    """
    Format Forklift output into an agent-readable context string.

    This replaces the old ad-hoc context assembly in runner.py.
    """
    parts = []

    # Header
    parts.append(f"Current Directory Files: {files}")
    parts.append(f"Iteration: {iteration}")

    # Plan status
    if plan_summary:
        parts.append(f"--- PLAN STATUS ---\n{plan_summary}")

    # Project docs
    if project_context:
        parts.append(project_context)

    # Forklift-loaded memories
    # Guidelines first (higher priority - synthesized meta-rules)
    if memories.get("guidelines"):
        guidelines_str = "\n".join(
            f"- [{g['score']:.2f}] [{g['category']}] {g['text']}"
            for g in memories["guidelines"]
        )
        parts.append(f"--- GUIDELINES (Meta-Rules) ---\n{guidelines_str}")

    if memories.get("lessons"):
        lessons_str = "\n".join(
            f"- [{l['score']:.2f}] {l['text']} (Tag: {l['tag']})"
            for l in memories["lessons"]
        )
        parts.append(f"--- RELEVANT LESSONS ---\n{lessons_str}")

    if memories.get("facts"):
        facts_str = "\n".join(
            f"[{f['tag']}]: {f['content']}"
            for f in memories["facts"]
        )
        parts.append(f"--- RELEVANT FACTS ---\n{facts_str}")

    if memories.get("context"):
        import json
        parts.append(f"--- WORKING CONTEXT ---\n{json.dumps(memories['context'], indent=2)}")

    if memories.get("files"):
        parts.append(f"--- SUGGESTED FILES ---\n{', '.join(memories['files'])}")

    # Debugger diagnosis
    if diagnosis_context:
        parts.append(diagnosis_context)

    return "\n\n".join(parts)

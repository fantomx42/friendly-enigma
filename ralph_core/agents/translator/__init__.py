"""
Translator Agent - Human Input to LLM-Optimized Format

The Translator is the first tier of the Ralph AI hierarchy. It converts
ambiguous human requests into structured TaskSpec objects that downstream
agents (Orchestrator, Engineer, Designer) can process more efficiently.

Key Responsibilities:
- Parse human ambiguity and extract intent
- Expand shorthand into explicit requirements
- Identify task type for ASIC routing
- Extract relevant context files
- Output structured TaskSpec

Example:
    Input:  "make the login faster"
    Output: TaskSpec(
        objective="Optimize login performance",
        requirements=["Reduce latency", "Target <100ms response time"],
        preferences=["Maintain backward compatibility"],
        task_type="refactor",
        context_files=["auth/login.py", "auth/session.py"]
    )
"""

from .agent import translate, TaskSpec

__all__ = ["translate", "TaskSpec"]

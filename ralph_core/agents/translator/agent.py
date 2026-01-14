"""
Translator Agent Implementation

Converts human input into structured TaskSpec using phi3:mini (or fallback).
This is the entry point for all human objectives into the Ralph AI system.
"""

import json
import re
import requests
from dataclasses import dataclass, asdict
from typing import Optional

# Configuration
TRANSLATOR_MODEL = "phi3:mini"
FALLBACK_MODEL = "deepseek-r1:14b"
OLLAMA_API = "http://localhost:11434/api/generate"


@dataclass
class TaskSpec:
    """
    Structured specification of a task, translated from human input.

    Attributes:
        objective: Clear, explicit goal statement
        requirements: Must-have constraints (non-negotiable)
        preferences: Nice-to-have features (can be deprioritized)
        task_type: Category for ASIC routing (code, test, refactor, docs, etc.)
        context_files: Relevant files to examine
        complexity: Estimated complexity (low, medium, high)
        original_input: The raw human input (for reference)
    """
    objective: str
    requirements: list[str]
    preferences: list[str]
    task_type: str
    context_files: list[str]
    complexity: str
    original_input: str

    def to_prompt_context(self) -> str:
        """Convert TaskSpec to a formatted string for downstream agents."""
        return f"""=== TRANSLATED TASK SPECIFICATION ===
OBJECTIVE: {self.objective}

REQUIREMENTS (Must-Have):
{chr(10).join(f"  - {r}" for r in self.requirements)}

PREFERENCES (Nice-to-Have):
{chr(10).join(f"  - {p}" for p in self.preferences)}

TASK TYPE: {self.task_type}
COMPLEXITY: {self.complexity}

CONTEXT FILES:
{chr(10).join(f"  - {f}" for f in self.context_files) if self.context_files else "  (none specified)"}

ORIGINAL REQUEST: "{self.original_input}"
=== END SPECIFICATION ==="""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


SYSTEM_PROMPT = """You are a Translator AI. Your job is to convert human requests into structured task specifications.

Given a human request, output a JSON object with these fields:
- objective: A clear, explicit goal statement (expand any shorthand)
- requirements: List of must-have constraints (things that MUST be true)
- preferences: List of nice-to-have features (can be deprioritized if needed)
- task_type: One of: code, test, refactor, docs, fix, regex, json, sql, config, deploy, other
- context_files: List of likely relevant file paths (can be empty if unclear)
- complexity: One of: low, medium, high

Rules:
1. ALWAYS expand shorthand: "make it faster" -> "Optimize for performance, reduce latency"
2. ALWAYS extract implicit requirements: "add login" implies security, validation, error handling
3. If task_type is unclear, default to "code"
4. For context_files, make educated guesses based on common patterns (auth/, api/, etc.)
5. Output ONLY valid JSON, no other text

Example Input: "fix the email bug"
Example Output:
{
  "objective": "Fix the bug affecting email functionality",
  "requirements": ["Identify root cause", "Ensure email delivery works", "Add error handling"],
  "preferences": ["Add logging for debugging", "Write regression test"],
  "task_type": "fix",
  "context_files": ["email/", "utils/mailer.py", "tests/test_email.py"],
  "complexity": "medium"
}"""


def _check_model_available(model: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return model in models or any(m.startswith(model.split(":")[0]) for m in models)
    except Exception:
        pass
    return False


def _call_translator(prompt: str) -> str:
    """Call the translator model."""
    model = TRANSLATOR_MODEL if _check_model_available(TRANSLATOR_MODEL) else FALLBACK_MODEL

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": f"{SYSTEM_PROMPT}\n\nHuman Request: {prompt}\n\nJSON Output:",
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low for structured output
                    "num_predict": 1024,
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            return None

        return response.json().get("response", "").strip()

    except Exception as e:
        print(f"Translator error: {e}")
        return None


def _parse_json_response(response: str) -> Optional[dict]:
    """Extract JSON from model response, handling common formatting issues."""
    if not response:
        return None

    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON in code block
    code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _create_fallback_spec(human_input: str) -> TaskSpec:
    """Create a basic TaskSpec when translation fails."""
    # Simple heuristics for task type
    input_lower = human_input.lower()

    if any(word in input_lower for word in ["fix", "bug", "error", "broken"]):
        task_type = "fix"
    elif any(word in input_lower for word in ["test", "unittest", "pytest"]):
        task_type = "test"
    elif any(word in input_lower for word in ["refactor", "clean", "optimize", "faster"]):
        task_type = "refactor"
    elif any(word in input_lower for word in ["doc", "readme", "comment"]):
        task_type = "docs"
    elif any(word in input_lower for word in ["regex", "pattern", "match"]):
        task_type = "regex"
    elif any(word in input_lower for word in ["json", "parse", "serialize"]):
        task_type = "json"
    elif any(word in input_lower for word in ["sql", "query", "database"]):
        task_type = "sql"
    else:
        task_type = "code"

    return TaskSpec(
        objective=human_input,
        requirements=["Complete the requested task"],
        preferences=[],
        task_type=task_type,
        context_files=[],
        complexity="medium",
        original_input=human_input
    )


def translate(human_input: str, project_context: str = "") -> TaskSpec:
    """
    Translate human input into a structured TaskSpec.

    Args:
        human_input: Raw human request (e.g., "make login faster")
        project_context: Optional context about the project (file list, etc.)

    Returns:
        TaskSpec: Structured specification for downstream agents
    """
    # Build prompt with optional context
    prompt = human_input
    if project_context:
        prompt = f"Project Context:\n{project_context}\n\nRequest: {human_input}"

    # Call translator
    response = _call_translator(prompt)

    # Parse response
    parsed = _parse_json_response(response)

    if not parsed:
        # Fallback to heuristics
        return _create_fallback_spec(human_input)

    # Build TaskSpec from parsed JSON
    return TaskSpec(
        objective=parsed.get("objective", human_input),
        requirements=parsed.get("requirements", ["Complete the task"]),
        preferences=parsed.get("preferences", []),
        task_type=parsed.get("task_type", "code"),
        context_files=parsed.get("context_files", []),
        complexity=parsed.get("complexity", "medium"),
        original_input=human_input
    )


# Quick test
if __name__ == "__main__":
    test_inputs = [
        "fix the login bug",
        "make it faster",
        "add email validation",
        "write tests for the API",
    ]

    for inp in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: {inp}")
        spec = translate(inp)
        print(spec.to_prompt_context())

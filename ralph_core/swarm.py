"""
swarm.py

The Core Swarm Interface for the Ralph Protocol.
This module abstracts the multi-model architecture into a single callable interface.

Roles:
- Orchestrator (DeepSeek-R1-14B): Strategic planning and reasoning.
- Engineer (Qwen2.5-Coder-14B): Code implementation and technical execution.
- Designer (Mistral-Nemo-12B): Critique, architectural review, and safety.
"""

import requests
import json
import subprocess
from typing import Dict, Any

# --- Configuration ---
MODELS = {
    "orchestrator": "deepseek-r1:14b",
    "engineer": "qwen2.5-coder:14b",
    "designer": "mistral-nemo:12b"
}

OLLAMA_API = "http://localhost:11434/api/generate"

# --- Core Interface ---

def call_model(role: str, prompt: str, stream: bool = False) -> str:
    """
    Generic handler to call a specific role in the swarm.
    """
    model_name = MODELS.get(role)
    if not model_name:
        raise ValueError(f"Unknown role: {role}")

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7 if role == "orchestrator" else 0.2  # Low temp for coding
                }
            },
            timeout=600  # DeepSeek reasoning can take time
        )
        if response.status_code == 404:
            return f"Error: Model {model_name} not found. Run 'ollama pull {model_name}'"
        
        response.raise_for_status()
        
        if stream:
            # TODO: Implement generator for streaming if needed
            return "Streaming not yet implemented in swarm.py"
        
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"CRITICAL SWARM ERROR ({role}): {str(e)}"

# --- Specialized Functions ---

def think(context: str, goal: str) -> str:
    """
    The Orchestrator's step. Produces a logical plan/reasoning trace.
    """
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"GOAL: {goal}\n\n"
        f"You are the Orchestrator. Plan the solution. "
        f"Note: You have an ENGINEER who can write code and an EXECUTOR who can run shell commands. "
        f"Plan to use <execute>command</execute> to verify your work."
    )
    return call_model("orchestrator", prompt)

def code(plan: str, context: str) -> str:
    """
    The Engineer's step. Produces executable code based on the plan.
    """
    prompt = (
        f"PLAN:\n{plan}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"You are the Engineer. Implement the plan. "
        f"RULES:\n"
        f"1. To CREATE a file, use this format:\n"
        f"```python:path/to/filename.py\ncode_here\n```\n"
        f"2. DO NOT write a python script to create other files. Output the file content directly.\n"
        f"3. To RUN a command (like installing deps or testing), use:\n"
        f"<execute>command_here</execute>\n"
    )
    return call_model("engineer", prompt)

def review(code: str, plan: str) -> str:
    """
    The Designer's step. Critiques and polishes the output.
    """
    prompt = (
        f"PLAN:\n{plan}\n\n"
        f"CODE:\n{code}\n\n"
        f"You are the Designer. Review the work. "
        f"CRITICAL: The system only saves files that YOU output in your review. "
        f"You MUST reproduce the full code blocks for every file, even if they are correct.\n"
        f"Format: ```python:filename.py\n...```\n"
        f"If a command was executed, check its output. "
        f"Only output <promise>COMPLETE</promise> if the objective is met AND verified."
    )
    return call_model("designer", prompt)

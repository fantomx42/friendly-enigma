#!/usr/bin/env python3
"""
Ralph - Single model architecture with Wheeler Memory integration.

Uses qwen3-coder-next for all cognitive tasks (translation, planning,
coding, review) via ultra-sparse MoE (80B total, 3B active per pass)
with 256K context window.

Replaces the multi-agent hierarchy with a unified loop:
  objective -> wheeler recall -> single LLM call -> store -> repeat

See /docs/SCM_AXIOMS.md for the theoretical foundation of
stability-weighted context budgeting.
"""

import subprocess
import sys
import os
import hashlib

# Ensure ralph_core is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ralph_core"))

from wheeler_weights import stability_tracker, PatternMetrics
from context_budget import get_weighted_context, WeightedPattern, estimate_tokens

MODEL = "qwen3-coder-next"
MAX_ITERATIONS = 10
COMPLETION_SIGNAL = "<promise>COMPLETE</promise>"


def call_ollama(prompt: str) -> str:
    """Single model call - no agent switching overhead."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Model call timed out after 300s"
    except FileNotFoundError:
        return "[ERROR] ollama not found - install from https://ollama.com"


def wheeler_recall(query: str) -> list:
    """
    Get relevant patterns from Wheeler Memory with stability scores.

    Returns list of WeightedPattern instances suitable for
    get_weighted_context().
    """
    patterns = []

    # Try Wheeler spatial memory
    try:
        from wheeler import WheelerMemoryBridge
        bridge = WheelerMemoryBridge()
        raw_result = bridge.recall(query)
        if raw_result:
            # Parse Wheeler output lines
            for line in raw_result.strip().split("\n"):
                line = line.strip("- ")
                if not line or line.startswith("---"):
                    continue

                # Extract score and text from format: [0.85] some text
                if line.startswith("[") and "]" in line:
                    bracket_end = line.index("]")
                    try:
                        score = float(line[1:bracket_end])
                    except ValueError:
                        score = 0.5
                    text = line[bracket_end + 1:].strip()
                else:
                    score = 0.5
                    text = line

                if text:
                    pid = hashlib.md5(text.encode()).hexdigest()[:12]
                    # Record the hit and get stability metrics
                    metrics = stability_tracker.record_hit(pid, text)
                    patterns.append(WeightedPattern(
                        text=text,
                        stability_score=metrics.stability_score,
                        pattern_id=pid,
                        source="wheeler",
                    ))
    except ImportError:
        pass

    # Try vector DB memories
    try:
        from vector_db import vector_memory
        results = vector_memory.search(query, top_k=5)
        for r in results:
            doc = r.get("document", {})
            text = doc.get("text", "")
            if text:
                pid = hashlib.md5(text.encode()).hexdigest()[:12]
                metrics = stability_tracker.record_hit(pid, text)
                patterns.append(WeightedPattern(
                    text=text,
                    stability_score=metrics.stability_score,
                    pattern_id=pid,
                    source="vector_db",
                ))
    except Exception:
        pass

    # Flush updated metrics to disk
    stability_tracker.flush()

    return patterns


def wheeler_store(context: str, response: str, success: bool):
    """Store iteration result and update stability tracking."""
    # Store in Wheeler Memory
    try:
        from wheeler import WheelerMemoryBridge
        bridge = WheelerMemoryBridge()
        # Store a summary of what happened
        summary = f"Objective: {context[:200]}\nResult: {'SUCCESS' if success else 'INCOMPLETE'}\n{response[:300]}"
        bridge.remember(summary)
    except ImportError:
        pass

    # Record context switch for stability tracking
    stability_tracker.record_context_switch()
    stability_tracker.flush()


def build_prompt(objective: str, history: list, wheeler_context: str) -> str:
    """
    Single unified prompt - the model handles all cognitive tasks internally.

    With qwen3-coder-next's 256K context and MoE architecture, a single
    model can handle translation, planning, coding, and review without
    the overhead of inter-agent message passing.
    """
    history_str = "\n".join(history[-3:]) if history else "(first iteration)"

    return f"""You are Ralph, an autonomous AI that accomplishes tasks through iterative refinement.

RELEVANT PATTERNS FROM WHEELER MEMORY (weighted by stability):
{wheeler_context if wheeler_context else "(no prior patterns)"}

OBJECTIVE: {objective}

PREVIOUS ITERATIONS:
{history_str}

Instructions:
1. Analyze the objective and any prior attempts
2. Plan your approach
3. Execute - produce working code/solution
4. Self-review for correctness
5. Output {COMPLETION_SIGNAL} when genuinely complete, or explain what needs fixing

Response:"""


def ralph_loop(objective: str) -> bool:
    """
    Main Ralph execution loop.

    Single-model architecture: one call per iteration handles all
    cognitive work. Wheeler Memory provides stability-weighted context.
    """
    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n>>> Iteration {iteration}/{MAX_ITERATIONS}")

        # Recall from Wheeler Memory with stability weights
        patterns = wheeler_recall(objective)
        wheeler_context = get_weighted_context(patterns, max_tokens=2000)

        # Single model call handles everything
        prompt = build_prompt(objective, history, wheeler_context)
        response = call_ollama(prompt)

        print(response)
        history.append(f"[Iteration {iteration}]: {response[:500]}...")

        # Store experience with success flag
        success = COMPLETION_SIGNAL in response
        wheeler_store(objective, response, success)

        if success:
            print(f"\nComplete after {iteration} iterations")
            return True

    print(f"\nMax iterations ({MAX_ITERATIONS}) reached")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ralph_simple.py 'your objective'")
        print()
        print("Single-model Ralph using qwen3-coder-next with Wheeler Memory.")
        print("See /docs/SCM_AXIOMS.md for stability theory.")
        sys.exit(1)
    ralph_loop(" ".join(sys.argv[1:]))

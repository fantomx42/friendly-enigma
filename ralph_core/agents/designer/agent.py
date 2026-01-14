import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from ..common.llm import call_model
from typing import Optional
import json
import re


def review(code: str, plan: str) -> str:
    """
    The Designer's step. Critiques and polishes the output.
    (Original function - kept for backward compatibility)
    """
    prompt = (
        f"PLAN:\n{plan}\n\n"
        f"ENGINEER OUTPUT:\n{code}\n\n"
        f"You are the Designer. Review and output the final result.\n"
        f"1. IF CODE was generated: You MUST reproduce the full code blocks for every file. Verify correctness.\n"
        f"2. IF TEXT was generated (e.g. a haiku or report): Output ONLY the final polished text. **CRITICAL: DO NOT add reviews, descriptions, or meta-talk like 'The orchestrator has successfully...'. Just the content.**\n"
        f"3. STATUS: Only output <promise>COMPLETE</promise> if the objective is met.\n"
    )
    return call_model("designer", prompt)


def verify(objective: str, final_output: str, execution_summary: str) -> str:
    """
    The Verifier's step. Strictly checks if the task is done.
    (Original function - kept for backward compatibility)
    """
    prompt = (
        f"OBJECTIVE: {objective}\n"
        f"AGENT OUTPUT: {final_output}\n"
        f"EXECUTION RESULTS: {execution_summary}\n\n"
        f"You are the Verifier. Did the agent complete the objective?\n"
        f"CRITERIA:\n"
        f"1. IF the task was to write text (haiku, joke, etc.) and the AGENT OUTPUT contains it -> PASS.\n"
        f"2. IF the task required code and the EXECUTION RESULTS show success -> PASS.\n"
        f"3. IF the task failed or is missing -> FAIL.\n\n"
        f"DECISION:\n"
        f"If PASS, output exactly: <promise>COMPLETE</promise>\n"
        f"If FAIL, explain what is missing."
    )
    return call_model("designer", prompt)


def review_with_feedback(
    code: str,
    plan: str,
    revision_round: int = 0,
) -> dict:
    """
    Designer review that returns structured feedback for potential revision.

    Args:
        code: Engineer's output
        plan: Original plan
        revision_round: Current revision round (0 = initial review)

    Returns:
        Dict with:
        - 'approved': bool - Whether output is acceptable
        - 'issues': list[str] - Problems found (if not approved)
        - 'suggestions': list[str] - Improvement suggestions
        - 'final_output': str - Polished output (if approved)
        - 'severity': str - How serious the issues are (low/medium/high)
    """
    prompt = (
        f"PLAN:\n{plan}\n\n"
        f"ENGINEER OUTPUT:\n{code}\n\n"
        f"REVISION ROUND: {revision_round}\n\n"
        f"You are the Designer. Analyze this output and provide structured feedback.\n\n"
        f"Output a JSON object with:\n"
        f'{{"approved": true/false, "issues": ["issue1", "issue2"], '
        f'"suggestions": ["suggestion1"], "severity": "low/medium/high", '
        f'"final_output": "polished output if approved"}}\n\n'
        f"RULES:\n"
        f"1. Set approved=true ONLY if the output fully meets the plan requirements\n"
        f"2. List ALL issues found, even minor ones\n"
        f"3. severity: 'high' if broken/wrong, 'medium' if incomplete, 'low' if style issues\n"
        f"4. If approved, include the polished final_output\n"
        f"5. Output ONLY valid JSON, nothing else\n"
    )

    response = call_model("designer", prompt)

    # Parse JSON response
    try:
        # Try direct parse
        result = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if not result:
        # Fallback: assume approved if no structured feedback
        return {
            "approved": True,
            "issues": [],
            "suggestions": [],
            "severity": "low",
            "final_output": code,
        }

    return {
        "approved": result.get("approved", True),
        "issues": result.get("issues", []),
        "suggestions": result.get("suggestions", []),
        "severity": result.get("severity", "low"),
        "final_output": result.get("final_output", code),
    }


def evaluate_options(options: list[str], task_description: str) -> dict:
    """
    Evaluate multiple options from ASICs and select the best one.

    Args:
        options: List of candidate solutions
        task_description: What the task was trying to accomplish

    Returns:
        Dict with:
        - 'selected_index': int - Best option index
        - 'reasoning': str - Why this was selected
        - 'scores': list[float] - Score for each option (0-1)
    """
    options_text = "\n\n".join(
        f"=== OPTION {i+1} ===\n{opt}\n=== END OPTION {i+1} ==="
        for i, opt in enumerate(options)
    )

    prompt = (
        f"TASK: {task_description}\n\n"
        f"OPTIONS:\n{options_text}\n\n"
        f"You are the Designer. Evaluate these options and select the best one.\n"
        f"Output JSON: {{"
        f'"selected_index": 0-based index of best option, '
        f'"reasoning": "brief explanation", '
        f'"scores": [0.0-1.0 score for each option]}}\n'
        f"Output ONLY valid JSON."
    )

    response = call_model("designer", prompt)

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if not result:
        # Fallback: select first option
        return {
            "selected_index": 0,
            "reasoning": "Default selection (parsing failed)",
            "scores": [0.5] * len(options),
        }

    return {
        "selected_index": result.get("selected_index", 0),
        "reasoning": result.get("reasoning", ""),
        "scores": result.get("scores", [0.5] * len(options)),
    }


def handle_message(message: "Message") -> "Message":
    """
    Handle an incoming message and produce a response.

    Args:
        message: Incoming Message object

    Returns:
        Response Message object
    """
    from protocols.messages import (
        Message, MessageType,
        revision_request, complete_message, error_message
    )

    msg_type = message.type
    payload = message.payload

    if msg_type == MessageType.CODE_OUTPUT:
        # Review code from Engineer
        code = payload.get("code", "")
        notes = payload.get("notes", "")

        # Get plan from metadata or notes
        plan = message.metadata.get("plan", notes)

        # Perform structured review
        review_result = review_with_feedback(code, plan)

        if review_result["approved"]:
            # Approved - send COMPLETE
            return complete_message(
                final_output=review_result["final_output"],
                summary="Review passed",
            )
        else:
            # Not approved - send REVISION_REQUEST
            round_num = message.metadata.get("revision_round", 0) + 1

            return revision_request(
                issues=review_result["issues"],
                suggestions=review_result["suggestions"],
                severity=review_result["severity"],
                round_number=round_num,
            )

    elif msg_type == MessageType.OPTIONS:
        # Evaluate ASIC options
        options = payload.get("options", [])
        task_type = payload.get("task_type", "unknown")

        eval_result = evaluate_options(options, f"ASIC task: {task_type}")

        # Return evaluation (Orchestrator or Engineer will decide next step)
        return Message(
            type=MessageType.EVALUATION,
            sender="designer",
            receiver="orchestrator",
            payload={
                "selected_index": eval_result["selected_index"],
                "selected_option": options[eval_result["selected_index"]] if options else "",
                "reasoning": eval_result["reasoning"],
                "scores": eval_result["scores"],
            }
        )

    else:
        return error_message(
            error=f"Designer cannot handle message type: {msg_type}",
            recoverable=True,
        )


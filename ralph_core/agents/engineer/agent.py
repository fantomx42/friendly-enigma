import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from ..common.llm import call_model
from typing import Optional


def code(plan: str, context: str) -> str:
    """
    The Engineer's step. Produces executable code based on the plan.
    (Original function - kept for backward compatibility)
    """
    prompt = (
        f"PLAN:\n{plan}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"You are the Engineer. Execute the plan.\n"
        f"MODES:\n"
        f"1. CODE MODE: If the plan needs software, create files using:\n"
        f"```python:path/to/filename.py\ncode_here\n```\n"
        f"   And verify with <execute>command</execute>\n"
        f"2. TEXT MODE: If the plan is for a poem, explanation, or direct answer, just WRITE THE TEXT content directly. **CRITICAL: DO NOT add any intro, outro, or meta-commentary. Just the content.**\n"
    )
    return call_model("engineer", prompt)


def code_with_revision(
    plan: str,
    context: str,
    revision_feedback: Optional[str] = None,
    revision_round: int = 0,
) -> str:
    """
    Engineer step with support for revision feedback from Designer.

    Args:
        plan: The strategic plan from Orchestrator
        context: Project context and relevant files
        revision_feedback: Feedback from Designer's review (if revision)
        revision_round: Which revision round this is (0 = initial)

    Returns:
        Generated code/text output
    """
    prompt_parts = [
        f"PLAN:\n{plan}\n",
        f"CONTEXT:\n{context}\n",
    ]

    if revision_feedback and revision_round > 0:
        prompt_parts.append(
            f"\n=== REVISION REQUEST (Round {revision_round}) ===\n"
            f"The Designer has requested changes:\n{revision_feedback}\n"
            f"=== END REVISION REQUEST ===\n"
            f"\nAddress ALL issues mentioned above. Provide the complete corrected solution.\n"
        )

    prompt_parts.append(
        f"\nYou are the Engineer. Execute the plan.\n"
        f"MODES:\n"
        f"1. CODE MODE: If the plan needs software, create files using:\n"
        f"```python:path/to/filename.py\ncode_here\n```\n"
        f"   And verify with <execute>command</execute>\n"
        f"2. TEXT MODE: If the plan is for a poem, explanation, or direct answer, "
        f"just WRITE THE TEXT content directly. No intro/outro.\n"
    )

    return call_model("engineer", "\n".join(prompt_parts))


def code_with_asic(
    plan: str,
    context: str,
    task_type: str,
    micro_task: str,
) -> dict:
    """
    Engineer step that spawns an ASIC for a micro-task.

    Args:
        plan: The strategic plan
        context: Project context
        task_type: ASIC type (regex, json, test, etc.)
        micro_task: Specific micro-task for the ASIC

    Returns:
        Dict with 'options' (from ASIC) and 'selected' (Engineer's choice)
    """
    # Import here to avoid circular imports
    try:
        from ...asic import spawn_asic
    except ImportError:
        # Fallback if ASIC module not available
        return {
            "options": [code(plan, context)],
            "selected": 0,
            "asic_used": False,
        }

    # Spawn ASIC for micro-task
    options = spawn_asic(task_type, micro_task)

    if not options or all("Error" in opt for opt in options):
        # ASIC failed, fall back to main engineer
        return {
            "options": [code(plan, context)],
            "selected": 0,
            "asic_used": False,
        }

    # Engineer selects best option (for now, just pick first valid one)
    # In future, could use another LLM call to evaluate options
    selected_idx = 0
    for i, opt in enumerate(options):
        if "Error" not in opt and opt.strip():
            selected_idx = i
            break

    return {
        "options": options,
        "selected": selected_idx,
        "asic_used": True,
        "task_type": task_type,
    }


# ASIC-eligible task types that can be delegated to specialists
ASIC_ELIGIBLE_TYPES = {"regex", "json", "sql", "test", "fix", "docstring", "tiny_code", "small_code"}


def handle_message(message: "Message") -> "Message":
    """
    Handle an incoming message and produce a response.

    The Engineer can delegate to ASICs for certain task types by returning
    an ASIC_REQUEST message instead of doing the work directly.

    Args:
        message: Incoming Message object

    Returns:
        Response Message object (could be CODE_OUTPUT or ASIC_REQUEST)
    """
    from protocols.messages import Message, MessageType, code_output, asic_request

    msg_type = message.type
    payload = message.payload

    if msg_type == MessageType.WORK_REQUEST:
        # Initial work request
        plan = payload.get("plan", "")
        context = payload.get("context", "")
        task_spec = payload.get("task_spec", {})
        task_type = task_spec.get("task_type", "code")

        # Check if we should delegate to an ASIC
        use_asic = message.metadata.get("use_asic", True)  # Default: try ASICs
        if use_asic and task_type in ASIC_ELIGIBLE_TYPES:
            print(f"[Engineer] Delegating to ASIC:{task_type}")
            objective = task_spec.get("objective", plan[:200])

            # Return ASIC_REQUEST - the bus will route it
            req = asic_request(
                task_type=task_type,
                prompt=f"{objective}\n\nContext:\n{context[:500]}",
                parallel=True,  # Use parallel for speed
                return_to="engineer",  # Send response back to us
            )
            # Store plan/context in metadata for when we get ASIC_RESPONSE
            req.metadata["original_plan"] = plan
            req.metadata["original_context"] = context
            req.metadata["task_spec"] = task_spec
            return req

        # Otherwise, do the work ourselves
        result = code(plan, context)

        return code_output(
            code=result,
            files_created=[],  # Would need to parse from result
            execute_commands=[],  # Would need to parse from result
            notes=f"TaskSpec: {task_type}",
        )

    elif msg_type == MessageType.ASIC_RESPONSE:
        # ASIC returned options - select best and wrap as code_output
        options = payload.get("options", [])
        task_type = payload.get("task_type", "unknown")
        model_used = payload.get("model_used", "")

        print(f"[Engineer] Received {len(options)} options from ASIC:{task_type}")

        if not options:
            # ASIC failed, fall back to doing it ourselves
            original_plan = message.metadata.get("original_plan", "")
            original_context = message.metadata.get("original_context", "")
            result = code(original_plan, original_context)
        else:
            # Select best option (first non-error one)
            result = options[0]
            for opt in options:
                if not opt.startswith("Error:") and not opt.startswith("ASIC ERROR:"):
                    result = opt
                    break

            # If task_spec suggests wrapping in code block, do so
            task_spec = message.metadata.get("task_spec", {})
            if task_type in {"regex", "json", "sql"}:
                # Wrap in appropriate code block
                lang = {"regex": "python", "json": "json", "sql": "sql"}.get(task_type, "text")
                result = f"```{lang}\n{result}\n```"

        return code_output(
            code=result,
            notes=f"ASIC:{task_type} (model: {model_used})",
        )

    elif msg_type == MessageType.REVISION_REQUEST:
        # Revision from Designer
        issues = payload.get("issues", [])
        round_num = payload.get("round", 1)

        # Get original context from correlation
        original_plan = payload.get("original_plan", "")
        original_context = payload.get("original_context", "")

        revision_feedback = "\n".join(f"- {issue}" for issue in issues)

        result = code_with_revision(
            plan=original_plan,
            context=original_context,
            revision_feedback=revision_feedback,
            revision_round=round_num,
        )

        return code_output(
            code=result,
            notes=f"Revision round {round_num}",
        )

    else:
        # Unknown message type
        from protocols.messages import error_message
        return error_message(
            error=f"Engineer cannot handle message type: {msg_type}",
            recoverable=True,
        )


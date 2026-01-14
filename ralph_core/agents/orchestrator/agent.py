import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from ..common.llm import call_model
from typing import Optional


def think(context: str, goal: str) -> str:
    """
    The Orchestrator's step. Produces a logical plan/reasoning trace.
    """
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"GOAL: {goal}\n\n"
        f"You are the Orchestrator. Plan the solution.\n"
        f"OPTIONS:\n"
        f"1. IF the goal requires code/actions: Plan to use the ENGINEER to write code and <execute>command</execute> to verify.\n"
        f"2. IF the goal is 'Status' or 'System Check': Plan to output the content of `STATUS.md` directly from the context. Do not summarize.\n"
        f"3. IF the goal is a direct question/creative task (e.g. 'write a haiku', 'explain X'): Plan to simply provide the text answer directly.\n"
    )
    return call_model("orchestrator", prompt)

def decompose(objective: str) -> str:
    """
    Breaks a complex objective into a sequential list of subtasks.
    """
    prompt = (
        f"OBJECTIVE: {objective}\n\n"
        f"You are the Planner. Break this objective down into 3-6 sequential, atomic subtasks.\n"
        f"Format: Provide ONLY a bulleted list of subtasks. No intro/outro.\n"
        f"Example:\n"
        f"- Create the database schema.\n"
        f"- Implement the API endpoints.\n"
        f"- Write the frontend dashboard.\n"
    )
    return call_model("orchestrator", prompt)


def create_work_request(
    task_spec: dict,
    plan: str,
    context: str,
) -> "Message":
    """
    Create a WORK_REQUEST message to send to the Engineer.

    Args:
        task_spec: TaskSpec dict from Translator
        plan: Strategic plan from think()
        context: Project context

    Returns:
        Message ready to send via bus
    """
    from protocols.messages import work_request
    return work_request(plan=plan, task_spec=task_spec, context=context)


def handle_message(message: "Message") -> Optional["Message"]:
    """
    Handle incoming messages to the Orchestrator.

    The Orchestrator receives:
    - COMPLETE: Task finished successfully
    - ERROR: Something went wrong
    - EVALUATION: Quality assessment from Designer

    Args:
        message: Incoming Message object

    Returns:
        Response Message or None (orchestrator often terminates flow)
    """
    from protocols.messages import Message, MessageType, error_message

    msg_type = message.type
    payload = message.payload

    if msg_type == MessageType.COMPLETE:
        # Task completed successfully - log and signal termination
        final_output = payload.get("final_output", "")
        summary = payload.get("summary", "")
        print(f"[Orchestrator] Task COMPLETE: {summary[:100]}...")

        # Return None to signal end of message loop
        return None

    elif msg_type == MessageType.ERROR:
        # Error occurred - decide whether to retry or abort
        error = payload.get("error", "Unknown error")
        recoverable = payload.get("recoverable", True)

        if recoverable:
            print(f"[Orchestrator] Recoverable error: {error}")
            # Could dispatch a new WORK_REQUEST with error context
            # For now, just acknowledge
            return None
        else:
            print(f"[Orchestrator] Fatal error: {error}")
            return None

    elif msg_type == MessageType.EVALUATION:
        # Quality assessment from Designer (e.g., after ASIC options)
        selected = payload.get("selected_option", "")
        reasoning = payload.get("reasoning", "")
        print(f"[Orchestrator] Evaluation received: {reasoning[:80]}...")
        return None

    else:
        print(f"[Orchestrator] Unknown message type: {msg_type}")
        return error_message(
            error=f"Orchestrator cannot handle message type: {msg_type}",
            recoverable=True,
        )


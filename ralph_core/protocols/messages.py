"""
Message Protocol Definitions

Defines the message types and structure for inter-agent communication.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of messages that can be sent between agents."""

    # Orchestrator → Engineer
    WORK_REQUEST = "work_request"        # Initial work assignment

    # Engineer → Designer
    CODE_OUTPUT = "code_output"          # Code/solution ready for review

    # Designer → Engineer
    REVISION_REQUEST = "revision_req"    # Request changes to code

    # Engineer → ASIC (NEW)
    ASIC_REQUEST = "asic_request"        # Request micro-task from specialist

    # ASIC → Engineer/Designer (RENAMED from OPTIONS)
    ASIC_RESPONSE = "asic_response"      # Multiple options from ASIC

    # Legacy: OPTIONS kept for backward compatibility
    OPTIONS = "options"                  # Multiple options from ASIC

    # Designer → Orchestrator
    EVALUATION = "evaluation"            # Quality assessment

    # Any → Orchestrator
    COMPLETE = "complete"                # Task finished successfully

    # Any → Any
    ERROR = "error"                      # Error occurred
    STATUS = "status"                    # Status update
    DIAGNOSTIC = "diagnostic"            # Diagnostic info for Wiggum Protocol (NEW)

    # Orchestrator → Any
    ABORT = "abort"                      # Cancel current work

    # Forklift Protocol (Memory → Agents)
    FORKLIFT_REQUEST = "forklift_request"    # Request relevant memories for a task
    FORKLIFT_RESPONSE = "forklift_response"  # Structured memory payload

    # Tool Protocol (Agent → Tool Dispatcher)
    TOOL_REQUEST = "tool_request"      # Agent requests tool execution
    TOOL_RESPONSE = "tool_response"    # Tool execution result
    TOOL_CONFIRM = "tool_confirm"      # User confirmation for dangerous tools

    # REM Sleep Protocol (Daemon → Sleeper Agent)
    REM_SLEEP_START = "rem_sleep_start"        # Begin memory consolidation cycle
    REM_SLEEP_COMPLETE = "rem_sleep_complete"  # Consolidation finished
    CONSOLIDATION_REQUEST = "consolidation_request"   # Request lesson clustering
    CONSOLIDATION_RESPONSE = "consolidation_response" # Return synthesized guidelines


@dataclass
class Message:
    """
    A message passed between agents.

    Attributes:
        type: The message type (determines handling)
        sender: Agent that sent the message
        receiver: Intended recipient agent
        payload: Message content (varies by type)
        id: Unique message identifier
        timestamp: When the message was created
        correlation_id: Links related messages (e.g., request/response)
        metadata: Additional context
    """
    type: MessageType
    sender: str
    receiver: str
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reply(self, msg_type: MessageType, payload: dict) -> "Message":
        """Create a reply message to this message."""
        return Message(
            type=msg_type,
            sender=self.receiver,  # Swap sender/receiver
            receiver=self.sender,
            payload=payload,
            correlation_id=self.id,  # Link to original
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create Message from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


# Convenience constructors for common message types

def work_request(
    plan: str,
    task_spec: Optional[dict] = None,
    context: Optional[str] = None,
) -> Message:
    """Create a WORK_REQUEST message from Orchestrator to Engineer."""
    return Message(
        type=MessageType.WORK_REQUEST,
        sender="orchestrator",
        receiver="engineer",
        payload={
            "plan": plan,
            "task_spec": task_spec,
            "context": context,
        }
    )


def code_output(
    code: str,
    files_created: list[str] = None,
    execute_commands: list[str] = None,
    notes: str = "",
) -> Message:
    """Create a CODE_OUTPUT message from Engineer to Designer."""
    return Message(
        type=MessageType.CODE_OUTPUT,
        sender="engineer",
        receiver="designer",
        payload={
            "code": code,
            "files_created": files_created or [],
            "execute_commands": execute_commands or [],
            "notes": notes,
        }
    )


def revision_request(
    issues: list[str],
    suggestions: list[str] = None,
    severity: str = "medium",
    round_number: int = 1,
) -> Message:
    """Create a REVISION_REQUEST message from Designer to Engineer."""
    return Message(
        type=MessageType.REVISION_REQUEST,
        sender="designer",
        receiver="engineer",
        payload={
            "issues": issues,
            "suggestions": suggestions or [],
            "severity": severity,
            "round": round_number,
        }
    )


def options_message(
    task_type: str,
    options: list[str],
    scores: list[float] = None,
) -> Message:
    """Create an OPTIONS message from ASIC to middle management."""
    return Message(
        type=MessageType.OPTIONS,
        sender=f"asic:{task_type}",
        receiver="engineer",  # or designer
        payload={
            "task_type": task_type,
            "options": options,
            "scores": scores,
        }
    )


def complete_message(
    final_output: str,
    files_modified: list[str] = None,
    summary: str = "",
) -> Message:
    """Create a COMPLETE message indicating task finished."""
    return Message(
        type=MessageType.COMPLETE,
        sender="designer",
        receiver="orchestrator",
        payload={
            "final_output": final_output,
            "files_modified": files_modified or [],
            "summary": summary,
        }
    )


def error_message(
    error: str,
    context: str = "",
    recoverable: bool = True,
) -> Message:
    """Create an ERROR message."""
    return Message(
        type=MessageType.ERROR,
        sender="system",
        receiver="orchestrator",
        payload={
            "error": error,
            "context": context,
            "recoverable": recoverable,
        }
    )


# ============================================
# ASIC Message Constructors
# ============================================

def asic_request(
    task_type: str,
    prompt: str,
    parallel: bool = False,
    return_to: str = "engineer",
) -> Message:
    """
    Create an ASIC_REQUEST message to spawn a specialist.

    Args:
        task_type: Type of ASIC (regex, json, sql, test, fix, etc.)
        prompt: The specific micro-task prompt
        parallel: Whether to use parallel option generation
        return_to: Agent to send response to (default: engineer)

    Returns:
        Message to send to ASIC handler
    """
    return Message(
        type=MessageType.ASIC_REQUEST,
        sender="engineer",
        receiver=f"asic:{task_type}",
        payload={
            "task_type": task_type,
            "prompt": prompt,
            "parallel": parallel,
        },
        metadata={
            "return_to": return_to,
        }
    )


def asic_response(
    task_type: str,
    options: list[str],
    model_used: str = "",
    return_to: str = "engineer",
) -> Message:
    """
    Create an ASIC_RESPONSE message with generated options.

    Args:
        task_type: Type of ASIC that generated the options
        options: List of candidate solutions
        model_used: Which model was actually used
        return_to: Agent to send response to

    Returns:
        Message with options for evaluation
    """
    return Message(
        type=MessageType.ASIC_RESPONSE,
        sender=f"asic:{task_type}",
        receiver=return_to,
        payload={
            "task_type": task_type,
            "options": options,
            "num_options": len(options),
            "model_used": model_used,
        }
    )


# ============================================
# Forklift Protocol Message Constructors
# ============================================

def forklift_request(
    objective: str,
    task_type: str = "code",
    requester: str = "engineer",
    scope: str = "standard",
    hints: list[str] = None,
) -> Message:
    """
    Create a FORKLIFT_REQUEST message to retrieve relevant memories.

    Args:
        objective: The task/objective to find memories for
        task_type: Type of task (code, test, fix, regex, etc.)
        requester: Agent requesting the memories
        scope: Retrieval scope - "minimal", "standard", or "comprehensive"
        hints: Optional tags/files the requester knows are relevant

    Returns:
        Message to send to Forklift handler
    """
    return Message(
        type=MessageType.FORKLIFT_REQUEST,
        sender=requester,
        receiver="forklift",
        payload={
            "objective": objective,
            "task_type": task_type,
            "requester": requester,
            "scope": scope,
            "hints": hints or [],
        }
    )


def forklift_response(
    memories: dict,
    scores: dict,
    retrieval_time_ms: int,
    requester: str,
    truncated: bool = False,
) -> Message:
    """
    Create a FORKLIFT_RESPONSE message with retrieved memories.

    Args:
        memories: Dict with lessons, facts, context, files
        scores: Relevance scores by category
        retrieval_time_ms: How long retrieval took
        requester: Agent to send response to
        truncated: Whether results were truncated

    Returns:
        Message with structured memory payload
    """
    return Message(
        type=MessageType.FORKLIFT_RESPONSE,
        sender="forklift",
        receiver=requester,
        payload={
            "memories": memories,
            "scores": scores,
            "retrieval_time_ms": retrieval_time_ms,
            "truncated": truncated,
        }
    )


# ============================================
# REM Sleep Protocol Message Constructors
# ============================================

def rem_sleep_start(
    max_duration: int = 120,
    trigger: str = "idle",
) -> Message:
    """
    Create a REM_SLEEP_START message to begin memory consolidation.

    Args:
        max_duration: Maximum seconds for the REM cycle
        trigger: What triggered REM sleep (idle, manual, scheduled)

    Returns:
        Message to send to Sleeper agent
    """
    return Message(
        type=MessageType.REM_SLEEP_START,
        sender="daemon",
        receiver="sleeper",
        payload={
            "max_duration": max_duration,
            "trigger": trigger,
        }
    )


def rem_sleep_complete(
    success: bool,
    new_guidelines: int,
    lessons_analyzed: int,
    duration_ms: int,
    phase_completed: str = "complete",
) -> Message:
    """
    Create a REM_SLEEP_COMPLETE message when consolidation finishes.

    Args:
        success: Whether consolidation completed successfully
        new_guidelines: Number of guidelines created
        lessons_analyzed: Number of lessons processed
        duration_ms: Total duration in milliseconds
        phase_completed: Last phase completed (for partial runs)

    Returns:
        Message with consolidation results
    """
    return Message(
        type=MessageType.REM_SLEEP_COMPLETE,
        sender="sleeper",
        receiver="daemon",
        payload={
            "success": success,
            "new_guidelines": new_guidelines,
            "lessons_analyzed": lessons_analyzed,
            "duration_ms": duration_ms,
            "phase_completed": phase_completed,
        }
    )


def consolidation_request(
    threshold: float = 0.80,
    min_cluster_size: int = 3,
) -> Message:
    """
    Create a CONSOLIDATION_REQUEST message for lesson clustering.

    Args:
        threshold: Similarity threshold for clustering (0.80 = conservative)
        min_cluster_size: Minimum lessons needed to form a cluster

    Returns:
        Message to send to Consolidator
    """
    return Message(
        type=MessageType.CONSOLIDATION_REQUEST,
        sender="sleeper",
        receiver="consolidator",
        payload={
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
        }
    )


def consolidation_response(
    guidelines: list,
    clusters_found: int,
    lessons_analyzed: int,
    duration_ms: int,
) -> Message:
    """
    Create a CONSOLIDATION_RESPONSE message with synthesized guidelines.

    Args:
        guidelines: List of guideline dicts (text, category, confidence, etc.)
        clusters_found: Number of lesson clusters identified
        lessons_analyzed: Total lessons processed
        duration_ms: Processing time

    Returns:
        Message with guideline synthesis results
    """
    return Message(
        type=MessageType.CONSOLIDATION_RESPONSE,
        sender="consolidator",
        receiver="sleeper",
        payload={
            "guidelines": guidelines,
            "clusters_found": clusters_found,
            "lessons_analyzed": lessons_analyzed,
            "duration_ms": duration_ms,
        },
    )


def diagnostic_message(
    error: str,
    traceback: str,
    agent_state: dict[str, Any],
    attempt_count: int = 1,
    sender: str = "system",
    receiver: str = "orchestrator",
) -> Message:
    """
    Create a DIAGNOSTIC message for the Wiggum Protocol.

    Args:
        error: Description of the error/failure
        traceback: Stack trace or detailed error context
        agent_state: Snapshot of agent internal state at time of failure
        attempt_count: How many times the task has been attempted
        sender: The agent that failed
        receiver: Recipient (usually orchestrator or reflector)

    Returns:
        Message with structured diagnostic payload
    """
    return Message(
        type=MessageType.DIAGNOSTIC,
        sender=sender,
        receiver=receiver,
        payload={
            "error": error,
            "error_context": {
                "traceback": traceback,
                "agent_state": agent_state,
                "attempt_count": attempt_count,
            }
        }
    )


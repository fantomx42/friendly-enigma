"""
Message Bus - Central Communication Hub

Routes messages between agents and maintains conversation history.
Implements circuit breakers to prevent infinite loops.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime, timedelta
from .messages import Message, MessageType


@dataclass
class BusConfig:
    """Configuration for the message bus."""
    max_messages: int = 50              # Max messages before forced stop
    max_revision_rounds: int = 3        # Max back-and-forth revisions
    message_timeout_sec: int = 300      # Timeout for waiting on messages
    enable_logging: bool = True


class MessageBus:
    """
    Central message router for agent communication.

    Features:
    - FIFO queue per recipient
    - Message history tracking
    - Circuit breaker for revision loops
    - Timeout handling
    """

    def __init__(self, config: Optional[BusConfig] = None):
        self.config = config or BusConfig()

        # Queues for each agent
        self._queues: dict[str, deque[Message]] = {
            "orchestrator": deque(),
            "engineer": deque(),
            "designer": deque(),
            "system": deque(),
            "tool_dispatcher": deque(),  # Tool execution handler
        }

        # Full message history
        self._history: list[Message] = []

        # Revision counter (for circuit breaker)
        self._revision_count: int = 0

        # Handlers for message types (optional callbacks)
        self._handlers: dict[MessageType, list[Callable[[Message], None]]] = {}

        # Stats
        self._stats = {
            "total_sent": 0,
            "by_type": {},
            "start_time": datetime.now(),
        }

    def send(self, message: Message) -> bool:
        """
        Send a message to its recipient.

        Returns:
            True if sent successfully, False if circuit breaker triggered
        """
        is_diagnostic = message.type == MessageType.DIAGNOSTIC

        # Circuit breaker: max messages (Bypassed for diagnostic messages)
        if not is_diagnostic and len(self._history) >= self.config.max_messages:
            if self.config.enable_logging:
                print(f"[BUS] Circuit breaker: Max messages ({self.config.max_messages}) reached")
            return False

        # Circuit breaker: revision loops
        if message.type == MessageType.REVISION_REQUEST:
            self._revision_count += 1
            if self._revision_count > self.config.max_revision_rounds:
                if self.config.enable_logging:
                    print(f"[BUS] Circuit breaker: Max revisions ({self.config.max_revision_rounds}) reached")
                return False

        # Add to recipient queue
        receiver = message.receiver
        if receiver not in self._queues:
            self._queues[receiver] = deque()

        self._queues[receiver].append(message)
        self._history.append(message)

        # Update stats
        self._stats["total_sent"] += 1
        msg_type = message.type.value
        self._stats["by_type"][msg_type] = self._stats["by_type"].get(msg_type, 0) + 1

        # Call handlers
        if message.type in self._handlers:
            for handler in self._handlers[message.type]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"[BUS] Handler error: {e}")

        if self.config.enable_logging:
            print(f"[BUS] {message.sender} --[{message.type.value}]--> {message.receiver}")
            # GUI JSON capture
            print(f"[MESSAGE] {json.dumps(message.to_dict())}", flush=True)

        return True

    def receive(self, agent: str, timeout_sec: Optional[int] = None) -> Optional[Message]:
        """
        Receive the next message for an agent.

        Args:
            agent: The agent receiving messages
            timeout_sec: How long to wait (not implemented for sync version)

        Returns:
            Next message or None if queue empty
        """
        if agent not in self._queues:
            return None

        queue = self._queues[agent]
        if queue:
            return queue.popleft()

        return None

    def peek(self, agent: str) -> Optional[Message]:
        """Look at next message without removing it."""
        if agent not in self._queues:
            return None

        queue = self._queues[agent]
        if queue:
            return queue[0]

        return None

    def has_messages(self, agent: str) -> bool:
        """Check if agent has pending messages."""
        return bool(self._queues.get(agent))

    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], None]
    ):
        """Register a callback for a message type."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    def get_history(self, limit: int = 0) -> list[Message]:
        """Get message history (most recent first)."""
        history = list(reversed(self._history))
        if limit > 0:
            return history[:limit]
        return history

    def get_conversation(self, between: tuple[str, str]) -> list[Message]:
        """Get messages between two specific agents."""
        a, b = between
        return [
            m for m in self._history
            if (m.sender == a and m.receiver == b) or (m.sender == b and m.receiver == a)
        ]

    def get_stats(self) -> dict:
        """Get bus statistics."""
        elapsed = datetime.now() - self._stats["start_time"]
        return {
            **self._stats,
            "elapsed_seconds": elapsed.total_seconds(),
            "revision_count": self._revision_count,
            "queued_messages": {
                agent: len(queue) for agent, queue in self._queues.items()
            },
        }

    def reset(self):
        """Reset the bus state."""
        for queue in self._queues.values():
            queue.clear()
        self._history.clear()
        self._revision_count = 0
        self._stats = {
            "total_sent": 0,
            "by_type": {},
            "start_time": datetime.now(),
        }

    def is_complete(self) -> bool:
        """Check if a COMPLETE message was sent."""
        return any(m.type == MessageType.COMPLETE for m in self._history)

    def get_final_output(self) -> Optional[str]:
        """Get the final output from COMPLETE message if available."""
        for m in reversed(self._history):
            if m.type == MessageType.COMPLETE:
                return m.payload.get("final_output")
        return None

    def format_history(self) -> str:
        """Format message history as readable string."""
        lines = ["=== Message History ==="]
        for m in self._history:
            time_str = m.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{time_str}] {m.sender} --[{m.type.value}]--> {m.receiver}")
            if m.type == MessageType.REVISION_REQUEST:
                issues = m.payload.get("issues", [])
                lines.append(f"         Issues: {issues[:2]}...")  # Show first 2
        lines.append("=" * 25)
        return "\n".join(lines)


# Global bus instance (can be replaced per-session)
_global_bus: Optional[MessageBus] = None


def get_bus() -> MessageBus:
    """Get or create the global message bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus


def reset_bus():
    """Reset the global message bus."""
    global _global_bus
    _global_bus = MessageBus()

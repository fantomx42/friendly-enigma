"""
towers.py - Guard Towers (Audit Logging)

The watchtowers that see everything leaving the compound.
Every output is logged for accountability and debugging.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class AuditLevel(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKED = "blocked"  # Output was blocked by security


class AuditTower:
    """
    Guard Tower - Watches and logs all output activity.

    Maintains audit trail of:
    - What was output
    - When it was output
    - Where it went (terminal, file, browser, etc.)
    - Who/what initiated it
    - Whether it passed security checks
    """

    def __init__(self, log_dir: str = "logs/security"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "audit.jsonl")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # In-memory buffer for current session
        self._session_log: list = []

    def log(
        self,
        event_type: str,
        level: AuditLevel,
        output_type: str,
        content_summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        blocked: bool = False,
        block_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log an audit event.

        Args:
            event_type: Type of event (output_attempt, output_success, output_blocked, etc.)
            level: Severity level
            output_type: Where output is going (terminal, file, browser, api, voice)
            content_summary: Brief summary of content (truncated for security)
            metadata: Additional context
            blocked: Whether the output was blocked
            block_reason: Why it was blocked (if applicable)

        Returns:
            The audit record that was logged
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "level": level.value,
            "output_type": output_type,
            "content_summary": self._truncate(content_summary, 200),
            "blocked": blocked,
            "block_reason": block_reason,
            "metadata": metadata or {}
        }

        # Add to session buffer
        self._session_log.append(record)

        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Tower] WARNING: Failed to write audit log: {e}")

        # Print warnings and above to console
        if level in (AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL, AuditLevel.BLOCKED):
            icon = {
                AuditLevel.WARNING: "⚠️",
                AuditLevel.ERROR: "❌",
                AuditLevel.CRITICAL: "🚨",
                AuditLevel.BLOCKED: "🛑"
            }.get(level, "📋")
            print(f"[Tower] {icon} {level.value.upper()}: {event_type} - {content_summary[:50]}...")

        return record

    def log_output_attempt(
        self,
        output_type: str,
        content: str,
        destination: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log an attempt to send output."""
        return self.log(
            event_type="output_attempt",
            level=AuditLevel.INFO,
            output_type=output_type,
            content_summary=content,
            metadata={"destination": destination}
        )

    def log_output_success(
        self,
        output_type: str,
        content: str,
        destination: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log successful output."""
        return self.log(
            event_type="output_success",
            level=AuditLevel.INFO,
            output_type=output_type,
            content_summary=content,
            metadata={"destination": destination}
        )

    def log_output_blocked(
        self,
        output_type: str,
        content: str,
        reason: str,
        detector: str
    ) -> Dict[str, Any]:
        """Log blocked output."""
        return self.log(
            event_type="output_blocked",
            level=AuditLevel.BLOCKED,
            output_type=output_type,
            content_summary=content,
            blocked=True,
            block_reason=reason,
            metadata={"detector": detector}
        )

    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        severity: AuditLevel = AuditLevel.WARNING
    ) -> Dict[str, Any]:
        """Log a security alert."""
        return self.log(
            event_type=f"security_alert:{alert_type}",
            level=severity,
            output_type="internal",
            content_summary=description
        )

    def get_session_log(self) -> list:
        """Get all logs from current session."""
        return self._session_log.copy()

    def get_blocked_count(self) -> int:
        """Get count of blocked outputs in current session."""
        return sum(1 for r in self._session_log if r.get("blocked", False))

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text for logging (don't log full secrets!)."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...[truncated]"


# Global instance
audit = AuditTower()

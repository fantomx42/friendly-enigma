"""
checkpoint.py - The Main Security Checkpoint

This is the central security hub that orchestrates all security checks
before anything leaves the compound.

Flow:
1. Towers log the attempt (audit)
2. Dogs sniff for threats (malware, secrets)
3. Guards validate (permissions, sanitization)
4. Decision: PASS or BLOCK
"""

from typing import Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .towers import AuditTower, AuditLevel, audit
from .dogs import MalwareDog, SecretDog, ThreatLevel, sniff_all
from .guards import SecurityGuard, ValidationResult, OutputType, Permission, guard


class Decision(Enum):
    """Final checkpoint decision."""
    PASS = "pass"
    BLOCK = "block"
    CONFIRM = "confirm"  # Needs user confirmation


@dataclass
class CheckpointResult:
    """Result from the security checkpoint."""
    decision: Decision
    passed_audit: bool
    passed_dogs: bool
    passed_guards: bool
    malware_findings: list
    secret_findings: list
    guard_issues: list
    block_reason: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    sanitized_content: Optional[str] = None


class SecurityCheckpoint:
    """
    The Security Checkpoint - All output must pass through here.

    Components:
    - Guard Towers (Audit): Log everything
    - Security Dogs (Sniffers): Detect malware and secrets
    - Security Guards (Validation): Check permissions and sanitize
    """

    def __init__(self):
        self.tower = audit  # Global audit tower
        self.malware_dog = MalwareDog()
        self.secret_dog = SecretDog()
        self.guard = guard  # Global security guard

    def check(
        self,
        content: Any,
        output_type: str,
        destination: Optional[str] = None,
        operation: Optional[str] = None
    ) -> CheckpointResult:
        """
        Run all security checks on content.

        Args:
            content: The content attempting to leave
            output_type: Type of output (terminal, file, shell, etc.)
            destination: Where it's going (file path, URL, etc.)
            operation: What operation is being performed

        Returns:
            CheckpointResult with decision and details
        """
        content_str = str(content) if not isinstance(content, str) else content

        # 1. TOWERS: Log the attempt
        self.tower.log_output_attempt(output_type, content_str, destination)

        # 2. DOGS: Sniff for threats
        malware_result = self.malware_dog.sniff(content_str)
        secret_result = self.secret_dog.sniff(content_str)

        # Check if dogs found critical threats
        dogs_passed = True
        block_reason = None

        if malware_result.threat_level == ThreatLevel.CRITICAL:
            dogs_passed = False
            block_reason = f"Malware: {malware_result.recommendation}"
            self.tower.log_output_blocked(
                output_type, content_str,
                reason=block_reason,
                detector="MalwareDog"
            )

        if secret_result.threat_level == ThreatLevel.CRITICAL:
            dogs_passed = False
            block_reason = f"Secret leak: {secret_result.recommendation}"
            self.tower.log_output_blocked(
                output_type, content_str,
                reason=block_reason,
                detector="SecretDog"
            )

        # If dogs block, return immediately
        if not dogs_passed:
            return CheckpointResult(
                decision=Decision.BLOCK,
                passed_audit=True,
                passed_dogs=False,
                passed_guards=False,
                malware_findings=malware_result.findings,
                secret_findings=secret_result.findings,
                guard_issues=[],
                block_reason=block_reason
            )

        # 3. GUARDS: Validate and sanitize
        try:
            otype = OutputType(output_type.lower())
        except ValueError:
            otype = OutputType.TERMINAL

        guard_result = self.guard.validate(content, otype, destination, operation)

        # Check if guards block
        if not guard_result.passed:
            block_reason = f"Validation failed: {', '.join(guard_result.issues)}"
            self.tower.log_output_blocked(
                output_type, content_str,
                reason=block_reason,
                detector="SecurityGuard"
            )
            return CheckpointResult(
                decision=Decision.BLOCK,
                passed_audit=True,
                passed_dogs=True,
                passed_guards=False,
                malware_findings=malware_result.findings,
                secret_findings=secret_result.findings,
                guard_issues=guard_result.issues,
                block_reason=block_reason
            )

        # 4. DECISION: Determine final outcome
        decision = Decision.PASS

        # Check if confirmation is needed
        requires_confirmation = guard_result.requires_confirmation
        confirmation_message = guard_result.confirmation_message

        # Also require confirmation for high (non-critical) threats
        if malware_result.threat_level == ThreatLevel.HIGH:
            requires_confirmation = True
            confirmation_message = f"Malware warning: {malware_result.findings}"

        if secret_result.threat_level == ThreatLevel.HIGH:
            requires_confirmation = True
            confirmation_message = f"Secret warning: {secret_result.findings}"

        if requires_confirmation:
            decision = Decision.CONFIRM

        # Log success (or pending confirmation)
        if decision == Decision.PASS:
            self.tower.log_output_success(output_type, content_str, destination)

        return CheckpointResult(
            decision=decision,
            passed_audit=True,
            passed_dogs=True,
            passed_guards=True,
            malware_findings=malware_result.findings,
            secret_findings=secret_result.findings,
            guard_issues=guard_result.issues,
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message,
            sanitized_content=guard_result.sanitized_content
        )

    def force_pass(self, content: str, output_type: str, reason: str) -> CheckpointResult:
        """
        Force pass content (after user confirmation).
        Still logs the action for audit trail.
        """
        self.tower.log(
            event_type="force_pass",
            level=AuditLevel.WARNING,
            output_type=output_type,
            content_summary=content,
            metadata={"reason": reason}
        )

        return CheckpointResult(
            decision=Decision.PASS,
            passed_audit=True,
            passed_dogs=True,
            passed_guards=True,
            malware_findings=[],
            secret_findings=[],
            guard_issues=[],
            block_reason=None
        )

    def get_session_stats(self) -> dict:
        """Get security stats for current session."""
        return {
            "audit_logs": len(self.tower.get_session_log()),
            "blocked_count": self.tower.get_blocked_count(),
            "rate_limits": self.guard.get_stats()
        }


# Global checkpoint instance
checkpoint = SecurityCheckpoint()


def check_output(
    content: Any,
    output_type: str,
    destination: Optional[str] = None
) -> CheckpointResult:
    """Convenience function to check output through the checkpoint."""
    return checkpoint.check(content, output_type, destination)


def is_safe_to_output(content: Any, output_type: str) -> Tuple[bool, str]:
    """
    Quick check if content is safe to output.

    Returns:
        Tuple of (is_safe, reason_if_not)
    """
    result = checkpoint.check(content, output_type)

    if result.decision == Decision.BLOCK:
        return False, result.block_reason or "Blocked by security"

    if result.decision == Decision.CONFIRM:
        return False, f"Requires confirmation: {result.confirmation_message}"

    return True, "Safe to output"

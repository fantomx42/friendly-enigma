"""
guards.py - Security Guards (Validation Layer)

The guards who check everything before it leaves:
- Type checking: Is the output in expected format?
- Permission verification: Is this action authorized?
- Content sanitization: Remove dangerous content
- Rate limiting: Prevent abuse
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class OutputType(Enum):
    """Types of output that can leave the compound."""
    TERMINAL = "terminal"
    FILE = "file"
    BROWSER = "browser"
    API = "api"
    VOICE = "voice"
    SHELL = "shell"
    GIT = "git"
    TOOL = "tool"  # Tool invocation through dispatcher


class Permission(Enum):
    """Permission levels for operations."""
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
    RATE_LIMITED = "rate_limited"


@dataclass
class ValidationResult:
    """Result of a security guard validation."""
    passed: bool
    permission: Permission
    issues: List[str] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None


class SecurityGuard:
    """Security Guard - Validates all output before it leaves."""

    def __init__(self):
        self._rate_limits: Dict[OutputType, int] = {
            OutputType.SHELL: 50,
            OutputType.FILE: 100,
            OutputType.GIT: 20,
            OutputType.BROWSER: 30,
            OutputType.API: 100,
        }
        self._call_counts: Dict[OutputType, int] = defaultdict(int)

        self._forbidden_paths = [
            "/etc/", "/usr/", "/bin/", "/sbin/",
            "/boot/", "/dev/", "/proc/", "/sys/",
            "~/.ssh/", "~/.gnupg/", ".git/hooks/",
        ]

        self._dangerous_patterns = [
            (r"rm\s+-rf\s+/", "Recursive delete from root"),
            (r"rm\s+-rf\s+\*", "Recursive delete wildcard"),
            (r"chmod\s+777", "World-writable permissions"),
            (r"chmod\s+\+s", "SetUID bit modification"),
            (r"mkfs\.", "Filesystem format command"),
            (r"dd\s+if=.*of=/dev/", "Direct disk write"),
            (r"curl.*\|\s*(ba)?sh", "Pipe to shell"),
            (r"wget.*\|\s*(ba)?sh", "Pipe to shell"),
        ]

    def validate(
        self,
        content: Any,
        output_type: OutputType,
        destination: Optional[str] = None,
        operation: Optional[str] = None
    ) -> ValidationResult:
        """Validate content before allowing output."""
        issues = []
        requires_confirmation = False
        confirmation_message = None
        sanitized = content

        # 1. Rate limiting check
        if output_type in self._rate_limits:
            if self._call_counts[output_type] >= self._rate_limits[output_type]:
                return ValidationResult(
                    passed=False,
                    permission=Permission.RATE_LIMITED,
                    issues=[f"Rate limit exceeded for {output_type.value}"]
                )
            self._call_counts[output_type] += 1

        # 2. Type-specific validation
        if output_type == OutputType.FILE:
            result = self._validate_file(destination)
            if not result.passed:
                return result
            issues.extend(result.issues)

        elif output_type == OutputType.SHELL:
            result = self._validate_shell(str(content))
            if not result.passed:
                return result
            if result.requires_confirmation:
                requires_confirmation = True
                confirmation_message = result.confirmation_message
            issues.extend(result.issues)

        elif output_type == OutputType.GIT:
            result = self._validate_git(str(content))
            if result.requires_confirmation:
                requires_confirmation = True
                confirmation_message = result.confirmation_message
            issues.extend(result.issues)

        # 3. Content sanitization
        if isinstance(content, str):
            sanitized = self._sanitize(content)

        return ValidationResult(
            passed=True,
            permission=Permission.ASK if requires_confirmation else Permission.ALLOW,
            issues=issues,
            sanitized_content=sanitized if sanitized != content else None,
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message
        )

    def _validate_file(self, destination: Optional[str]) -> ValidationResult:
        """Validate file write operations."""
        issues = []
        if destination:
            for forbidden in self._forbidden_paths:
                if forbidden in destination:
                    return ValidationResult(
                        passed=False,
                        permission=Permission.DENY,
                        issues=[f"Forbidden path: {destination}"]
                    )

            # Warn about executable files
            exec_extensions = [".sh", ".bash", ".py", ".exe", ".bat"]
            if any(destination.endswith(ext) for ext in exec_extensions):
                issues.append(f"Writing executable file: {destination}")

        return ValidationResult(passed=True, permission=Permission.ALLOW, issues=issues)

    def _validate_shell(self, command: str) -> ValidationResult:
        """Validate shell commands."""
        issues = []
        for pattern, description in self._dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return ValidationResult(
                    passed=True,
                    permission=Permission.ASK,
                    issues=[description],
                    requires_confirmation=True,
                    confirmation_message=f"Dangerous: {description}. Command: {command[:60]}..."
                )
        return ValidationResult(passed=True, permission=Permission.ALLOW, issues=issues)

    def _validate_git(self, content: str) -> ValidationResult:
        """Validate git operations."""
        dangerous_ops = [
            ("push --force", "Force push"),
            ("push -f", "Force push"),
            ("reset --hard", "Hard reset"),
            ("clean -fd", "Force clean"),
            ("branch -D", "Force delete branch"),
        ]
        for pattern, description in dangerous_ops:
            if pattern in content.lower():
                return ValidationResult(
                    passed=True,
                    permission=Permission.ASK,
                    issues=[description],
                    requires_confirmation=True,
                    confirmation_message=f"Dangerous git operation: {description}"
                )
        return ValidationResult(passed=True, permission=Permission.ALLOW)

    def _sanitize(self, content: str) -> str:
        """Remove potentially dangerous patterns from content."""
        sanitized = content
        # Remove ANSI escape codes
        sanitized = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', sanitized)
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        # Remove carriage returns (potential terminal manipulation)
        sanitized = sanitized.replace('\r', '')
        return sanitized

    def reset_rate_limits(self):
        """Reset rate limit counters (call at session start)."""
        self._call_counts.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get current rate limit stats."""
        return {
            otype.value: f"{self._call_counts[otype]}/{self._rate_limits.get(otype, 'unlimited')}"
            for otype in OutputType
        }


# Global instance
guard = SecurityGuard()


def validate_output(
    content: Any,
    output_type: str,
    destination: Optional[str] = None
) -> ValidationResult:
    """Convenience function to validate output."""
    try:
        otype = OutputType(output_type.lower())
    except ValueError:
        otype = OutputType.TERMINAL
    return guard.validate(content, otype, destination)

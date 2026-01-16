"""
security/ - The Security Checkpoint of Ralph Compound.

Nothing leaves the compound without passing through here.

Components:
- Towers (Audit): Log everything for accountability
- Dogs (Sniffers): Detect malware patterns and leaked secrets
- Guards (Validation): Type check, sanitize, verify permissions
- Gate OUT: Unified output handler

Flow: Content -> Checkpoint -> (Towers, Dogs, Guards) -> Gate OUT -> External World
"""

from .towers import AuditTower, AuditLevel, audit
from .dogs import MalwareDog, SecretDog, ThreatLevel, sniff_all, is_safe
from .guards import SecurityGuard, ValidationResult, OutputType, Permission, validate_output, guard
from .checkpoint import SecurityCheckpoint, CheckpointResult, Decision, checkpoint, check_output, is_safe_to_output
from .gate_out import GateOut, GateResult, OutputChannel, gate_out, send_output, print_secure, write_file_secure, run_shell_secure

__all__ = [
    # Towers (Audit)
    "AuditTower",
    "AuditLevel",
    "audit",
    # Dogs (Sniffers)
    "MalwareDog",
    "SecretDog",
    "ThreatLevel",
    "sniff_all",
    "is_safe",
    # Guards (Validation)
    "SecurityGuard",
    "ValidationResult",
    "OutputType",
    "Permission",
    "validate_output",
    "guard",
    # Checkpoint (Orchestrator)
    "SecurityCheckpoint",
    "CheckpointResult",
    "Decision",
    "checkpoint",
    "check_output",
    "is_safe_to_output",
    # Gate OUT (Output Handler)
    "GateOut",
    "GateResult",
    "OutputChannel",
    "gate_out",
    "send_output",
    "print_secure",
    "write_file_secure",
    "run_shell_secure",
]

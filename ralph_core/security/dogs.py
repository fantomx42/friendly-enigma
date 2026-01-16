"""
dogs.py - Security Dogs (Sniffers)

Two types of dogs patrol the checkpoint:
1. Malware Dogs - Sniff for dangerous code patterns
2. Secret Dogs - Sniff for leaked credentials/keys
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Threat severity levels (ordered by severity)."""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SniffResult:
    """Result from a security dog sniff."""
    detected: bool
    threat_level: ThreatLevel
    findings: List[str]
    dog_name: str
    recommendation: str


class MalwareDog:
    """
    Malware Sniffer - Detects dangerous code patterns.

    Looks for:
    - Shell injection patterns
    - Dangerous system calls
    - Network exfiltration attempts
    - File system attacks
    - Code execution tricks
    """

    def __init__(self):
        self.name = "MalwareDog"

        # Dangerous patterns to detect
        self.patterns = {
            # Shell injection
            "shell_injection": [
                (r";\s*rm\s+-rf", ThreatLevel.CRITICAL, "Recursive delete command"),
                (r"\|\s*bash", ThreatLevel.HIGH, "Pipe to bash"),
                (r"\$\(.*\)", ThreatLevel.MEDIUM, "Command substitution"),
                (r"`.*`", ThreatLevel.MEDIUM, "Backtick command execution"),
                (r"eval\s*\(", ThreatLevel.HIGH, "Eval execution"),
                (r"exec\s*\(", ThreatLevel.MEDIUM, "Exec call"),
            ],
            # System attacks
            "system_attacks": [
                (r"chmod\s+777", ThreatLevel.HIGH, "World-writable permissions"),
                (r"chmod\s+\+s", ThreatLevel.CRITICAL, "SetUID bit"),
                (r"/etc/passwd", ThreatLevel.HIGH, "Password file access"),
                (r"/etc/shadow", ThreatLevel.CRITICAL, "Shadow file access"),
                (r"mkfs\.", ThreatLevel.CRITICAL, "Filesystem format"),
                (r"dd\s+if=.*of=/dev/", ThreatLevel.CRITICAL, "Direct disk write"),
            ],
            # Network exfiltration
            "exfiltration": [
                (r"curl\s+.*-d\s+", ThreatLevel.MEDIUM, "Curl POST data"),
                (r"wget\s+.*--post-data", ThreatLevel.MEDIUM, "Wget POST"),
                (r"nc\s+-e", ThreatLevel.CRITICAL, "Netcat reverse shell"),
                (r"python.*-c.*socket", ThreatLevel.HIGH, "Python socket"),
                (r"base64.*\|.*curl", ThreatLevel.HIGH, "Encoded exfiltration"),
            ],
            # Dangerous Python
            "dangerous_python": [
                (r"__import__\s*\(\s*['\"]os['\"]\s*\)", ThreatLevel.HIGH, "Dynamic OS import"),
                (r"subprocess\.call\s*\(\s*shell\s*=\s*True", ThreatLevel.HIGH, "Shell subprocess"),
                (r"os\.system\s*\(", ThreatLevel.MEDIUM, "OS system call"),
                (r"pickle\.loads?\s*\(", ThreatLevel.MEDIUM, "Pickle deserialization"),
                (r"yaml\.load\s*\([^,)]+\)", ThreatLevel.MEDIUM, "Unsafe YAML load"),
            ],
        }

    def sniff(self, content: str) -> SniffResult:
        """
        Sniff content for malware patterns.

        Args:
            content: The content to check

        Returns:
            SniffResult with detection details
        """
        findings = []
        max_threat = ThreatLevel.SAFE

        for category, patterns in self.patterns.items():
            for pattern, threat, description in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings.append(f"[{category}] {description}")
                    if threat.value > max_threat.value:
                        max_threat = threat

        detected = len(findings) > 0

        if max_threat == ThreatLevel.CRITICAL:
            recommendation = "BLOCK: Critical threat detected. Do not allow output."
        elif max_threat == ThreatLevel.HIGH:
            recommendation = "BLOCK: High threat detected. Manual review required."
        elif max_threat == ThreatLevel.MEDIUM:
            recommendation = "WARNING: Suspicious pattern. Consider sanitizing."
        elif max_threat == ThreatLevel.LOW:
            recommendation = "NOTICE: Minor concern. Proceed with caution."
        else:
            recommendation = "SAFE: No threats detected."

        return SniffResult(
            detected=detected,
            threat_level=max_threat,
            findings=findings,
            dog_name=self.name,
            recommendation=recommendation
        )


class SecretDog:
    """
    Secret Sniffer - Detects leaked credentials and keys.

    Looks for:
    - API keys
    - Passwords in plain text
    - Private keys
    - Tokens
    - Connection strings
    """

    def __init__(self):
        self.name = "SecretDog"

        # Secret patterns to detect
        self.patterns = {
            # API Keys
            "api_keys": [
                (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]{20,}", ThreatLevel.HIGH, "API key detected"),
                (r"sk-[a-zA-Z0-9]{32,}", ThreatLevel.CRITICAL, "OpenAI API key"),
                (r"AIza[0-9A-Za-z-_]{35}", ThreatLevel.HIGH, "Google API key"),
                (r"ghp_[a-zA-Z0-9]{36}", ThreatLevel.HIGH, "GitHub token"),
                (r"gho_[a-zA-Z0-9]{36}", ThreatLevel.HIGH, "GitHub OAuth token"),
                (r"AKIA[0-9A-Z]{16}", ThreatLevel.CRITICAL, "AWS Access Key ID"),
            ],
            # Passwords
            "passwords": [
                (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}", ThreatLevel.HIGH, "Password in text"),
                (r"(?i)(secret|token)\s*[:=]\s*['\"]?[^\s'\"]{16,}", ThreatLevel.HIGH, "Secret/token in text"),
            ],
            # Private keys
            "private_keys": [
                (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", ThreatLevel.CRITICAL, "Private key"),
                (r"-----BEGIN PGP PRIVATE KEY BLOCK-----", ThreatLevel.CRITICAL, "PGP private key"),
            ],
            # Connection strings
            "connection_strings": [
                (r"(?i)mongodb(\+srv)?://[^\s]+:[^\s]+@", ThreatLevel.HIGH, "MongoDB connection string"),
                (r"(?i)postgres://[^\s]+:[^\s]+@", ThreatLevel.HIGH, "PostgreSQL connection string"),
                (r"(?i)mysql://[^\s]+:[^\s]+@", ThreatLevel.HIGH, "MySQL connection string"),
                (r"(?i)redis://:[^\s]+@", ThreatLevel.MEDIUM, "Redis connection string"),
            ],
            # Environment variables with secrets
            "env_secrets": [
                (r"(?i)export\s+(API_KEY|SECRET|PASSWORD|TOKEN)\s*=", ThreatLevel.HIGH, "Exported secret"),
                (r"(?i)\.env.*=.*['\"]?[a-zA-Z0-9]{20,}['\"]?", ThreatLevel.MEDIUM, "Env file secret"),
            ],
        }

    def sniff(self, content: str) -> SniffResult:
        """
        Sniff content for leaked secrets.

        Args:
            content: The content to check

        Returns:
            SniffResult with detection details
        """
        findings = []
        max_threat = ThreatLevel.SAFE

        for category, patterns in self.patterns.items():
            for pattern, threat, description in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Don't include the actual secret in findings!
                    findings.append(f"[{category}] {description} (count: {len(matches)})")
                    if threat.value > max_threat.value:
                        max_threat = threat

        detected = len(findings) > 0

        if max_threat == ThreatLevel.CRITICAL:
            recommendation = "BLOCK: Critical secret detected. NEVER output this."
        elif max_threat == ThreatLevel.HIGH:
            recommendation = "BLOCK: Secret detected. Redact before output."
        elif max_threat == ThreatLevel.MEDIUM:
            recommendation = "WARNING: Possible secret. Review before output."
        else:
            recommendation = "SAFE: No secrets detected."

        return SniffResult(
            detected=detected,
            threat_level=max_threat,
            findings=findings,
            dog_name=self.name,
            recommendation=recommendation
        )


def sniff_all(content: str) -> Tuple[SniffResult, SniffResult]:
    """
    Run all security dogs on content.

    Args:
        content: Content to check

    Returns:
        Tuple of (malware_result, secret_result)
    """
    malware_dog = MalwareDog()
    secret_dog = SecretDog()

    return malware_dog.sniff(content), secret_dog.sniff(content)


def is_safe(content: str) -> Tuple[bool, List[str]]:
    """
    Quick check if content is safe to output.

    Args:
        content: Content to check

    Returns:
        Tuple of (is_safe, reasons_if_not_safe)
    """
    malware_result, secret_result = sniff_all(content)

    reasons = []

    if malware_result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
        reasons.append(f"Malware: {malware_result.recommendation}")
        reasons.extend(malware_result.findings)

    if secret_result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
        reasons.append(f"Secrets: {secret_result.recommendation}")
        reasons.extend(secret_result.findings)

    return len(reasons) == 0, reasons

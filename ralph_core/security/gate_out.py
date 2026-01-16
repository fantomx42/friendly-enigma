"""
gate_out.py - The Gate OUT (Unified Output Handler)

The single exit point from the compound. All outputs must pass through here.
Routes to appropriate output channel after security clearance.

Output Channels:
- Terminal: Console/stdout
- File: File system writes
- Browser: Web automation actions
- API: HTTP responses
- Voice: Text-to-speech output
- Shell: Command execution
- Git: Version control operations
"""

import os
import sys
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .checkpoint import checkpoint, CheckpointResult, Decision


class OutputChannel(Enum):
    """Available output channels."""
    TERMINAL = "terminal"
    FILE = "file"
    BROWSER = "browser"
    API = "api"
    VOICE = "voice"
    SHELL = "shell"
    GIT = "git"


@dataclass
class GateResult:
    """Result from attempting to send through Gate OUT."""
    success: bool
    channel: OutputChannel
    checkpoint_result: CheckpointResult
    output_result: Optional[Any] = None
    error: Optional[str] = None


class GateOut:
    """
    Gate OUT - The single exit point from the compound.

    All output goes through:
    1. Security Checkpoint (towers, dogs, guards)
    2. Channel Router (sends to appropriate output)
    3. Confirmation Handler (if needed)
    """

    def __init__(self):
        self._channels: Dict[OutputChannel, Callable] = {
            OutputChannel.TERMINAL: self._output_terminal,
            OutputChannel.FILE: self._output_file,
            OutputChannel.SHELL: self._output_shell,
            OutputChannel.GIT: self._output_git,
        }

        # Confirmation callback (can be overridden)
        self._confirm_callback: Optional[Callable[[str], bool]] = None

    def set_confirmation_handler(self, callback: Callable[[str], bool]):
        """Set the callback for user confirmation requests."""
        self._confirm_callback = callback

    def send(
        self,
        content: Any,
        channel: str,
        destination: Optional[str] = None,
        operation: Optional[str] = None,
        skip_security: bool = False
    ) -> GateResult:
        """
        Send content through Gate OUT.

        Args:
            content: Content to output
            channel: Output channel (terminal, file, shell, etc.)
            destination: Target (file path, URL, etc.)
            operation: What operation is being done
            skip_security: DANGER - Skip security checks (for emergencies only)

        Returns:
            GateResult with success/failure and details
        """
        try:
            output_channel = OutputChannel(channel.lower())
        except ValueError:
            output_channel = OutputChannel.TERMINAL

        # Security check (unless explicitly skipped)
        if not skip_security:
            check_result = checkpoint.check(
                content=content,
                output_type=channel,
                destination=destination,
                operation=operation
            )

            # Handle BLOCK
            if check_result.decision == Decision.BLOCK:
                print(f"[Gate OUT] BLOCKED: {check_result.block_reason}")
                return GateResult(
                    success=False,
                    channel=output_channel,
                    checkpoint_result=check_result,
                    error=check_result.block_reason
                )

            # Handle CONFIRM
            if check_result.decision == Decision.CONFIRM:
                confirmed = self._request_confirmation(
                    check_result.confirmation_message or "Confirm this action?"
                )
                if not confirmed:
                    return GateResult(
                        success=False,
                        channel=output_channel,
                        checkpoint_result=check_result,
                        error="User declined confirmation"
                    )
                # User confirmed - force pass through checkpoint
                check_result = checkpoint.force_pass(
                    str(content), channel, "User confirmed"
                )

            # Use sanitized content if available
            if check_result.sanitized_content:
                content = check_result.sanitized_content

        else:
            # Skip security - create dummy result
            check_result = CheckpointResult(
                decision=Decision.PASS,
                passed_audit=False,
                passed_dogs=False,
                passed_guards=False,
                malware_findings=[],
                secret_findings=[],
                guard_issues=["SECURITY BYPASSED"]
            )
            print("[Gate OUT] WARNING: Security checks bypassed!")

        # Route to appropriate channel
        handler = self._channels.get(output_channel)
        if handler:
            try:
                result = handler(content, destination)
                return GateResult(
                    success=True,
                    channel=output_channel,
                    checkpoint_result=check_result,
                    output_result=result
                )
            except Exception as e:
                return GateResult(
                    success=False,
                    channel=output_channel,
                    checkpoint_result=check_result,
                    error=str(e)
                )
        else:
            return GateResult(
                success=False,
                channel=output_channel,
                checkpoint_result=check_result,
                error=f"No handler for channel: {output_channel.value}"
            )

    def _request_confirmation(self, message: str) -> bool:
        """Request user confirmation."""
        if self._confirm_callback:
            return self._confirm_callback(message)

        # Default: print and ask via stdin
        print(f"\n[Gate OUT] CONFIRMATION REQUIRED:")
        print(f"  {message}")
        response = input("  Proceed? (y/N): ").strip().lower()
        return response in ('y', 'yes')

    # --- Output Channel Handlers ---

    def _output_terminal(self, content: Any, destination: Optional[str] = None) -> str:
        """Output to terminal/console."""
        output = str(content)
        print(output)
        return output

    def _output_file(self, content: Any, destination: Optional[str] = None) -> str:
        """Output to file system."""
        if not destination:
            raise ValueError("File output requires a destination path")

        # Ensure directory exists
        dir_path = os.path.dirname(destination)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Write file
        with open(destination, 'w', encoding='utf-8') as f:
            f.write(str(content))

        return f"Written to {destination}"

    def _output_shell(self, content: Any, destination: Optional[str] = None) -> dict:
        """Execute shell command."""
        import subprocess

        command = str(content)
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def _output_git(self, content: Any, destination: Optional[str] = None) -> dict:
        """Execute git command."""
        import subprocess

        command = f"git {content}"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }


# Global Gate OUT instance
gate_out = GateOut()


def send_output(
    content: Any,
    channel: str,
    destination: Optional[str] = None
) -> GateResult:
    """Convenience function to send output through Gate OUT."""
    return gate_out.send(content, channel, destination)


def print_secure(content: Any) -> GateResult:
    """Print to terminal through security checkpoint."""
    return gate_out.send(content, "terminal")


def write_file_secure(content: Any, path: str) -> GateResult:
    """Write to file through security checkpoint."""
    return gate_out.send(content, "file", destination=path)


def run_shell_secure(command: str) -> GateResult:
    """Run shell command through security checkpoint."""
    return gate_out.send(command, "shell")

"""
executor.py - The Motor Cortex of Ralph.
Handles autonomous execution of shell commands and capture of output.
"""

import subprocess
from typing import Dict, Any

class Executor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def run(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command and returns the results.
        """
        print(f"[Executor] Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error: Command timed out after {self.timeout} seconds.",
                "exit_code": 124
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"CRITICAL EXECUTOR ERROR: {str(e)}",
                "exit_code": 1
            }

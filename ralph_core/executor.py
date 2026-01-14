"""
executor.py - The Motor Cortex of Ralph.
Handles autonomous execution of shell commands and capture of output.

Includes verification tools for automated quality checks:
- run_tests(): pytest execution
- run_lint(): ruff/flake8 linting
- run_typecheck(): mypy type checking
"""

import subprocess
import os
import shlex
import shutil
from typing import Dict, Any, List, Optional


def _find_tool(name: str) -> Optional[str]:
    """Find a tool in PATH or common locations."""
    # Try PATH first
    path = shutil.which(name)
    if path:
        return path

    # Try common locations
    common_paths = [
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
        os.path.expanduser(f"~/.local/bin/{name}"),
        os.path.expanduser(f"~/.cargo/bin/{name}"),  # For ruff
    ]

    for p in common_paths:
        if os.path.exists(p):
            return p

    return None


class Executor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.sandbox_mode = os.environ.get("RALPH_SANDBOX") == "1"

    def run(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command and returns the results.
        """
        
        if self.sandbox_mode:
            # Wrap command for Docker
            # escape the inner command for the outer shell
            print(f"[Executor] (SANDBOXED) Running: {command}")
            # We use shlex.quote to safely pass the command string to bash -c
            # But wait, we are inside python.
            # Docker command: docker exec ralph_sandbox /bin/bash -c <command>
            
            # Simple approach: pass the command array to subprocess (shell=False)
            docker_cmd = [
                "docker", "exec", "ralph_sandbox", 
                "/bin/bash", "-c", command
            ]
            
            try:
                result = subprocess.run(
                    docker_cmd,
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
                    "stderr": f"CRITICAL SANDBOX ERROR: {str(e)}",
                    "exit_code": 1
                }

        # Standard Local Execution
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

    # ==========================================
    # VERIFICATION TOOLS
    # ==========================================

    def run_tests(
        self,
        path: str = ".",
        test_pattern: str = "test_*.py",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run pytest on the specified path.

        Args:
            path: Directory or file to test
            test_pattern: Pattern for test file discovery
            verbose: Enable verbose output

        Returns:
            Dict with success, stdout, stderr, tests_passed, tests_failed
        """
        pytest_path = _find_tool("pytest")

        if not pytest_path:
            return {
                "success": False,
                "stdout": "",
                "stderr": "pytest not found. Install with: pip install pytest",
                "exit_code": 127,
                "tests_passed": 0,
                "tests_failed": 0,
                "tool_available": False,
            }

        cmd = [pytest_path, path]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "-q"])

        print(f"[Executor] Running tests: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,  # Tests may take longer
            )

            # Parse test counts from output
            tests_passed = 0
            tests_failed = 0
            for line in result.stdout.split("\n"):
                if " passed" in line:
                    try:
                        tests_passed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                if " failed" in line:
                    try:
                        tests_failed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tool_available": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Tests timed out after {self.timeout * 2} seconds",
                "exit_code": 124,
                "tests_passed": 0,
                "tests_failed": 0,
                "tool_available": True,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Test execution error: {str(e)}",
                "exit_code": 1,
                "tests_passed": 0,
                "tests_failed": 0,
                "tool_available": True,
            }

    def run_lint(
        self,
        path: str = ".",
        fix: bool = False,
    ) -> Dict[str, Any]:
        """
        Run ruff (or flake8 as fallback) on the specified path.

        Args:
            path: Directory or file to lint
            fix: Attempt to auto-fix issues (ruff only)

        Returns:
            Dict with success, stdout, stderr, issues_count
        """
        # Prefer ruff, fallback to flake8
        ruff_path = _find_tool("ruff")
        flake8_path = _find_tool("flake8")

        if ruff_path:
            cmd = [ruff_path, "check", path]
            if fix:
                cmd.append("--fix")
            tool_name = "ruff"
        elif flake8_path:
            cmd = [flake8_path, path, "--max-line-length=100"]
            tool_name = "flake8"
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": "No linter found. Install with: pip install ruff",
                "exit_code": 127,
                "issues_count": 0,
                "tool_available": False,
                "tool_name": None,
            }

        print(f"[Executor] Running lint ({tool_name}): {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Count issues from output
            issues_count = len([
                line for line in result.stdout.split("\n")
                if line.strip() and ":" in line
            ])

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
                "issues_count": issues_count,
                "tool_available": True,
                "tool_name": tool_name,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Lint timed out after {self.timeout} seconds",
                "exit_code": 124,
                "issues_count": 0,
                "tool_available": True,
                "tool_name": tool_name,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Lint error: {str(e)}",
                "exit_code": 1,
                "issues_count": 0,
                "tool_available": True,
                "tool_name": tool_name,
            }

    def run_typecheck(
        self,
        path: str = ".",
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Run mypy type checker on the specified path.

        Args:
            path: Directory or file to check
            strict: Enable strict mode

        Returns:
            Dict with success, stdout, stderr, errors_count
        """
        mypy_path = _find_tool("mypy")

        if not mypy_path:
            return {
                "success": True,  # Don't fail if mypy not available
                "stdout": "mypy not installed - skipping type check",
                "stderr": "",
                "exit_code": 0,
                "errors_count": 0,
                "tool_available": False,
            }

        cmd = [mypy_path, path, "--ignore-missing-imports"]
        if strict:
            cmd.append("--strict")

        print(f"[Executor] Running typecheck: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,
            )

            # Count errors
            errors_count = len([
                line for line in result.stdout.split("\n")
                if ": error:" in line
            ])

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
                "errors_count": errors_count,
                "tool_available": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Type check timed out after {self.timeout * 2} seconds",
                "exit_code": 124,
                "errors_count": 0,
                "tool_available": True,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Type check error: {str(e)}",
                "exit_code": 1,
                "errors_count": 0,
                "tool_available": True,
            }

    def run_all_verification(
        self,
        path: str = ".",
        run_tests: bool = True,
        run_lint: bool = True,
        run_typecheck: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all verification tools and aggregate results.

        Returns:
            Dict with overall success and individual tool results
        """
        results = {
            "success": True,
            "tests": None,
            "lint": None,
            "typecheck": None,
            "summary": [],
        }

        if run_tests:
            results["tests"] = self.run_tests(path)
            if not results["tests"]["success"] and results["tests"]["tool_available"]:
                results["success"] = False
                results["summary"].append(
                    f"Tests: {results['tests']['tests_failed']} failed"
                )
            elif results["tests"]["tool_available"]:
                results["summary"].append(
                    f"Tests: {results['tests']['tests_passed']} passed"
                )

        if run_lint:
            results["lint"] = self.run_lint(path)
            if not results["lint"]["success"] and results["lint"]["tool_available"]:
                results["success"] = False
                results["summary"].append(
                    f"Lint: {results['lint']['issues_count']} issues"
                )
            elif results["lint"]["tool_available"]:
                results["summary"].append("Lint: passed")

        if run_typecheck:
            results["typecheck"] = self.run_typecheck(path)
            if not results["typecheck"]["success"] and results["typecheck"]["tool_available"]:
                results["success"] = False
                results["summary"].append(
                    f"Typecheck: {results['typecheck']['errors_count']} errors"
                )
            elif results["typecheck"]["tool_available"]:
                results["summary"].append("Typecheck: passed")

        return results

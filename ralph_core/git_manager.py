import subprocess
import os
import sys
from typing import Optional, Tuple

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.abspath(__file__))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from agents.common.llm import call_model

class GitManager:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir

    def _run_git(self, args: list) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0, result.stdout.strip() + result.stderr.strip()
        except Exception as e:
            return False, str(e)

    def is_git_repo(self) -> bool:
        success, _ = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return success

    def has_changes(self) -> bool:
        # Check for staged or unstaged changes
        success, output = self._run_git(["status", "--porcelain"])
        return success and bool(output.strip())

    def get_diff(self) -> str:
        success, output = self._run_git(["diff", "HEAD"])
        return output if success else ""

    def generate_commit_message(self) -> str:
        """Uses the Orchestrator to write a commit message based on the diff."""
        diff = self.get_diff()
        if not diff:
            return "refactor: minor updates"
            
        # Truncate diff if too large
        if len(diff) > 2000:
            diff = diff[:2000] + "\n... (truncated)"

        prompt = (
            f"DIFF:\n{diff}\n\n"
            f"You are the Scribe. Write a semantic commit message for these changes.\n"
            f"Format: Conventional Commits (e.g., 'feat: add user login', 'fix: regex error').\n"
            f"Constraint: Return ONLY the commit message. No quotes, no intro.\n"
        )
        return call_model("orchestrator", prompt)

    def commit_all(self, message: Optional[str] = None) -> str:
        """Stages all changes and commits them."""
        if not self.is_git_repo():
            return "Error: Not a git repository."
        
        if not self.has_changes():
            return "No changes to commit."

        if not message:
            print("[Git] Generating commit message...")
            message = self.generate_commit_message()

        # Add all changes
        self._run_git(["add", "."])
        
        # Commit
        success, output = self._run_git(["commit", "-m", message])
        if success:
            return f"Git Commit Success: {message}"
        else:
            return f"Git Commit Failed: {output}"

    def create_branch(self, branch_name: str) -> str:
        """Creates and switches to a new branch."""
        success, output = self._run_git(["checkout", "-b", branch_name])
        if success:
            return f"Switched to new branch: {branch_name}"
        return f"Error creating branch: {output}"

    def checkout(self, branch_name: str) -> str:
        """Switches to an existing branch."""
        success, output = self._run_git(["checkout", branch_name])
        if success:
            return f"Switched to branch: {branch_name}"
        return f"Error switching branch: {output}"

    def revert_last_commit(self) -> str:
        """Reverts the last commit."""
        # git revert HEAD --no-edit
        success, output = self._run_git(["revert", "HEAD", "--no-edit"])
        if success:
            return "Successfully reverted last commit."
        return f"Error reverting: {output}"

# Global instance
git = GitManager()

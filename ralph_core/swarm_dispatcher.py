import subprocess
import os
import time
import sys
from typing import List

RUNNER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")

class SwarmDispatcher:
    def __init__(self):
        pass

    def dispatch(self, subtasks: List[str]) -> str:
        """
        Spawns multiple Ralph instances to handle subtasks in parallel.
        """
        processes = []
        print(f"[Swarm Dispatcher] Launching {len(subtasks)} parallel workers...")
        
        for i, task in enumerate(subtasks):
            instance_id = f"worker_{i+1}"
            print(f"[Swarm Dispatcher] Spawning {instance_id} -> '{task}'")
            
            # Prepare environment
            env = os.environ.copy()
            env["RALPH_INSTANCE_ID"] = instance_id
            
            # Spawn process (detached)
            # We run for limited iterations (e.g., 5) to avoid infinite loops in background
            # Note: runner.py takes OBJECTIVE and ITERATION
            # We'll run a mini-loop here? No, runner.py runs ONE iteration.
            # So we need a wrapper script or loop?
            # Ideally, we spawn a python process that runs a loop.
            # Or simplified: We spawn runner.py iteration 1. If it doesn't finish, we assume it failed for now (MVP).
            # Better: Write a mini-loop wrapper in python.
            
            p = subprocess.Popen(
                [sys.executable, RUNNER_SCRIPT, task, "1"], 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((instance_id, p, task))
            
        # Wait for all
        results = []
        for instance_id, p, task in processes:
            stdout, stderr = p.communicate()
            exit_code = p.returncode
            
            status = "SUCCESS" if exit_code == 0 else "FAILURE"
            results.append(f"Worker: {instance_id}\nTask: {task}\nStatus: {status}\nOutput:\n{stdout[:500]}...")
            
        return "\n\n=== SWARM RESULTS ===\n" + "\n\n".join(results)

# Global Instance
dispatcher = SwarmDispatcher()

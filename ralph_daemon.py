import time
import subprocess
import os
import sys

# Add path to import core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ralph_core.triggers.queue_manager import queue
from ralph_core.dreamer import generate_dream

IDLE_THRESHOLD = 60 # Seconds before dreaming (Short for demo purposes)

def main():
    print("[Daemon] Ralph Daemon Online. Monitoring queue...")
    idle_timer = 0
    
    while True:
        task = queue.get_next_task()
        if task:
            idle_timer = 0 # Reset idle
            print(f"[Daemon] Found task: {task['task']}")
            
            # Execute Ralph Loop
            try:
                print(f"[Daemon] Launching Ralph Loop...")
                subprocess.run(
                    ["./ralph_loop.sh", task['task']],
                    check=True
                )
                print(f"[Daemon] Task Complete.")
                queue.mark_complete(task['task'])
            except Exception as e:
                print(f"[Daemon] Error executing task: {e}")
        else:
            # IDLE STATE
            sys.stdout.write(f"\r[Daemon] Idle... ({idle_timer}/{IDLE_THRESHOLD})")
            sys.stdout.flush()
            
            idle_timer += 10
            if idle_timer >= IDLE_THRESHOLD:
                print("\n[Daemon] Entering Dream Mode...")
                try:
                    dream_task = generate_dream()
                    print(f"[Dreamer] Generated: {dream_task}")
                    queue.add_task("Dreamer", dream_task)
                    idle_timer = 0 # Reset so we don't spam dreams instantly if one is queued
                except Exception as e:
                    print(f"[Dreamer] Error: {e}")
                    idle_timer = 0
        
        time.sleep(10) # Check every 10 seconds

if __name__ == "__main__":
    main()

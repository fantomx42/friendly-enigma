import time
import subprocess
import os
import sys
import random

# Add path to import core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ralph_core.triggers.queue_manager import queue
from ralph_core.dreamer import generate_dream

IDLE_THRESHOLD = 60  # Seconds before entering sleep mode
REM_COOLDOWN = 300   # Minimum seconds between REM cycles (5 minutes)

# Track last REM sleep time
_last_rem_time = 0


def should_consolidate() -> bool:
    """Check if there are enough lessons to warrant consolidation."""
    global_lessons_file = os.path.expanduser("~/.ralph/global_memory/lessons.json")

    if not os.path.exists(global_lessons_file):
        return False

    try:
        import json
        with open(global_lessons_file, 'r') as f:
            lessons = json.load(f)
        return len(lessons) >= 3  # Need at least 3 lessons
    except:
        return False


def check_for_task() -> bool:
    """Check if a task has arrived (used for REM interruption)."""
    task = queue.get_next_task()
    if task:
        # Put it back - we just peeked
        queue.add_task("priority", task['task'])
        return True
    return False


def initiate_rem_sleep_cycle() -> dict:
    """
    Trigger REM Sleep memory consolidation.

    Returns dict with results or error info.
    """
    try:
        from ralph_core.agents.sleeper.agent import initiate_rem_sleep
        return initiate_rem_sleep(
            max_duration=90,
            check_interrupt=check_for_task
        )
    except ImportError as e:
        return {"success": False, "error": f"Import error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    global _last_rem_time

    print("[Daemon] Ralph Daemon Online. Monitoring queue...")
    print("[Daemon] REM Sleep consolidation enabled (REM First strategy)")
    idle_timer = 0

    while True:
        task = queue.get_next_task()
        if task:
            idle_timer = 0  # Reset idle
            print(f"\n[Daemon] Found task: {task['task']}")

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
                current_time = time.time()

                # === REM SLEEP FIRST (per user preference) ===
                # Only run REM if enough time has passed since last cycle
                if current_time - _last_rem_time >= REM_COOLDOWN:
                    if should_consolidate():
                        print("\n[Daemon] Entering REM Sleep Mode...")
                        result = initiate_rem_sleep_cycle()

                        if result.get("success"):
                            print(f"[REM] Consolidated {result.get('new_guidelines', 0)} guidelines")
                            print(f"[REM] Analyzed {result.get('lessons_analyzed', 0)} lessons")
                            _last_rem_time = current_time
                        else:
                            error = result.get("error", "Unknown error")
                            if "Not enough lessons" not in str(error):
                                print(f"[REM] Error: {error}")
                    else:
                        print("\n[Daemon] Skipping REM Sleep (not enough lessons)")

                # === DREAM GENERATION (after REM, 30% chance) ===
                # Generate dreams less frequently since REM takes priority
                if random.random() < 0.3:
                    print("[Daemon] Entering Dream Mode...")
                    try:
                        dream_task = generate_dream()
                        print(f"[Dreamer] Generated: {dream_task}")
                        queue.add_task("Dreamer", dream_task)
                    except Exception as e:
                        print(f"[Dreamer] Error: {e}")

                idle_timer = 0  # Reset so we don't spam

        time.sleep(10)  # Check every 10 seconds


if __name__ == "__main__":
    main()

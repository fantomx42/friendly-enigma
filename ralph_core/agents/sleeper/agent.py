"""
sleeper/agent.py - REM Sleep Agent for Memory Consolidation

Orchestrates the 4-phase REM Sleep cycle:
1. REPLAY  - Load and parse execution history
2. ANALYZE - Cluster similar lessons
3. SYNTHESIZE - Generate meta-guidelines from clusters
4. PERSIST - Save guidelines and log consolidation event
"""

import os
import sys
import json
import time
from typing import Dict, Optional, Callable
from datetime import datetime

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from memory import Memory
from vector_db import vector_memory
from consolidator import (
    consolidate_lessons,
    format_guidelines_markdown,
    SIMILARITY_THRESHOLD
)

# Global memory paths
GLOBAL_MEMORY_DIR = os.path.expanduser("~/.ralph/global_memory")
GUIDELINES_FILE = os.path.join(GLOBAL_MEMORY_DIR, "global_guidelines.md")
CONSOLIDATION_LOG = os.path.join(GLOBAL_MEMORY_DIR, "consolidation_log.json")


class REMSleepController:
    """
    Controller for interruptible REM Sleep cycle.

    Provides cancellation and timeout mechanisms so the daemon
    can abort REM sleep if a new task arrives.
    """

    def __init__(self, max_duration: int = 120):
        """
        Args:
            max_duration: Maximum seconds before forced exit
        """
        self.cancelled = False
        self.start_time = time.time()
        self.max_duration = max_duration
        self.current_phase = "idle"

    def should_continue(self) -> bool:
        """Check if REM sleep should continue."""
        if self.cancelled:
            return False
        if time.time() - self.start_time > self.max_duration:
            return False
        return True

    def cancel(self):
        """Cancel the REM sleep cycle."""
        self.cancelled = True
        print(f"[REM] Cancelled during {self.current_phase} phase")

    def elapsed_ms(self) -> int:
        """Return elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)


def load_execution_history(progress_file: str = "RALPH_PROGRESS.md") -> str:
    """
    Phase 1: REPLAY - Load full execution history.

    Unlike the live Reflector which only reads last 3000 chars,
    REM Sleep analyzes the complete history for deeper patterns.
    """
    if not os.path.exists(progress_file):
        return ""

    try:
        with open(progress_file, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"[REM] Error loading history: {e}")
        return ""


def extract_failure_patterns(history: str) -> Dict:
    """
    Parse execution history to identify failure patterns.

    Returns dict with:
    - error_types: Counter of error categories
    - iteration_failures: Iterations that didn't complete
    - repeated_errors: Errors that occurred multiple times
    """
    patterns = {
        "error_types": {},
        "total_iterations": 0,
        "failed_iterations": 0,
        "error_messages": []
    }

    if not history:
        return patterns

    # Simple parsing - count iterations and errors
    lines = history.split('\n')

    for line in lines:
        lower = line.lower()

        # Count iterations
        if "## iteration" in lower:
            patterns["total_iterations"] += 1

        # Identify error types
        if "error" in lower or "exception" in lower or "failed" in lower:
            if "traceback" in lower or "error:" in lower:
                patterns["error_messages"].append(line[:200])

            # Categorize errors
            if "import" in lower:
                patterns["error_types"]["ImportError"] = patterns["error_types"].get("ImportError", 0) + 1
            elif "file" in lower or "path" in lower:
                patterns["error_types"]["FileError"] = patterns["error_types"].get("FileError", 0) + 1
            elif "syntax" in lower:
                patterns["error_types"]["SyntaxError"] = patterns["error_types"].get("SyntaxError", 0) + 1
            elif "type" in lower:
                patterns["error_types"]["TypeError"] = patterns["error_types"].get("TypeError", 0) + 1
            elif "name" in lower:
                patterns["error_types"]["NameError"] = patterns["error_types"].get("NameError", 0) + 1
            else:
                patterns["error_types"]["Other"] = patterns["error_types"].get("Other", 0) + 1

    return patterns


def persist_guidelines(guidelines: list, consolidation_result: Dict) -> bool:
    """
    Phase 4: PERSIST - Save guidelines and log the consolidation event.
    """
    os.makedirs(GLOBAL_MEMORY_DIR, exist_ok=True)

    # Save markdown guidelines
    try:
        md_content = format_guidelines_markdown(guidelines)
        with open(GUIDELINES_FILE, 'w') as f:
            f.write(md_content)
        print(f"[REM] Saved guidelines to {GUIDELINES_FILE}")
    except Exception as e:
        print(f"[REM] Error saving guidelines: {e}")
        return False

    # Log consolidation event
    try:
        log = {"events": []}
        if os.path.exists(CONSOLIDATION_LOG):
            with open(CONSOLIDATION_LOG, 'r') as f:
                log = json.load(f)

        log["events"].append({
            "timestamp": datetime.now().isoformat(),
            "lessons_analyzed": consolidation_result.get("lessons_analyzed", 0),
            "clusters_found": consolidation_result.get("clusters_found", 0),
            "guidelines_created": consolidation_result.get("guidelines_created", 0),
            "duration_ms": consolidation_result.get("duration_ms", 0)
        })

        # Keep only last 100 events
        log["events"] = log["events"][-100:]

        with open(CONSOLIDATION_LOG, 'w') as f:
            json.dump(log, f, indent=2)

    except Exception as e:
        print(f"[REM] Error logging consolidation: {e}")

    # Index guidelines in vector DB for semantic retrieval
    for g in guidelines:
        vector_memory.add(
            g["text"],
            {
                "tag": "guideline",
                "category": g.get("category", "General"),
                "source": "rem_sleep",
                "confidence": g.get("confidence", 0.8)
            },
            scope="global"
        )

    return True


def should_consolidate(min_lessons: int = 3) -> bool:
    """
    Check if there are enough lessons to warrant consolidation.
    """
    lessons_file = os.path.join(GLOBAL_MEMORY_DIR, "lessons.json")

    if not os.path.exists(lessons_file):
        return False

    try:
        with open(lessons_file, 'r') as f:
            lessons = json.load(f)
        return len(lessons) >= min_lessons
    except:
        return False


def initiate_rem_sleep(
    brain: Memory = None,
    max_duration: int = 120,
    check_interrupt: Callable[[], bool] = None
) -> Dict:
    """
    Main entry point for REM Sleep cycle.

    Orchestrates the 4-phase consolidation process:
    1. REPLAY - Load full execution history
    2. ANALYZE - Cluster similar lessons using semantic similarity
    3. SYNTHESIZE - Generate meta-guidelines from clusters
    4. PERSIST - Save guidelines and audit trail

    Args:
        brain: Memory instance (optional, for future extensions)
        max_duration: Maximum seconds before forced exit
        check_interrupt: Optional callback to check if we should abort

    Returns:
        Dict with:
        - success: bool
        - new_guidelines: int count of guidelines created
        - lessons_analyzed: int
        - duration_ms: int
        - phase_completed: str (last completed phase)
    """
    controller = REMSleepController(max_duration)

    result = {
        "success": False,
        "new_guidelines": 0,
        "lessons_analyzed": 0,
        "clusters_found": 0,
        "duration_ms": 0,
        "phase_completed": "none",
        "failure_patterns": {}
    }

    print("[REM] === Entering REM Sleep Cycle ===")

    # ========== PHASE 1: REPLAY ==========
    controller.current_phase = "replay"
    print("[REM] Phase 1: REPLAY - Loading execution history...")

    if not controller.should_continue():
        result["phase_completed"] = "cancelled_before_replay"
        return result

    history = load_execution_history()
    patterns = extract_failure_patterns(history)
    result["failure_patterns"] = patterns

    print(f"[REM] Found {patterns['total_iterations']} iterations, {len(patterns['error_types'])} error types")

    # Check for interrupt
    if check_interrupt and check_interrupt():
        controller.cancel()
        result["phase_completed"] = "replay"
        return result

    # ========== PHASE 2 & 3: ANALYZE + SYNTHESIZE ==========
    controller.current_phase = "analyze"
    print("[REM] Phase 2: ANALYZE - Clustering lessons...")

    if not controller.should_continue():
        result["phase_completed"] = "replay"
        return result

    # Consolidate lessons (this does both clustering and synthesis)
    consolidation_result = consolidate_lessons(
        vector_db=vector_memory,
        threshold=SIMILARITY_THRESHOLD
    )

    result["lessons_analyzed"] = consolidation_result.get("lessons_analyzed", 0)
    result["clusters_found"] = consolidation_result.get("clusters_found", 0)

    if not consolidation_result.get("success", False):
        print(f"[REM] Consolidation failed: {consolidation_result.get('error', 'Unknown')}")
        result["phase_completed"] = "analyze"
        result["duration_ms"] = controller.elapsed_ms()
        return result

    guidelines = consolidation_result.get("guidelines", [])
    result["new_guidelines"] = len(guidelines)

    print(f"[REM] Phase 3: SYNTHESIZE - Generated {len(guidelines)} guidelines")

    # Check for interrupt
    if check_interrupt and check_interrupt():
        controller.cancel()
        result["phase_completed"] = "synthesize"
        return result

    # ========== PHASE 4: PERSIST ==========
    controller.current_phase = "persist"
    print("[REM] Phase 4: PERSIST - Saving guidelines...")

    if not controller.should_continue():
        result["phase_completed"] = "synthesize"
        return result

    if guidelines:
        persist_success = persist_guidelines(guidelines, consolidation_result)
        if not persist_success:
            result["phase_completed"] = "persist_failed"
            result["duration_ms"] = controller.elapsed_ms()
            return result

    # ========== COMPLETE ==========
    result["success"] = True
    result["phase_completed"] = "complete"
    result["duration_ms"] = controller.elapsed_ms()

    print(f"[REM] === REM Sleep Complete ({result['duration_ms']}ms) ===")
    print(f"[REM] Analyzed {result['lessons_analyzed']} lessons")
    print(f"[REM] Created {result['new_guidelines']} guidelines")

    return result


if __name__ == "__main__":
    # Test REM Sleep cycle
    print("=== Manual REM Sleep Test ===\n")

    result = initiate_rem_sleep(max_duration=180)

    print(f"\n=== Result ===")
    print(json.dumps(result, indent=2, default=str))

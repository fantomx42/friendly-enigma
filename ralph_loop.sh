#!/bin/bash
# ralph_loop.sh - The Outer Control Loop for the Ralph Protocol
# Usage: ./ralph_loop.sh "Your Objective"

OBJECTIVE="$1"
MAX_ITERATIONS=10
ITERATION=1
PROGRESS_FILE="RALPH_PROGRESS.md"
TASK_FILE="RALPH_TASK.md"

# 1. Initialize State
echo "# Ralph Task: $OBJECTIVE" > "$TASK_FILE"
echo "## Current Status: Pending" >> "$TASK_FILE"
touch "$PROGRESS_FILE"

# 2. The Core Mechanical Loop
while [ $ITERATION -le $MAX_ITERATIONS ]; do
    echo "=========================================="
    echo "🔄 Ralph Iteration $ITERATION / $MAX_ITERATIONS"
    echo "=========================================="

    # A. Context Rotation / Clean Slate
    # We pass the content of files, not the history of the conversation.
    # This prevents "Context Pollution."
    
    TASK_CONTENT=$(cat "$TASK_FILE")
    PROGRESS_CONTENT=$(cat "$PROGRESS_FILE")
    
    # B. Execution Phase (Python Bridge to Swarm)
    # We use a dedicated python script to handle the logic cycle
    # (Analyze -> Execute -> Verify) inside the iteration.
    
    python3 ralph_core/runner.py "$OBJECTIVE" "$ITERATION"
    EXIT_CODE=$?

    # C. The Stop Hook & Exit Interception
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Ralph successfully completed the iteration."
        
        # Check for the "Completion Promise" in the task file or runner output
        # (For now, we assume runner.py only exits 0 if it sees <promise>COMPLETE</promise>)
        if grep -q "<promise>COMPLETE</promise>" "$PROGRESS_FILE"; then
            echo "🎉 <promise>COMPLETE</promise> detected. Mission Accomplished."
            exit 0
        fi
    else
        echo "⚠️ Iteration $ITERATION failed or incomplete. Re-injecting..."
        # Deterministic Failure: We don't stop, we just loop again.
    fi

    ITERATION=$((ITERATION+1))
    sleep 2 # Brief pause for thermal regulation
done

echo "❌ Max iterations reached without completion signal."
exit 1

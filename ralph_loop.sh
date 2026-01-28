#!/bin/bash
# ralph_loop.sh - The Outer Control Loop for the Ralph Protocol
# Usage: ./ralph_loop.sh [--sandbox] "Your Objective"

# Argument Parsing
SANDBOX_MODE=0
OBJECTIVE=""
USE_V2=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --sandbox)
      SANDBOX_MODE=1
      shift # past argument
      ;;
    --v2)
      USE_V2=1
      shift
      ;;
    -*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      OBJECTIVE="$1"
      shift # past argument
      ;;
  esac
done

if [ -z "$OBJECTIVE" ]; then
    echo "Usage: ./ralph_loop.sh [--sandbox] [--v2] \"Your Objective\""
    exit 1
fi

if [ $USE_V2 -eq 1 ]; then
    echo "🔥 RALPH V2 MODE ACTIVE"
    export RALPH_V2=1
else
    export RALPH_V2=0
fi

# Sandbox Initialization
if [ $SANDBOX_MODE -eq 1 ]; then
    echo "🛡️  SANDBOX MODE ACTIVE"
    echo "Starting Docker Container..."
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker not found. Cannot run in sandbox mode."
        exit 1
    fi
    
    # Start Container
    (cd sandbox && docker-compose up -d)
    
    export RALPH_SANDBOX=1
else
    export RALPH_SANDBOX=0
fi

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

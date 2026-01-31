#!/bin/bash
# =============================================================================
# Ralph Wiggum + Wheeler Memory Integration (Local Version)
# =============================================================================
#
# A Ralph loop with persistent memory using LOCAL AI (Ollama).
#
# -----------------------------------------------------------------------------
# INSTRUCTIONS & SETUP
# -----------------------------------------------------------------------------
# 1. Prerequisites:
#    - Ensure Ollama is installed and running (https://ollama.com).
#    - Pull your desired model first:
#         ollama pull llama3
#         ollama pull mistral
#    - Ensure Python dependencies are installed for the memory scripts:
#         pip install numpy
#
# 2. Preparation:
#    - Create a text file containing your prompt or task description.
#      Example: echo "Refactor this python script to use classes" > task.md
#
# 3. Usage Examples:
#
#    # Basic usage (uses default 'llama3' model):
#    chmod +x ralph_wheeler_local.sh
#    ./ralph_wheeler_local.sh task.md
#
#    # Use a specific local model (e.g., mistral, codellama, deepseek-r1):
#    ./ralph_wheeler_local.sh task.md --model mistral
#
#    # Set a maximum iteration limit (default is 50) to prevent infinite loops:
#    ./ralph_wheeler_local.sh task.md --max-iterations 10
#
#    # Stop automatically if the AI outputs a specific success token (e.g., "DONE"):
#    ./ralph_wheeler_local.sh task.md --completion-promise "DONE"
#
#    # Customize the output directory (default is ./ralph_output):
#    ./ralph_wheeler_local.sh task.md --output-dir ./my_project_logs
#
# -----------------------------------------------------------------------------
# How it works:
#     1. Reads your PROMPT_FILE.
#     2. Checks ~/.wheeler_memory for similar past attempts (Context).
#     3. Appends that context to your prompt.
#     4. Runs the selected Ollama model.
#     5. Saves the output and analyzes it (success/fail) into memory.
#     6. Repeats until max iterations or completion promise is met.
# =============================================================================

set -e

# Default settings
MAX_ITERATIONS=50
COMPLETION_PROMISE=""
PROMPT_FILE=""
ITERATION=0
OUTPUT_DIR="./ralph_output"
MODEL="llama3"  # Default local model

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --completion-promise)
            COMPLETION_PROMISE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 PROMPT.md [options]"
            echo ""
            echo "Options:"
            echo "  --model NAME            Ollama model to use (default: llama3)"
            echo "  --max-iterations N      Stop after N iterations (default: 50)"
            echo "  --completion-promise S  Stop when output contains S"
            echo "  --output-dir DIR        Directory for output files"
            exit 0
            ;;
        *)
            PROMPT_FILE="$1"
            shift
            ;;
    esac
done

# Validate
if [[ -z "$PROMPT_FILE" ]]; then
    echo "Error: No prompt file specified"
    echo "Usage: $0 PROMPT.md [options]"
    exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory for Python files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Ralph Wiggum + Wheeler Memory (Local)"
echo "=============================================="
echo "Prompt: $PROMPT_FILE"
echo "Model: $MODEL (Ollama)"
echo "Max iterations: $MAX_ITERATIONS"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Main loop
while [[ $ITERATION -lt $MAX_ITERATIONS ]]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo ">>> Iteration $ITERATION / $MAX_ITERATIONS"
    echo "-------------------------------------------"

    # Read the prompt
    PROMPT=$(cat "$PROMPT_FILE")

    # Recall relevant context from Wheeler Memory
    echo "[Wheeler] Recalling relevant context..."
    CONTEXT=$(python3 "$SCRIPT_DIR/wheeler_recall.py" "$PROMPT" 2>/dev/null || echo "")

    # Build the full prompt
    if [[ -n "$CONTEXT" ]]; then
        FULL_PROMPT="$CONTEXT

---

$PROMPT"
    else
        FULL_PROMPT="$PROMPT"
    fi

    # Save the full prompt for debugging
    echo "$FULL_PROMPT" > "$OUTPUT_DIR/iteration_${ITERATION}_prompt.txt"

    # Run Local LLM (Ollama)
    echo "[Ollama:$MODEL] Processing..."
    OUTPUT_FILE="$OUTPUT_DIR/iteration_${ITERATION}_output.txt"

    # Feed to Ollama and capture output
    # Note: We use process substitution to pipe cleanly to tee
    echo "$FULL_PROMPT" | ollama run "$MODEL" 2>&1 | tee "$OUTPUT_FILE"

    # Store this iteration in Wheeler Memory
    echo ""
    echo "[Wheeler] Storing iteration result..."
    python3 "$SCRIPT_DIR/wheeler_store.py" "$OUTPUT_FILE" --type iteration 2>&1 || true

    # Check for completion promise
    if [[ -n "$COMPLETION_PROMISE" ]]; then
        if grep -q "$COMPLETION_PROMISE" "$OUTPUT_FILE"; then
            echo ""
            echo "=============================================="
            echo "✓ Completion promise found: $COMPLETION_PROMISE"
            echo "✓ Finished after $ITERATION iterations"
            echo "=============================================="
            exit 0
        fi
    fi

    # Small delay (optional for local, but good for stability)
    sleep 1
done

echo ""
echo "=============================================="
echo "⚠ Max iterations reached ($MAX_ITERATIONS)"
echo "=============================================="
exit 1

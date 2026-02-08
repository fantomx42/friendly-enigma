#!/bin/bash
# =============================================================================
# Ralph Loop — Entry point for the Ralph AI iterative loop
# =============================================================================
#
# Delegates to ralph_simple.py which uses persistent streaming
# connections to Ollama for fast iterative inference.
#
# Usage:
#   ./ralph_loop.sh "Your objective here"
#   RALPH_MODEL=qwen3:8b ./ralph_loop.sh "Your objective"
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv if it exists
if [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

PYTHONPATH="$SCRIPT_DIR" exec python3 "$SCRIPT_DIR/ralph_simple.py" "$@"

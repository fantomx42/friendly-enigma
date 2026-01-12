#!/bin/bash
# Start the Ralph UI
# Usage: sudo ./start_ui.sh (if using port 80)

# Ensure python path includes current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Ralph UI on http://ralph.ai (Port 80)..."
python3 ralph_ui/backend/main.py
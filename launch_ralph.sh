#!/bin/bash
# launch_ralph.sh
# Master launcher for Ralph AI Desktop App

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Launching Ralph AI..."

# Check if server is already running
if pgrep -f "llama-server" > /dev/null; then
    echo "✅ Server already running."
else
    echo "🔌 Starting Ralph v2 Server (GPU + NPU)..."
    ./start_ralph_v2.sh > ralph_server.log 2>&1 &
    SERVER_PID=$!
    echo "   (PID: $SERVER_PID)"
    
    # Wait for server to warm up (basic check)
    echo "⏳ Waiting for engine ignition..."
    sleep 5
fi

# Launch GUI
echo "🖥️  Opening Dashboard..."
cd ralph_gui
./target/release/ralph_gui

# Cleanup when GUI closes
echo "👋 Dashboard closed."
if [ ! -z "$SERVER_PID" ]; then
    echo "🛑 Stopping Server..."
    kill $SERVER_PID 2>/dev/null
    # Also kill the librarian daemon if it was spawned by start_ralph_v2
    pkill -f "librarian_daemon.py"
fi

exit 0

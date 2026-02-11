#!/bin/bash
# TWAI Mission Control Launcher
PROJECT_DIR="/home/tristan/ralph/twai"

# Kill any existing instances
pkill -f "twai/backend" || true
pkill -f "trunk serve" || true

echo "Starting TWAI Services..."

# Start Backend in background
cd "$PROJECT_DIR/backend"
cargo run > /tmp/twai_backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend in background
cd "$PROJECT_DIR/frontend"
trunk serve > /tmp/twai_frontend.log 2>&1 &
FRONTEND_PID=$!

echo "Initializing (PID Backend: $BACKEND_PID, Frontend: $FRONTEND_PID)..."

# Wait for ports to open
MAX_RETRIES=30
COUNT=0

while ! curl -s http://127.0.0.1:8080 > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    echo -n "."
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo -e "\nError: Frontend failed to start. Check /tmp/twai_frontend.log"
        exit 1
    fi
done

echo -e "\nServices are online. Opening Mission Control..."
xdg-open http://127.0.0.1:8080
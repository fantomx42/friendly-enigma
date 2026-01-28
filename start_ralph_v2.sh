#!/bin/bash
# Ralph v2 Launcher

# Cleanup on exit
trap 'kill $LIBRARIAN_PID 2>/dev/null; exit' SIGINT SIGTERM

# 1. Start the Librarian (NPU) in the background
echo "📚 Starting Librarian Daemon (NPU Memory Management)..."
./venv/bin/python ralph_core/librarian_daemon.py &
LIBRARIAN_PID=$!

# 2. Start the Inference Server (GPU + CPU)
# Paths
MODEL_DIR="/home/tristan/ralph_brain/models"
CHALMERS_MODEL="$MODEL_DIR/chalmers/chalmers-qwen-14b.gguf"
WIGGUM_MODEL="$MODEL_DIR/wiggum/wiggum-qwen-1.5b.gguf"
SERVER_BIN="./vendor/llama.cpp/build/bin/llama-server"

# Check Models
if [ ! -f "$CHALMERS_MODEL" ]; then
    echo "⚠️  Chalmers (Main Model) not found at $CHALMERS_MODEL"
    echo "Please place the Qwen 14B GGUF file there."
    kill $LIBRARIAN_PID
    exit 1
fi

if [ ! -f "$WIGGUM_MODEL" ]; then
    echo "⚠️  Wiggum (Draft Model) not found at $WIGGUM_MODEL"
    echo "Please place the Qwen 1.5B GGUF file there."
    kill $LIBRARIAN_PID
    exit 1
fi

echo "🔥 Igniting Ralph v2 (CPU Fallback Mode)..."
$SERVER_BIN \
    --model "$CHALMERS_MODEL" \
    --n-gpu-layers 0 \
    --threads 8 \
    --port 8080 \
    --ctx-size 2048 \
    --parallel 1 \
    --cont-batching

# Kill librarian when server stops
kill $LIBRARIAN_PID

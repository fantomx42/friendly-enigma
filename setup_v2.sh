#!/bin/bash
set -e

# Add local cmake to PATH
export PATH=$PWD/cmake-3.29.0-linux-x86_64/bin:$PATH

echo "🚀 Ralph v2: Tri-Brid Setup Initialization"
echo "=========================================="

# 1. Verification
if ! command -v hipcc &> /dev/null; then
    echo "❌ Error: ROCm (hipcc) not found. Please install ROCm."
    exit 1
fi
echo "✅ ROCm detected."

if ! command -v git &> /dev/null; then
    echo "❌ Error: git not found."
    exit 1
fi

# 2. Directory Setup (Redundant but safe)
BRAIN_DIR="/home/tristan/ralph_brain"
echo "🧠 Verifying Brain Structure at $BRAIN_DIR..."
mkdir -p "$BRAIN_DIR"/{hot,cold/{summaries,archive},models/{chalmers,wiggum,librarian}}
touch "$BRAIN_DIR/hot/active_task.md"
touch "$BRAIN_DIR/hot/scratchpad.md"

# 3. Llama.cpp Build (Vendor Strategy)
VENDOR_DIR="vendor"
mkdir -p "$VENDOR_DIR"

if [ ! -d "$VENDOR_DIR/llama.cpp" ]; then
    echo "📦 Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$VENDOR_DIR/llama.cpp"
else
    echo "📦 llama.cpp already exists, pulling updates..."
    cd "$VENDOR_DIR/llama.cpp" && git pull && cd - > /dev/null
fi

echo "🛠️  Building llama.cpp (Server) with ROCm/HIP..."
echo "   Targeting GPU: RX 9070 XT (Assuming GFX1200/1201 or Native)"

cd "$VENDOR_DIR/llama.cpp"

# Build with CMake (ROCm/HIP)
mkdir -p build
cd build
cmake .. -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=native
cmake --build . --config Release -j $(nproc) --target llama-server

echo "✅ Build Complete."
cd ../../..

# Update binary path in launch script
SERVER_BIN="./vendor/llama.cpp/build/bin/llama-server"

# 4. NPU / Librarian Setup
echo "🧠 Setting up Librarian (NPU) environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
# Install requirements (excluding chromadb/sentence-transformers for now due to onnx conflict)
pip install requests pydantic python-dotenv watchdog openvino openvino-genai ollama fastapi uvicorn websockets

# Create a mock model directory if it doesn't exist to prevent immediate crashes
# In a real scenario, the user would use 'optimum-cli' to export Phi-3 to OpenVINO.
if [ ! -d "$BRAIN_DIR/models/librarian/openvino_model" ]; then
    echo "💡 Note: You will need to export a model to OpenVINO format for the NPU."
    echo "   Example: ./venv/bin/pip install optimum-intel[openvino,nncf] && ./venv/bin/optimum-cli export openvino --model microsoft/Phi-3-mini-4k-instruct --task text-generation-with-past --weight-format int4 $BRAIN_DIR/models/librarian/"
fi

# 5. Create Launch Script
cat > start_ralph_v2.sh << 'EOF'
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

echo "🔥 Igniting Ralph v2 (Chalmers + Wiggum)..."
$SERVER_BIN \
    --model "$CHALMERS_MODEL" \
    --model-draft "$WIGGUM_MODEL" \
    --n-gpu-layers 99 \
    --draft-max 8 \
    --threads 8 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 2 \
    --cont-batching

# Kill librarian when server stops
kill $LIBRARIAN_PID
EOF

chmod +x start_ralph_v2.sh

echo "=========================================="
echo "🎉 Setup Complete!"
echo "1. Download your GGUF models to: $BRAIN_DIR/models/"
echo "2. Run: ./start_ralph_v2.sh"

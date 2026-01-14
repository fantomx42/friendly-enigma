#!/bin/bash
# setup_voice.sh - Installs dependencies for Ralph's Voice Interface

echo "🎙️  Setting up Ralph Voice Interface..."

# 1. Install Python Audio/STT Deps
echo "📦 Installing Python dependencies (openai-whisper, sounddevice, scipy)..."
pip install openai-whisper sounddevice scipy numpy

# 2. Install Piper (TTS)
echo "🗣️  Setting up Piper TTS..."
mkdir -p ralph_core/voice_models
cd ralph_core/voice_models

# Download Piper Binary (Linux x86_64) if not present
if [ ! -f "piper" ]; then
    echo "Downloading Piper binary..."
    wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
    tar -xvf piper.tar.gz
    cp piper/piper .
    rm -rf piper piper.tar.gz
fi

# Download a Voice Model (e.g., en_US-lessac-medium)
if [ ! -f "en_US-lessac-medium.onnx" ]; then
    echo "Downloading Voice Model (en_US-lessac-medium)..."
    wget -O en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -O en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
fi

echo "✅ Voice Setup Complete."
echo "Run 'python3 ralph_voice.py' to talk to Ralph."

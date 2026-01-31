#!/bin/bash
# Wheeler AI Training Environment Setup
# For CachyOS with AMD Radeon RX 9070 XT (RDNA4)
# =============================================

set -e

echo "=========================================="
echo "Wheeler AI - Training Environment Setup"
echo "=========================================="
echo ""

# Check if running on Arch-based system
if ! command -v pacman &> /dev/null; then
    echo "Error: This script is for Arch-based systems (CachyOS)"
    exit 1
fi

echo "[1/6] Installing system dependencies..."
sudo pacman -S --needed --noconfirm \
    python python-pip \
    rocm-hip-runtime rocm-hip-sdk \
    rocm-opencl-runtime rocm-opencl-sdk \
    miopen-hip \
    git wget curl \
    htop btop nvtop \
    base-devel cmake ninja

echo ""
echo "[2/6] Setting up ROCm environment..."

# Add user to video and render groups (required for GPU access)
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Set ROCm environment variables
cat >> ~/.bashrc << 'EOF'

# ROCm Environment for Wheeler AI Training
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/hip/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/hip/lib

# AMD GPU target architecture
# RX 9070 XT is RDNA4 = gfx1201 (check with rocminfo if different)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx1100;gfx1101;gfx1102"

# Performance tuning
export GPU_MAX_HW_QUEUES=8
export AMD_SERIALIZE_KERNEL=3
EOF

# Source for current session
source ~/.bashrc

echo ""
echo "[3/6] Creating Python virtual environment..."

# Create project directory
mkdir -p ~/wheeler_ai_training
cd ~/wheeler_ai_training

# Create venv
python -m venv venv
source venv/bin/activate

echo ""
echo "[4/6] Installing PyTorch with ROCm support..."

# Install PyTorch with ROCm 6.2 (latest stable as of Jan 2026)
# Check https://pytorch.org/get-started/locally/ for updates
pip install --upgrade pip wheel setuptools

# PyTorch ROCm build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "[5/6] Installing ML dependencies..."

pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    einops \
    wandb \
    tensorboard \
    matplotlib \
    seaborn \
    tqdm \
    numpy \
    scipy \
    numba \
    pillow

# Optional: Flash Attention for AMD (Triton backend)
# pip install triton
# FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install flash-attn --no-build-isolation

echo ""
echo "[6/6] Verifying installation..."

python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available (ROCm): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Quick test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"Matrix multiply test: OK ({y.shape})")
else:
    print("WARNING: GPU not detected!")
    print("You may need to:")
    print("  1. Reboot after adding user to video/render groups")
    print("  2. Check ROCm installation with 'rocminfo'")
    print("  3. Verify GPU with 'rocm-smi'")

import transformers
print(f"Transformers version: {transformers.__version__}")

from datasets import load_dataset
print("HuggingFace datasets: OK")
PYEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to log out and back in"
echo "(or reboot) for group permissions to take effect."
echo ""
echo "To activate the environment:"
echo "  cd ~/wheeler_ai_training"
echo "  source venv/bin/activate"
echo ""
echo "To verify GPU access:"
echo "  rocm-smi"
echo "  rocminfo"
echo ""
echo "Next: Copy the training code to ~/wheeler_ai_training/"

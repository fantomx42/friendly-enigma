#!/bin/bash
# Wheeler AI Training Setup for CachyOS + AMD RX 9070 XT
# ========================================================
#
# This script sets up everything needed to train the Wheeler AI
# text encoder/decoder on your system.
#
# Hardware target:
#   - Intel Core Ultra 7 265K (20 cores)
#   - AMD Radeon RX 9070 XT (RDNA4)
#   - 32GB DDR5-7000
#
# What gets installed:
#   1. ROCm (AMD's CUDA equivalent)
#   2. PyTorch with ROCm support
#   3. Training dependencies
#   4. TinyStories dataset
#   5. Wheeler AI training code

set -e  # Exit on error

echo "========================================"
echo "Wheeler AI Training Environment Setup"
echo "========================================"
echo ""

# Check if running as root for system packages
if [ "$EUID" -ne 0 ]; then
    echo "Some steps require root. You may be prompted for sudo password."
fi

# -----------------------------------------------------------------------------
# STEP 1: Install ROCm
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Checking ROCm installation..."

if command -v rocminfo &> /dev/null; then
    echo "  ROCm already installed:"
    rocminfo | grep "Marketing Name" | head -1
else
    echo "  Installing ROCm..."
    echo "  Note: CachyOS may have ROCm in repos, otherwise:"
    echo ""
    echo "  For Arch/CachyOS, add the ROCm repo:"
    echo "  ----------------------------------------"
    cat << 'EOF'
  # Add to /etc/pacman.conf:
  [ROCm]
  SigLevel = Never
  Server = https://repo.radeon.com/rocm/arch/rocm-6.3.0/
  
  # Then install:
  sudo pacman -Sy rocm-hip-runtime rocm-opencl-runtime
  
  # Or use the AUR:
  paru -S rocm-hip-sdk
EOF
    echo ""
    echo "  After installing ROCm, re-run this script."
    echo "  Alternatively, we can proceed with CPU-only for now."
    read -p "  Continue with CPU-only setup? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    USE_CPU_ONLY=1
fi

# Check GPU is detected
if command -v rocminfo &> /dev/null; then
    echo "  Checking GPU detection..."
    if rocminfo | grep -q "9070\|gfx12"; then
        echo "  ✓ RX 9070 XT detected!"
        # RDNA4 is gfx12xx architecture
        export PYTORCH_ROCM_ARCH="gfx1200"
    else
        echo "  GPU not detected by ROCm. You may need to:"
        echo "    1. Add user to 'render' and 'video' groups:"
        echo "       sudo usermod -aG render,video $USER"
        echo "    2. Reboot"
        echo "    3. Check with: rocminfo"
    fi
fi

# -----------------------------------------------------------------------------
# STEP 2: Create Python environment
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Setting up Python environment..."

WHEELER_DIR="$HOME/wheeler_ai_training"
mkdir -p "$WHEELER_DIR"
cd "$WHEELER_DIR"

# Check for python
if ! command -v python3 &> /dev/null; then
    echo "  Python3 not found. Install with: sudo pacman -S python"
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "  ✓ Virtual environment activated"

# -----------------------------------------------------------------------------
# STEP 3: Install PyTorch with ROCm
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing PyTorch..."

if [ -z "$USE_CPU_ONLY" ]; then
    echo "  Installing PyTorch with ROCm support..."
    # ROCm 6.3 is latest as of early 2026
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
else
    echo "  Installing PyTorch (CPU only)..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
fi

# Verify
python -c "import torch; print(f'  PyTorch {torch.__version__}')"
python -c "import torch; print(f'  CUDA/ROCm available: {torch.cuda.is_available()}')"
if [ -z "$USE_CPU_ONLY" ]; then
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
fi

# -----------------------------------------------------------------------------
# STEP 4: Install training dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing training dependencies..."

pip install \
    numpy \
    datasets \
    transformers \
    tokenizers \
    accelerate \
    wandb \
    tqdm \
    einops \
    pillow \
    matplotlib \
    numba

# Optional: Flash Attention for AMD (Triton backend)
echo "  Attempting Flash Attention install (may fail on new GPUs)..."
pip install triton || echo "  Triton install failed - will use standard attention"

echo "  ✓ Dependencies installed"

# -----------------------------------------------------------------------------
# STEP 5: Download TinyStories dataset
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Downloading TinyStories dataset..."

python << 'EOF'
from datasets import load_dataset
import os

cache_dir = os.path.expanduser("~/wheeler_ai_training/data")
os.makedirs(cache_dir, exist_ok=True)

print("  Downloading TinyStories (this may take a few minutes)...")
dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)

print(f"  ✓ Train: {len(dataset['train']):,} examples")
print(f"  ✓ Val: {len(dataset['validation']):,} examples")

# Save a sample
sample = dataset['train'][0]['text']
print(f"  Sample story: {sample[:100]}...")
EOF

# -----------------------------------------------------------------------------
# STEP 6: Create project structure
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Creating project structure..."

mkdir -p "$WHEELER_DIR"/{models,checkpoints,logs,data}

# Create a test script
cat > "$WHEELER_DIR/test_setup.py" << 'EOF'
"""Test that the training environment is properly configured."""
import torch
import numpy as np
from datasets import load_dataset

print("=" * 50)
print("Wheeler AI Training Environment Test")
print("=" * 50)

# Test PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quick GPU test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"GPU compute test: ✓")

# Test dataset
print("\nLoading TinyStories sample...")
ds = load_dataset("roneneldan/TinyStories", split="train[:100]")
print(f"Sample loaded: {len(ds)} examples")
print(f"Example: {ds[0]['text'][:80]}...")

# Test numpy/numba
print("\nTesting Numba JIT...")
try:
    from numba import jit
    @jit(nopython=True)
    def test_numba(x):
        return x * 2
    result = test_numba(np.array([1, 2, 3]))
    print(f"Numba test: ✓")
except Exception as e:
    print(f"Numba test: Failed ({e})")

print("\n" + "=" * 50)
print("Environment ready for training!")
print("=" * 50)
EOF

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Project directory: $WHEELER_DIR"
echo ""
echo "To activate the environment:"
echo "  cd $WHEELER_DIR"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup:"
echo "  python test_setup.py"
echo ""
echo "Next steps:"
echo "  1. Run the test script to verify GPU detection"
echo "  2. Copy the model code to $WHEELER_DIR/models/"
echo "  3. Start training with: python train.py"
echo ""

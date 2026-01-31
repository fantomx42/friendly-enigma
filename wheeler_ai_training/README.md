# Wheeler AI Training Setup

## Your System
- **CPU:** Intel Core Ultra 7 265K (20 cores)
- **GPU:** AMD Radeon RX 9070 XT (RDNA4)
- **RAM:** 32GB DDR5-7000
- **OS:** CachyOS Linux

## Quick Start (5 steps)

### Step 1: Install ROCm (AMD's GPU compute stack)

CachyOS should have ROCm in the repos. Open a terminal:

```bash
# Install ROCm packages
sudo pacman -S rocm-hip-runtime rocm-hip-sdk rocm-opencl-runtime

# Add yourself to the GPU groups
sudo usermod -aG render,video $USER

# IMPORTANT: Log out and back in (or reboot) for groups to take effect
```

After logging back in, verify it works:

```bash
rocminfo | grep "Marketing Name"
```

You should see your RX 9070 XT listed.

### Step 2: Create project directory

```bash
mkdir -p ~/wheeler_ai_training
cd ~/wheeler_ai_training
```

### Step 3: Set up Python environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 4: Install PyTorch with ROCm

```bash
# Install PyTorch with AMD GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify GPU is detected
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

If the GPU isn't detected, you may need to set the architecture:

```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.0
export PYTORCH_ROCM_ARCH="gfx1200"
```

Add these to your `~/.bashrc` to make them permanent.

### Step 5: Install dependencies and download the code

```bash
# Install training dependencies
pip install numpy datasets transformers tokenizers accelerate wandb tqdm einops pillow matplotlib numba

# Copy the Wheeler AI files to your project directory
# (You downloaded these from Claude)
cp models.py train.py ~/wheeler_ai_training/

# Download the TinyStories dataset (first run will cache it)
python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories')"
```

## Start Training

Basic training (uses default settings):

```bash
cd ~/wheeler_ai_training
source venv/bin/activate
python train.py
```

### Recommended settings for your hardware

Your 9070 XT has plenty of VRAM, so you can use larger batches:

```bash
python train.py \
    --batch_size 64 \
    --grid_size 64 \
    --d_model 256 \
    --epochs 10 \
    --fp16
```

### Start smaller for testing

If you want to verify everything works first:

```bash
python train.py \
    --batch_size 16 \
    --grid_size 32 \
    --max_train_samples 10000 \
    --epochs 1
```

This will train on 10K samples with a 32x32 grid (faster to iterate).

## Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--grid_size` | 64 | Size of 2D latent grid (64x64 = 4,096 spatial tokens) |
| `--d_model` | 256 | Hidden dimension of the model |
| `--batch_size` | 32 | Samples per batch (increase for faster training) |
| `--epochs` | 10 | Number of passes through the dataset |
| `--fp16` | off | Mixed precision training (faster, less memory) |
| `--lr` | 3e-4 | Learning rate |
| `--max_train_samples` | all | Limit training data (for quick tests) |

## Expected Training Time

On your RX 9070 XT with the default 21M parameter model:

| Grid Size | Batch Size | Time per Epoch | Total (10 epochs) |
|-----------|------------|----------------|-------------------|
| 32x32 | 64 | ~30 min | ~5 hours |
| 64x64 | 32 | ~2 hours | ~20 hours |
| 64x64 | 64 | ~1.5 hours | ~15 hours |

## What's Being Trained

```
Your Text: "Once upon a time there was a little rabbit"
           │
           ▼
    ┌──────────────┐
    │   ENCODER    │  (Text → 2D Grid)
    │              │
    │  Cross-attention maps variable-length
    │  text into fixed 64x64 spatial grid
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   2D GRID    │  ← This is Wheeler Memory compatible!
    │   64 x 64    │
    │   x 256 dim  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   DECODER    │  (2D Grid → Text)
    │              │
    │  Autoregressive generation from
    │  spatial features back to words
    └──────┬───────┘
           │
           ▼
Output: "Once upon a time there was a little rabbit"
```

The model learns to compress text into a 2D spatial representation and reconstruct it. Once trained, this grid is compatible with Wheeler Memory dynamics.

## Monitor Training

Watch GPU usage:

```bash
# AMD GPU monitor
watch -n 1 rocm-smi
```

Or use nvtop (works with AMD too):

```bash
sudo pacman -S nvtop
nvtop
```

## After Training

Once you have a trained model, you can:

1. **Encode text to Wheeler frames:**
```python
from models import WheelerTextAutoencoder
model = WheelerTextAutoencoder.load_from_checkpoint("checkpoints/best.pt")
grid = model.encode("Hello world")  # (1, 64, 64, 256)
```

2. **Convert to Wheeler Memory format:**
```python
wheeler_frame = model.get_grid_as_image(grid)  # (1, 64, 64) in [-1, 1]
```

3. **Run Wheeler dynamics on it:**
```python
from wheeler_cpu import WheelerMemoryCPU
dynamics = WheelerMemoryCPU(64, 64)
# Use the encoded frame as input to dynamics...
```

4. **Decode back to text:**
```python
text = model.decode(grid, start_token_id=tokenizer.bos_token_id, end_token_id=tokenizer.eos_token_id)
```

## Troubleshooting

**"ROCm not found" or GPU not detected:**
```bash
# Check ROCm installation
rocminfo

# Make sure you're in the right groups
groups  # Should include 'render' and 'video'

# Try setting the architecture manually
export HSA_OVERRIDE_GFX_VERSION=12.0.0
```

**Out of memory:**
```bash
# Reduce batch size
python train.py --batch_size 16

# Or reduce grid size
python train.py --grid_size 32
```

**Slow training:**
```bash
# Enable mixed precision
python train.py --fp16

# Use torch.compile (PyTorch 2.0+)
python train.py --compile
```

## Files in This Package

| File | Description |
|------|-------------|
| `models.py` | Neural network architecture (encoder + decoder) |
| `train.py` | Training script |
| `setup_training_env.sh` | Automated setup script (optional) |
| `README.md` | This file |

## Next Steps

After training completes:

1. Test the autoencoder reconstructs text well
2. Integrate with Wheeler Memory dynamics
3. Train on larger/different datasets
4. Experiment with grid-space manipulations (interpolation, arithmetic)

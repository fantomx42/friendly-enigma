# Wheeler Memory L3 Cortex Classifier

## Overview

Two new files implement a ~11K parameter neural network classifier for MMLU answer selection, integrating cortex features (settlement, choice similarities, SCM layer scores) into a 4-way classification.

## Files Created

### 1. `wheeler_memory/cortex_classifier.py` (7.8 KB)

Pure numpy neural network with manual backprop and Xavier initialization.

**Architecture**:
- Input: 21-dimensional feature vector (K=10 settlement + 4 choice sims + 7 SCM layers)
- Layer 1: 21 → 128 (ReLU)
- Layer 2: 128 → 64 (ReLU)
- Output: 64 → 4 (softmax)
- Parameters: 11,460

**API**:

```python
@dataclass
class ClassifierWeights:
    w1, b1, w2, b2, w3, b3  # Network weights and biases

def init_weights(input_dim=21, seed=42) -> ClassifierWeights
    """Xavier initialization"""

def forward(x: np.ndarray, weights) -> np.ndarray
    """Forward pass → (4,) softmax probs"""

def classify(settlement, choice_sims, scm_layers, weights) -> (int, float)
    """Classify → (predicted_index, confidence)"""

def cross_entropy_loss(probs, target) -> float
    """Cross-entropy loss for single example"""

def backward(x, weights, target, learning_rate=0.001) -> (updated_weights, loss)
    """Single-example SGD step with full manual backprop"""

def save_weights(weights, path)
    """Save to .npz file"""

def load_weights(path) -> ClassifierWeights
    """Load from .npz file"""
```

**Key Features**:
- Numerically stable softmax (subtracts max for numerical stability)
- Full manual backpropagation through 3-layer network
- ReLU activations with proper gradient masking
- Pure numpy, no torch/scipy
- Save/load via np.savez/np.load

### 2. `scripts/train_cortex_classifier.py` (14 KB)

Training harness for MMLU dev+validation data.

**Usage**:
```bash
python scripts/train_cortex_classifier.py --epochs 10 --lr 0.001
python scripts/train_cortex_classifier.py --subjects abstract_algebra --epochs 5
python scripts/train_cortex_classifier.py --seed 42 --output /tmp/classifier.npz
```

**CLI Arguments**:
- `--subjects`: MMLU subjects to train on (default: all 57)
- `--samples-per-subject`: Max samples per subject (default: all)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--seed`: Random seed (default: 42)
- `--output`: Output path for weights (default: ~/.wheeler_memory/cortex_classifier.npz)
- `--data-dir`: Wheeler Memory data directory

**Training Pipeline**:

1. **Data Loading** (`load_training_data`):
   - Loads MMLU dev+validation data
   - Encodes questions and choices with hippocampus
   - Retrieves top-K attractors from AttractorCache (if available)
   - Computes choice similarities (Pearson correlation with question attractor)
   - Runs cortex reasoning (graph building + settlement diffusion)
   - Computes SCM scores (7 layers: Temperature, Salience, Energy, Integration, Polarity, Net Warrant, Explanation Readiness)
   - Builds feature vectors: settlement + choice_sims + scm_layers

2. **Training** (`train_classifier`):
   - Shuffles data each epoch
   - SGD with learning_rate
   - Validation split (default 20%)
   - Prints loss and accuracy per epoch

3. **Save**:
   - Saves weights to .npz file

**Feature Vector Construction** (21 dimensions):
- Settled opinions: (K,) from L2 settlement CA
- Choice similarities: (4,) Pearson r of each choice to question attractor
- SCM layers: (7,) [T, S, E, I, P, NW, ERF]

### 3. `wheeler_memory/constants.py` — New Constants Added

```python
# Cortex parameters (tunable via this file)
CORTEX_K = 10                          # retrieved attractors for cortex input
CORTEX_CLUSTER_THRESHOLD = 0.5         # Pearson r for same-cluster
CORTEX_CONTRADICTION_THRESHOLD = 0.3   # negative r for contradiction flag
CORTEX_SETTLE_MAX_STEPS = 100
CORTEX_SETTLE_THRESHOLD = 1e-4
CORTEX_SETTLE_INERTIA = 0.8
CORTEX_CLASSIFIER_LR = 0.001
CORTEX_CLASSIFIER_PATH = "cortex_classifier.npz"  # relative to data_dir
```

## Integration Points

### Cortex Module Integration

The classifier integrates three cortex components:

1. **L1 Graph Reasoning** (`cortex.py`):
   - Builds attractor correlation adjacency matrix
   - Identifies clusters and bridges
   - Detects contradictions

2. **L2 Settlement Dynamics** (`cortex.py`):
   - Diffuses opinions via correlation-weighted averaging
   - Returns settled opinions as classifier input

3. **Structural Coherence Measure** (`cortex_scm.py`):
   - Seven scoring layers (T, S, E, I, P, NW, ERF)
   - Unified coherence score for answer validity

### Data Flow

```
Question → Hippocampus → Frame ↓
         Choice A/B/C/D → Frames → AttractorCache Search

Similarities → cortex_reason() → Graph + Settlement
             → compute_scm() → SCM layers

[Settlement, Choice_Sims, SCM_Layers] → Classifier → [A, B, C, D]
```

## Testing

All functions have been tested:

```
✓ Weight initialization (Xavier scaling)
✓ Forward pass (softmax normalization)
✓ Classification (argmax + confidence)
✓ Cross-entropy loss
✓ Backprop (gradient computation and weight updates)
✓ Save/load (np.savez/np.load round-trip)
✓ Pearson correlation helper
✓ Mini SGD training loop
```

## Parameter Tuning

Since this is autoresearch-style parameter tuning, only `constants.py` should be modified:

```python
# Cortex thresholds
CORTEX_CLUSTER_THRESHOLD = 0.5         # [0.3 .. 0.7] for clustering sensitivity
CORTEX_CONTRADICTION_THRESHOLD = 0.3   # [0.1 .. 0.5] for contradiction detection

# Settlement dynamics
CORTEX_SETTLE_MAX_STEPS = 100          # [50 .. 200] for convergence speed
CORTEX_SETTLE_THRESHOLD = 1e-4         # [1e-6 .. 1e-2] for precision
CORTEX_SETTLE_INERTIA = 0.8            # [0.5 .. 0.9] for momentum

# Classifier training
CORTEX_CLASSIFIER_LR = 0.001           # [0.0001 .. 0.01] for step size
```

## Future Enhancements

1. **Batch training**: Vectorize training loop for faster convergence
2. **Momentum/Adam**: Replace vanilla SGD with adaptive optimizers
3. **Regularization**: Add L2/L1 penalties to prevent overfitting
4. **Validation plot**: Track loss/accuracy curves during training
5. **Early stopping**: Stop on validation plateau
6. **Choice-specific SCM**: Compute SCM per choice, not just best choice
7. **Attractor features**: Use top-K attractor embeddings directly instead of just similarities

"""L3 Neural Network Classifier for MMLU answer selection.

Implements a ~11K parameter 3-layer network for choosing among 4 answer choices
given cortex features: settlement, choice similarities, and SCM layer scores.

Pure numpy implementation with manual backprop and Xavier initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClassifierWeights:
    """Neural network weights for the cortex classifier.

    Attributes:
        w1: (input_dim, 128) first layer weights
        b1: (128,) first layer bias
        w2: (128, 64) second layer weights
        b2: (64,) second layer bias
        w3: (64, 4) output layer weights
        b3: (4,) output layer bias
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray


def init_weights(input_dim: int = 24, seed: int = 42) -> ClassifierWeights:
    """Initialize network weights with Xavier initialization.

    Args:
        input_dim: Number of input features (default 24: K + 4 + 7 + 3)
        seed: Random seed for reproducibility

    Returns:
        ClassifierWeights with Xavier-initialized parameters
    """
    np.random.seed(seed)

    # Xavier init: scale by 1/sqrt(fan_in)
    w1 = np.random.randn(input_dim, 128) / np.sqrt(input_dim)
    b1 = np.zeros(128, dtype=np.float32)

    w2 = np.random.randn(128, 64) / np.sqrt(128)
    b2 = np.zeros(64, dtype=np.float32)

    w3 = np.random.randn(64, 4) / np.sqrt(64)
    b3 = np.zeros(4, dtype=np.float32)

    return ClassifierWeights(
        w1=w1.astype(np.float32),
        b1=b1,
        w2=w2.astype(np.float32),
        b2=b2,
        w3=w3.astype(np.float32),
        b3=b3,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: (4,) raw output logits

    Returns:
        (4,) probability distribution
    """
    logits = logits - np.max(logits)  # Subtract max for numerical stability
    exp = np.exp(logits)
    return exp / np.sum(exp)


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0.0)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU gradient mask (1 where x > 0, else 0)."""
    return (x > 0.0).astype(np.float32)


def forward(x: np.ndarray, weights: ClassifierWeights) -> np.ndarray:
    """Forward pass through the network.

    Args:
        x: (input_dim,) feature vector
        weights: ClassifierWeights

    Returns:
        (4,) softmax probability distribution over answer choices
    """
    # Layer 1: input -> 128 with ReLU
    h1 = _relu(x @ weights.w1 + weights.b1)

    # Layer 2: 128 -> 64 with ReLU
    h2 = _relu(h1 @ weights.w2 + weights.b2)

    # Output layer: 64 -> 4 with softmax
    logits = h2 @ weights.w3 + weights.b3
    probs = _softmax(logits)

    return probs


def _forward_with_cache(
    x: np.ndarray, weights: ClassifierWeights
) -> tuple[np.ndarray, dict]:
    """Forward pass that caches intermediate values for backprop.

    Args:
        x: (input_dim,) feature vector
        weights: ClassifierWeights

    Returns:
        Tuple of (probs, cache) where cache contains intermediate values
    """
    z1 = x @ weights.w1 + weights.b1
    h1 = _relu(z1)

    z2 = h1 @ weights.w2 + weights.b2
    h2 = _relu(z2)

    logits = h2 @ weights.w3 + weights.b3
    probs = _softmax(logits)

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "logits": logits,
        "probs": probs,
    }

    return probs, cache


def classify(
    settlement: np.ndarray,
    choice_sims: np.ndarray,
    scm_layers: np.ndarray,
    weights: ClassifierWeights,
    coevo_layers: np.ndarray | None = None,
) -> tuple[int, float]:
    """Classify a question to one of 4 answer choices.

    Args:
        settlement: (K,) settled opinion values from cortex L2
        choice_sims: (4,) Pearson similarities of each choice to query attractor
        scm_layers: (7,) SCMResult fields in order [temperature, salience,
            energy, integration, polarity, net_warrant, explanation_readiness].
            See cortex_scm.py:SCMResult. Note: these labels are first-initial
            abbreviations of SCMResult fields and intentionally distinct from
            the canonical SCM v2.0 variable namespace (T, E, S, I, P, NW, ERF,
            FC) — letters collide, concepts do not. See CANON §3.5.2.
        weights: ClassifierWeights
        coevo_layers: (3,) optional coevolution layer features

    Returns:
        Tuple of (predicted_choice_index, confidence)
        - predicted_choice_index: int in [0, 1, 2, 3]
        - confidence: float in [0, 1], the max softmax probability
    """
    # Concatenate features: K + 4 + 7 = K + 11, optionally + 3 = K + 14
    x = np.concatenate([settlement, choice_sims, scm_layers]).astype(np.float32)
    if coevo_layers is not None:
        x = np.concatenate([x, coevo_layers]).astype(np.float32)

    # Forward pass
    probs = forward(x, weights)

    # Predict argmax and get confidence
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return idx, confidence


def cross_entropy_loss(probs: np.ndarray, target: int) -> float:
    """Cross-entropy loss for single example.

    Args:
        probs: (4,) softmax probabilities
        target: int in [0, 1, 2, 3], the correct answer index

    Returns:
        float, the cross-entropy loss (higher is worse)
    """
    # Clip to avoid log(0)
    epsilon = 1e-10
    probs_clipped = np.clip(probs, epsilon, 1.0)

    loss = -np.log(probs_clipped[target])
    return float(loss)


def backward(
    x: np.ndarray,
    weights: ClassifierWeights,
    target: int,
    learning_rate: float = 0.001,
) -> tuple[ClassifierWeights, float]:
    """Single-example SGD step with full manual backprop.

    Args:
        x: (input_dim,) feature vector
        weights: ClassifierWeights
        target: int in [0, 1, 2, 3], correct answer index
        learning_rate: float, step size for SGD

    Returns:
        Tuple of (updated_weights, loss)
    """
    # Forward with cache
    probs, cache = _forward_with_cache(x, weights)

    # Loss
    loss = cross_entropy_loss(probs, target)

    # --- Backprop ---
    # Output layer gradient
    # d(loss)/d(logits) = probs - one_hot(target)
    dlogits = probs.copy()
    dlogits[target] -= 1.0

    # Gradient w.r.t. w3 and b3
    h2 = cache["h2"]
    dw3 = np.outer(h2, dlogits)  # (64, 1) @ (1, 4) = (64, 4)
    db3 = dlogits.copy()  # (4,)

    # Gradient w.r.t. h2
    dh2 = dlogits @ weights.w3.T  # (4,) @ (4, 64) = (64,)

    # ReLU gate at z2
    dz2 = dh2 * _relu_derivative(cache["z2"])

    # Gradient w.r.t. w2 and b2
    h1 = cache["h1"]
    dw2 = np.outer(h1, dz2)  # (128, 1) @ (1, 64) = (128, 64)
    db2 = dz2.copy()  # (64,)

    # Gradient w.r.t. h1
    dh1 = dz2 @ weights.w2.T  # (64,) @ (64, 128) = (128,)

    # ReLU gate at z1
    dz1 = dh1 * _relu_derivative(cache["z1"])

    # Gradient w.r.t. w1 and b1
    dw1 = np.outer(x, dz1)  # (input_dim, 1) @ (1, 128) = (input_dim, 128)
    db1 = dz1.copy()  # (128,)

    # --- Update weights ---
    new_weights = ClassifierWeights(
        w1=(weights.w1 - learning_rate * dw1).astype(np.float32),
        b1=(weights.b1 - learning_rate * db1).astype(np.float32),
        w2=(weights.w2 - learning_rate * dw2).astype(np.float32),
        b2=(weights.b2 - learning_rate * db2).astype(np.float32),
        w3=(weights.w3 - learning_rate * dw3).astype(np.float32),
        b3=(weights.b3 - learning_rate * db3).astype(np.float32),
    )

    return new_weights, loss


def save_weights(weights: ClassifierWeights, path: Path) -> None:
    """Save classifier weights to .npz file.

    Args:
        weights: ClassifierWeights
        path: Path to save to (should end in .npz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        w1=weights.w1,
        b1=weights.b1,
        w2=weights.w2,
        b2=weights.b2,
        w3=weights.w3,
        b3=weights.b3,
    )


def load_weights(path: Path) -> ClassifierWeights:
    """Load classifier weights from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        ClassifierWeights
    """
    path = Path(path)
    data = np.load(path)

    return ClassifierWeights(
        w1=data["w1"],
        b1=data["b1"],
        w2=data["w2"],
        b2=data["b2"],
        w3=data["w3"],
        b3=data["b3"],
    )

"""Tests for wheeler_memory.cortex_classifier — L3 neural network classifier."""

import numpy as np
import pytest

from wheeler_memory.cortex_classifier import (
    ClassifierWeights,
    backward,
    classify,
    cross_entropy_loss,
    forward,
    init_weights,
    load_weights,
    save_weights,
)


class TestInitWeights:
    """init_weights() Xavier initialization."""

    def test_shapes(self):
        """Default dims: (24,128), (128,), (128,64), (64,), (64,4), (4,)."""
        w = init_weights()
        assert w.w1.shape == (24, 128)
        assert w.b1.shape == (128,)
        assert w.w2.shape == (128, 64)
        assert w.b2.shape == (64,)
        assert w.w3.shape == (64, 4)
        assert w.b3.shape == (4,)

    def test_custom_input_dim(self):
        """Non-default input_dim changes w1 shape."""
        w = init_weights(input_dim=10)
        assert w.w1.shape == (10, 128)

    def test_dtypes_float32(self):
        """All weight arrays are float32."""
        w = init_weights()
        for arr in [w.w1, w.b1, w.w2, w.b2, w.w3, w.b3]:
            assert arr.dtype == np.float32

    def test_biases_zero_initialized(self):
        """Biases start at zero."""
        w = init_weights()
        np.testing.assert_array_equal(w.b1, 0.0)
        np.testing.assert_array_equal(w.b2, 0.0)
        np.testing.assert_array_equal(w.b3, 0.0)

    def test_deterministic_same_seed(self):
        """Same seed → identical weights."""
        w1 = init_weights(seed=123)
        w2 = init_weights(seed=123)
        np.testing.assert_array_equal(w1.w1, w2.w1)

    def test_different_seeds_differ(self):
        """Different seeds → different weights."""
        w1 = init_weights(seed=1)
        w2 = init_weights(seed=2)
        assert not np.array_equal(w1.w1, w2.w1)


class TestForward:
    """forward() network inference."""

    def test_output_shape(self):
        """Output is (4,) probability vector."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        probs = forward(x, w)
        assert probs.shape == (4,)

    def test_output_sums_to_one(self):
        """Softmax output sums to 1."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        probs = forward(x, w)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_output_nonnegative(self):
        """All probabilities >= 0."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        probs = forward(x, w)
        assert np.all(probs >= 0.0)

    def test_deterministic(self):
        """Same input → same output."""
        w = init_weights()
        x = np.ones(24, dtype=np.float32)
        assert np.array_equal(forward(x, w), forward(x, w))


class TestClassify:
    """classify() end-to-end classification."""

    def test_returns_index_and_confidence(self):
        """Returns (int, float) tuple."""
        w = init_weights()
        settlement = np.random.default_rng(1).standard_normal(13).astype(np.float32)
        choice_sims = np.array([0.3, 0.5, 0.1, 0.8], dtype=np.float32)
        scm_layers = np.random.default_rng(2).standard_normal(7).astype(np.float32)
        idx, conf = classify(settlement, choice_sims, scm_layers, w)
        assert isinstance(idx, int)
        assert 0 <= idx <= 3
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_with_coevo_layers(self):
        """Optional coevolution layers expand feature vector."""
        w = init_weights(input_dim=27)  # 13 + 4 + 7 + 3
        settlement = np.zeros(13, dtype=np.float32)
        choice_sims = np.zeros(4, dtype=np.float32)
        scm_layers = np.zeros(7, dtype=np.float32)
        coevo = np.array([0.5, 0.3, 0.7], dtype=np.float32)
        idx, conf = classify(settlement, choice_sims, scm_layers, w, coevo_layers=coevo)
        assert 0 <= idx <= 3


class TestCrossEntropyLoss:
    """cross_entropy_loss() computation."""

    def test_perfect_prediction_near_zero(self):
        """Confident correct prediction → near-zero loss."""
        probs = np.array([0.97, 0.01, 0.01, 0.01])
        loss = cross_entropy_loss(probs, target=0)
        assert loss < 0.1

    def test_wrong_prediction_high_loss(self):
        """Confident wrong prediction → high loss."""
        probs = np.array([0.01, 0.01, 0.97, 0.01])
        loss = cross_entropy_loss(probs, target=0)
        assert loss > 3.0

    def test_uniform_distribution(self):
        """Uniform probs → -log(0.25) ≈ 1.386."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        loss = cross_entropy_loss(probs, target=0)
        assert loss == pytest.approx(-np.log(0.25), abs=1e-5)

    def test_nonnegative(self):
        """Loss is always >= 0."""
        probs = np.array([0.5, 0.2, 0.2, 0.1])
        for t in range(4):
            assert cross_entropy_loss(probs, target=t) >= 0.0


class TestBackward:
    """backward() SGD update."""

    def test_returns_updated_weights_and_loss(self):
        """Returns (ClassifierWeights, float) pair."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        new_w, loss = backward(x, w, target=2)
        assert isinstance(new_w, ClassifierWeights)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_weights_change_after_step(self):
        """At least some weights differ after one SGD step."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        new_w, _ = backward(x, w, target=1)
        assert not np.array_equal(w.w1, new_w.w1)

    def test_loss_decreases_on_repeated_steps(self):
        """Multiple SGD steps on same example → loss decreases."""
        w = init_weights()
        x = np.random.default_rng(42).standard_normal(24).astype(np.float32)
        _, loss_before = backward(x, w, target=0)
        for _ in range(20):
            w, _ = backward(x, w, target=0, learning_rate=0.01)
        _, loss_after = backward(x, w, target=0)
        assert loss_after < loss_before


class TestSaveLoadWeights:
    """save_weights / load_weights round-trip."""

    def test_roundtrip(self, tmp_path):
        """Save then load recovers identical weights."""
        w = init_weights()
        path = tmp_path / "weights.npz"
        save_weights(w, path)
        loaded = load_weights(path)
        np.testing.assert_array_equal(w.w1, loaded.w1)
        np.testing.assert_array_equal(w.b1, loaded.b1)
        np.testing.assert_array_equal(w.w2, loaded.w2)
        np.testing.assert_array_equal(w.b2, loaded.b2)
        np.testing.assert_array_equal(w.w3, loaded.w3)
        np.testing.assert_array_equal(w.b3, loaded.b3)

    def test_creates_parent_dirs(self, tmp_path):
        """save_weights creates parent directories."""
        path = tmp_path / "deep" / "nested" / "weights.npz"
        w = init_weights()
        save_weights(w, path)
        assert path.exists()

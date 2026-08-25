import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_network import NeuralNetwork


class TestNeuralNetwork:
    """Basic tests for the NeuralNetwork class."""

    def test_initialization(self):
        nn = NeuralNetwork([784, 128, 64, 10])
        assert nn.layer_sizes == [784, 128, 64, 10]
        assert len(nn.weights) == 3
        assert len(nn.biases) == 3

    def test_weight_shapes(self):
        nn = NeuralNetwork([784, 128, 64, 10])
        assert nn.weights[0].shape == (784, 128)
        assert nn.weights[1].shape == (128, 64)
        assert nn.weights[2].shape == (64, 10)

    def test_forward_pass_shape(self):
        nn = NeuralNetwork([784, 128, 64, 10])
        X = np.random.randn(5, 784)
        output = nn.forward(X)
        assert output.shape == (5, 10)

    def test_predict_returns_classes(self):
        nn = NeuralNetwork([784, 128, 64, 10])
        X = np.random.randn(10, 784)
        preds = nn.predict(X)
        assert preds.shape == (10,)
        assert all(0 <= p <= 9 for p in preds)

    def test_predict_proba_sums_to_one(self):
        nn = NeuralNetwork([784, 128, 64, 10])
        X = np.random.randn(3, 784)
        proba = nn.predict_proba(X)
        assert proba.shape == (3, 10)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_softmax_stability(self):
        nn = NeuralNetwork([784, 10], activation="relu")
        X = np.random.randn(1, 784) * 1000
        output = nn.forward(X)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        np.testing.assert_allclose(output.sum(), 1.0, atol=1e-6)

    def test_training_reduces_loss(self):
        np.random.seed(42)
        nn = NeuralNetwork([10, 32, 3], learning_rate=0.1, l2_lambda=0.0)
        X = np.random.randn(100, 10)
        y = np.array([0] * 33 + [1] * 34 + [2] * 33)
        history = nn.train(X, y, X, y, epochs=20, batch_size=32)
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_save_and_load(self, tmp_path):
        nn = NeuralNetwork([784, 32, 10])
        X = np.random.randn(5, 784)
        preds_before = nn.predict(X)

        path = str(tmp_path / "test_model.npy")
        nn.save_weights(path)
        nn2 = NeuralNetwork.load_weights(path)
        preds_after = nn2.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_all_activations(self):
        for act in ["relu", "sigmoid", "tanh", "leaky_relu"]:
            nn = NeuralNetwork([784, 32, 10], activation=act)
            X = np.random.randn(5, 784)
            output = nn.forward(X)
            assert output.shape == (5, 10)
            assert not np.any(np.isnan(output))

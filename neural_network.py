import numpy as np


class NeuralNetwork:
    """
    A feedforward neural network for multi-class classification.
    Built from scratch using only NumPy for matrix operations.

    Architecture:
        Input (784) -> Hidden1 (128, ReLU) -> Hidden2 (64, ReLU) -> Output (10, Softmax)

    Training uses cross-entropy loss with gradient descent optimization.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, lr_decay=0.95, lr_min=0.001,
                 l2_lambda=0.001, activation='relu'):
        """
        Initialize the neural network with random weights and zero biases.

        Args:
            layer_sizes: list of ints, number of neurons in each layer
                         e.g. [784, 128, 64, 10] for MNIST classification
            learning_rate: float, step size for gradient descent updates
            lr_decay: float, multiply learning rate by this after each epoch
            lr_min: float, minimum learning rate (don't decay below this)
            l2_lambda: float, L2 regularization strength (0 to disable)
            activation: str, activation function for hidden layers
                       'relu', 'sigmoid', 'tanh', or 'leaky_relu'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.current_lr = learning_rate
        self.l2_lambda = l2_lambda
        self.activation_name = activation

        # Set activation functions based on name
        self.activation_fn, self.activation_deriv = self._get_activation(activation)

        # Initialize weights using He initialization for ReLU layers
        # This scales weights by sqrt(2/n_in) to keep variance stable across layers
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Use different initialization based on activation
            if activation == 'relu' or activation == 'leaky_relu':
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            elif activation == 'sigmoid' or activation == 'tanh':
                # Xavier initialization works better for sigmoid/tanh
                w = np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
            else:
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        # Storage for cached values during forward pass (used in backprop)
        self.z_cache = []  # pre-activation values
        self.a_cache = []  # post-activation values

    def _get_activation(self, name):
        """Get activation function and its derivative by name."""
        activations = {
            'relu': (self.relu, self.relu_derivative),
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative)
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from: {list(activations.keys())}")
        return activations[name]

    def relu(self, z):
        """ReLU activation: max(0, z). Introduces non-linearity."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0."""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid activation: 1 / (1 + e^(-z)). Maps to (0, 1)."""
        # Clip z to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))."""
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z):
        """Tanh activation: (e^z - e^(-z)) / (e^z + e^(-z)). Maps to (-1, 1)."""
        return np.tanh(z)

    def tanh_derivative(self, z):
        """Derivative of tanh: 1 - tanh(z)^2."""
        return 1 - np.tanh(z) ** 2

    def leaky_relu(self, z, alpha=0.01):
        """Leaky ReLU: z if z > 0, else alpha * z. Prevents dead neurons."""
        return np.where(z > 0, z, alpha * z)

    def leaky_relu_derivative(self, z, alpha=0.01):
        """Derivative of Leaky ReLU: 1 if z > 0, else alpha."""
        return np.where(z > 0, 1.0, alpha)

    def softmax(self, z):
        """
        Softmax activation for output layer.
        Converts raw logits to probabilities that sum to 1.
        Uses numerical stability trick by subtracting the max value.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation: pass input through the network layer by layer.

        For each hidden layer: z = X@W + b, then a = relu(z)
        For output layer: z = X@W + b, then a = softmax(z)

        Args:
            X: input data, shape (n_samples, n_features)

        Returns:
            output probabilities, shape (n_samples, n_classes)
        """
        self.z_cache = []
        self.a_cache = [X]

        current_input = X

        for i in range(self.num_layers - 1):
            # Linear transformation: z = input @ weight + bias
            z = current_input @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)

            # Apply activation function
            if i < self.num_layers - 2:
                # Hidden layers use selected activation
                a = self.activation_fn(z)
            else:
                # Output layer uses Softmax
                a = self.softmax(z)

            self.a_cache.append(a)
            current_input = a

        return current_input

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Cross-entropy loss with optional L2 regularization.
        L2 adds a penalty for large weights to prevent overfitting.

        Args:
            y_pred: predicted probabilities (n_samples, n_classes)
            y_true: one-hot encoded true labels (n_samples, n_classes)

        Returns:
            scalar loss value
        """
        n_samples = y_true.shape[0]
        # Clip predictions to avoid log(0) which gives -inf
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples

        # Add L2 regularization term: lambda/2 * sum(w^2)
        if self.l2_lambda > 0:
            l2_loss = 0
            for w in self.weights:
                l2_loss += np.sum(w ** 2)
            loss += (self.l2_lambda / 2) * l2_loss / n_samples

        return loss

    def one_hot_encode(self, y, num_classes):
        """Convert integer labels to one-hot vectors."""
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def backward(self, y_true):
        """
        Backpropagation: compute gradients of loss w.r.t. all weights and biases.

        Uses the chain rule to propagate error backwards from output to input.
        For each layer, we compute:
            dZ = error * activation_derivative
            dW = A_prev.T @ dZ / n_samples
            db = sum(dZ, axis=0, keepdims=True) / n_samples

        Args:
            y_true: one-hot encoded true labels (n_samples, n_classes)
        """
        n_samples = y_true.shape[0]
        num_classes = self.layer_sizes[-1]

        self.d_weights = []
        self.d_biases = []

        # Output layer error: dZ = y_pred - y_true
        # This gradient comes from combining softmax + cross-entropy derivatives
        y_pred = self.a_cache[-1]
        dz = (y_pred - y_true) / n_samples

        # Compute gradients for each layer going backwards
        for i in range(self.num_layers - 2, -1, -1):
            a_prev = self.a_cache[i]

            # Gradient for weights: dW = A_prev.T @ dZ
            dw = a_prev.T @ dz
            # Gradient for biases: db = sum(dZ, axis=0)
            db = np.sum(dz, axis=0, keepdims=True)

            # Add L2 regularization gradient: lambda * w
            if self.l2_lambda > 0:
                dw += self.l2_lambda * self.weights[i]

            self.d_weights.insert(0, dw)
            self.d_biases.insert(0, db)

            # Propagate error to previous layer (if not at input layer)
            if i > 0:
                dz = dz @ self.weights[i].T * self.activation_deriv(self.z_cache[i - 1])

        # Update weights and biases using gradient descent
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def compute_accuracy(self, X, y):
        """
        Compute classification accuracy.

        Args:
            X: input features (n_samples, n_features)
            y: integer labels (n_samples,)

        Returns:
            accuracy as a float between 0 and 1
        """
        predictions = self.forward(X)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == y)
        return accuracy

    def predict(self, X):
        """
        Predict class labels for input data.

        Args:
            X: input features (n_samples, n_features)

        Returns:
            predicted class labels (n_samples,)
        """
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        Args:
            X: input features (n_samples, n_features)

        Returns:
            class probabilities (n_samples, n_classes)
        """
        return self.forward(X)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128,
              early_stopping_patience=10, lr_schedule=True):
        """
        Train the neural network using mini-batch gradient descent.

        Args:
            X_train: training features (n_samples, n_features)
            y_train: training labels (n_samples,)
            X_val: validation features
            y_val: validation labels
            epochs: number of complete passes through training data
            batch_size: number of samples per mini-batch
            early_stopping_patience: stop if val loss doesn't improve for this many epochs
            lr_schedule: whether to decay learning rate over time

        Returns:
            dict with training history (losses, accuracies)
        """
        n_samples = X_train.shape[0]
        y_train_onehot = self.one_hot_encode(y_train, self.layer_sizes[-1])

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # Shuffle training data each epoch for better convergence
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                self.forward(X_batch)
                # Backward pass (computes gradients and updates weights)
                self.backward(y_batch)

            # Record metrics at end of each epoch
            train_loss = self.cross_entropy_loss(
                self.forward(X_train), y_train_onehot
            )
            val_loss = self.cross_entropy_loss(
                self.forward(X_val), self.one_hot_encode(y_val, self.layer_sizes[-1])
            )
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(self.current_lr)

            # Learning rate scheduling
            if lr_schedule:
                self.current_lr = max(self.current_lr * self.lr_decay, self.lr_min)
                self.learning_rate = self.current_lr

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"LR: {self.current_lr:.6f}"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                # Restore best weights
                self.weights = best_weights
                self.biases = best_biases
                break

        return history

    def save_weights(self, filepath):
        """Save all weights and biases to a .npy file."""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'lr_min': self.lr_min,
            'l2_lambda': self.l2_lambda,
            'activation': self.activation_name
        }
        np.save(filepath, data, allow_pickle=True)
        print(f"Model weights saved to {filepath}")

    @classmethod
    def load_weights(cls, filepath):
        """Load a saved model from a .npy file."""
        data = np.load(filepath, allow_pickle=True).item()
        model = cls(
            layer_sizes=data['layer_sizes'],
            learning_rate=data['learning_rate'],
            lr_decay=data.get('lr_decay', 0.95),
            lr_min=data.get('lr_min', 0.001),
            l2_lambda=data.get('l2_lambda', 0.001),
            activation=data.get('activation', 'relu')
        )
        model.weights = data['weights']
        model.biases = data['biases']
        print(f"Model weights loaded from {filepath}")
        return model

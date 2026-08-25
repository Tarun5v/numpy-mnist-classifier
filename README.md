# NumPy MNIST Classifier

A neural network built from scratch that classifies handwritten digits (0-9) from the MNIST dataset using only Python and NumPy. No deep learning frameworks, no AI libraries — just raw matrix math.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~96%+ |
| Architecture | 784 → 128 → 64 → 10 |
| Training Epochs | 30 |
| Framework | NumPy only |

## How It Works

### Network Architecture

```
Input Layer (784 neurons)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
```

Each MNIST image is 28×28 pixels = 784 features. The network learns to map these pixel values to one of 10 digit classes.

### Forward Propagation

For each layer, we compute:

```
z = X · W + b
a = activation(z)
```

Where:
- `X` = input to the layer
- `W` = weight matrix (learned parameters)
- `b` = bias vector (learned parameters)
- `z` = pre-activation value
- `a` = post-activation value

**Hidden layers** use ReLU: `f(z) = max(0, z)`
**Output layer** uses Softmax: `f(z_i) = e^(z_i) / Σ e^(z_j)`

Softmax converts raw scores (logits) into probabilities that sum to 1.

### Loss Function: Cross-Entropy

```
L = -1/N * Σ Σ y_true * log(y_pred)
```

This measures how far our predicted probabilities are from the true labels. When the model assigns high probability to the correct class, the loss is low.

### Backpropagation

To train the network, we need to compute how much each weight contributes to the error. We use the chain rule to propagate gradients backward through the network.

**Output layer gradient** (softmax + cross-entropy simplifies nicely):
```
dZ = (y_pred - y_true) / N
```

**Hidden layer gradients:**
```
dW = A_prev^T · dZ / N
db = Σ dZ / N
dZ_prev = dZ · W^T ⊙ ReLU'(z)
```

Where `⊙` is element-wise multiplication and `ReLU'(z) = 1 if z > 0, else 0`.

### Weight Updates (Gradient Descent)

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

The learning rate (0.1) controls how big each update step is.

### Weight Initialization

Weights are initialized using He initialization: `W ~ N(0, sqrt(2/n_in))`

This keeps the variance of activations stable across layers, which helps with ReLU networks.

## Project Structure

```
numpy-mnist-classifier/
├── neural_network.py    # NeuralNetwork class (forward, backward, train, predict)
├── main.py              # Training script with data loading and visualization
├── predict.py           # Load saved model and make predictions
├── requirements.txt     # Dependencies (numpy, matplotlib)
├── models/              # Saved model weights (.npy files)
├── results/             # Generated plots and visualizations
├── data/                # MNIST dataset (downloaded automatically)
└── README.md
```

## Usage

### 1. Setup

```bash
cd numpy-mnist-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This will:
- Download the MNIST dataset automatically
- Train the neural network for 30 epochs
- Save model weights to `models/mnist_model.npy`
- Generate training plots and confusion matrix in `results/`

### 3. Make Predictions

```bash
python predict.py
```

This loads the saved model and evaluates it on test images.

### 4. Use in Your Code

```python
from neural_network import NeuralNetwork

# Load trained model
model = NeuralNetwork.load_weights('models/mnist_model.npy')

# Predict on new data
predictions = model.predict(X_new)  # Returns class labels (0-9)
probabilities = model.predict_proba(X_new)  # Returns probability distribution
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for visualization only)

No GPU required. Trains in about 2-3 minutes on a CPU.

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip
from urllib.request import urlretrieve
from neural_network import NeuralNetwork


def download_mnist(data_dir='data'):
    """
    Download MNIST dataset files if not already present.
    MNIST consists of 60k training and 10k test images (28x28 grayscale).
    """
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    os.makedirs(data_dir, exist_ok=True)

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath.replace('.gz', '')):
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filepath)

            # Decompress the gzipped file
            with gzip.open(filepath, 'rb') as f_in:
                with open(filepath.replace('.gz', ''), 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(filepath)
            print(f"  Extracted {filename}")


def load_mnist_images(filename):
    """Load MNIST images from the IDX file format."""
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read image data and reshape to (n, 784)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        # Normalize pixel values to [0, 1]
        images = images.astype(np.float64) / 255.0
    return images


def load_mnist_labels(filename):
    """Load MNIST labels from the IDX file format."""
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir='data'):
    """Load the full MNIST dataset."""
    print("Loading MNIST dataset...")
    X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Image size:   {X_train.shape[1]} pixels (28x28)")
    return X_train, y_train, X_test, y_test


def split_validation(X_train, y_train, val_ratio=0.1):
    """Split training data into train and validation sets."""
    n = X_train.shape[0]
    n_val = int(n * val_ratio)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx]


def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training and validation loss/accuracy over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, num_classes=10, save_path='results/confusion_matrix.png'):
    """Plot a confusion matrix showing classification results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    return cm


def plot_sample_predictions(X, y_true, y_pred, num_samples=16, save_path='results/sample_predictions.png'):
    """Visualize sample predictions with correct/incorrect labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    indices = np.random.choice(len(X), num_samples, replace=False)

    for idx, ax in zip(indices, axes.flat):
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(
            f"True: {y_true[idx]}, Pred: {y_pred[idx]}",
            color=color, fontsize=10
        )
        ax.axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def main():
    np.random.seed(42)

    # Download and load MNIST data
    download_mnist()
    X_train_full, y_train_full, X_test, y_test = load_mnist()

    # Split into train and validation
    X_train, y_train, X_val, y_val = split_validation(X_train_full, y_train_full)

    print(f"\nDataset split:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")

    # Initialize neural network
    # Architecture: 784 input -> 128 hidden -> 64 hidden -> 10 output
    model = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=0.1
    )

    print(f"\nNetwork Architecture:")
    print(f"  Layers: {model.layer_sizes}")
    print(f"  Learning Rate: {model.learning_rate}")
    total_params = sum(
        w.size + b.size for w, b in zip(model.weights, model.biases)
    )
    print(f"  Total Parameters: {total_params:,}")

    # Train the model
    print(f"\nStarting training...")
    print("-" * 80)
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=128
    )
    print("-" * 80)

    # Evaluate on test set
    test_acc = model.compute_accuracy(X_test, y_test)
    test_loss = model.cross_entropy_loss(
        model.forward(X_test),
        model.one_hot_encode(y_test, 10)
    )
    print(f"\nTest Set Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Save model weights
    os.makedirs('models', exist_ok=True)
    model.save_weights('models/mnist_model.npy')

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(X_test, y_test, y_pred)

    print("\nAll done!")


if __name__ == '__main__':
    main()

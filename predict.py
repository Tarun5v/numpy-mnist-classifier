import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip
from neural_network import NeuralNetwork


def load_test_data(data_dir='data'):
    """Load MNIST test data for prediction demo."""
    images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

    with open(images_file, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        images = images.astype(np.float64) / 255.0

    with open(labels_file, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def visualize_predictions(model, X, y, num_samples=25, save_path='results/predictions.png'):
    """
    Visualize model predictions on random test images.
    Shows the image, true label, predicted label, and confidence.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    indices = np.random.choice(len(X), num_samples, replace=False)
    images = X[indices]
    true_labels = y[indices]

    # Get predictions and probabilities
    probs = model.predict_proba(images)
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Plot results
    cols = 5
    rows = num_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        correct = true_labels[i] == pred_labels[i]
        color = 'green' if correct else 'red'
        ax.set_title(
            f"True: {true_labels[i]}\n"
            f"Pred: {pred_labels[i]} ({confidences[i]:.1%})",
            color=color, fontsize=9
        )
        ax.axis('off')

    plt.suptitle('MNIST Digit Classification Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction visualization saved to {save_path}")


def predict_single_digit(model, image_vector):
    """
    Predict a single digit and print the result with probabilities.

    Args:
        model: trained NeuralNetwork instance
        image_vector: flattened 784-element array representing a 28x28 image
    """
    image_vector = image_vector.reshape(1, -1)
    probs = model.predict_proba(image_vector)[0]
    predicted = np.argmax(probs)

    print(f"\nPredicted digit: {predicted}")
    print(f"Confidence: {probs[predicted]:.2%}")
    print(f"\nAll class probabilities:")
    for digit in range(10):
        bar = "#" * int(probs[digit] * 40)
        print(f"  {digit}: {probs[digit]:.4f} {bar}")


def main():
    # Load saved model
    model_path = 'models/mnist_model.npy'
    if not os.path.exists(model_path):
        print(f"Error: No saved model found at {model_path}")
        print("Please run main.py first to train and save the model.")
        return

    model = NeuralNetwork.load_weights(model_path)

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"Loaded {len(X_test)} test images")

    # Evaluate accuracy
    accuracy = model.compute_accuracy(X_test, y_test)
    print(f"\nModel accuracy on test set: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Visualize predictions on random samples
    visualize_predictions(model, X_test, y_test)

    # Demo: predict a single random image
    idx = np.random.randint(len(X_test))
    print(f"\n--- Single Image Prediction (index {idx}) ---")
    predict_single_digit(model, X_test[idx])
    print(f"True label: {y_test[idx]}")


if __name__ == '__main__':
    main()

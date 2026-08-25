"""
Benchmark script to compare different neural network configurations.
Tests various activation functions, learning rates, and architectures
to find the best settings for MNIST classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from neural_network import NeuralNetwork
from main import load_mnist, split_validation


def run_benchmark(X_train, y_train, X_val, y_val, config, epochs=20):
    """
    Run a single benchmark with given configuration.
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        config: dict with model parameters
        epochs: number of training epochs
        
    Returns:
        dict with results
    """
    print(f"\nTesting: {config['name']}")
    print("-" * 50)
    
    # Create model with config parameters
    model = NeuralNetwork(
        layer_sizes=config['layer_sizes'],
        learning_rate=config['learning_rate'],
        lr_decay=config.get('lr_decay', 0.95),
        l2_lambda=config.get('l2_lambda', 0.0),
        activation=config.get('activation', 'relu')
    )
    
    # Train and time it
    start_time = time.time()
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=config.get('batch_size', 128),
        lr_schedule=config.get('lr_schedule', True),
        early_stopping_patience=config.get('early_stopping_patience', 10)
    )
    training_time = time.time() - start_time
    
    # Get final metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"\nResults for {config['name']}:")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Val Accuracy:   {final_val_acc:.4f}")
    print(f"  Train Loss:     {final_train_loss:.4f}")
    print(f"  Val Loss:       {final_val_loss:.4f}")
    print(f"  Training Time:  {training_time:.2f}s")
    
    return {
        'name': config['name'],
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'time': training_time,
        'history': history
    }


def plot_benchmark_results(results, save_path='results/benchmark_comparison.png'):
    """Plot comparison of different benchmark runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    
    # Accuracy comparison
    train_accs = [r['train_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_accs, width, label='Train', color='steelblue')
    axes[0, 0].bar(x + width/2, val_accs, width, label='Validation', color='coral')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Loss comparison
    train_losses = [r['train_loss'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    
    axes[0, 1].bar(x - width/2, train_losses, width, label='Train', color='steelblue')
    axes[0, 1].bar(x + width/2, val_losses, width, label='Validation', color='coral')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Training time comparison
    times = [r['time'] for r in results]
    axes[1, 0].bar(x, times, color='steelblue')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Training curves (val accuracy over epochs)
    for r in results:
        axes[1, 1].plot(r['history']['val_acc'], label=r['name'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Validation Accuracy Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBenchmark plot saved to {save_path}")


def main():
    np.random.seed(42)
    
    # Load data
    print("Loading MNIST dataset...")
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    X_train, y_train, X_val, y_val = split_validation(X_train_full, y_train_full)
    
    # Define configurations to test
    configs = [
        {
            'name': 'Baseline (ReLU)',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.1,
            'activation': 'relu',
            'l2_lambda': 0.0,
            'lr_schedule': False
        },
        {
            'name': 'ReLU + LR Schedule',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.1,
            'activation': 'relu',
            'l2_lambda': 0.0,
            'lr_schedule': True,
            'lr_decay': 0.95
        },
        {
            'name': 'ReLU + L2 Reg',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.1,
            'activation': 'relu',
            'l2_lambda': 0.001,
            'lr_schedule': False
        },
        {
            'name': 'Sigmoid',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.5,
            'activation': 'sigmoid',
            'l2_lambda': 0.0,
            'lr_schedule': True,
            'lr_decay': 0.95
        },
        {
            'name': 'Tanh',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.3,
            'activation': 'tanh',
            'l2_lambda': 0.0,
            'lr_schedule': True,
            'lr_decay': 0.95
        },
        {
            'name': 'Leaky ReLU',
            'layer_sizes': [784, 128, 64, 10],
            'learning_rate': 0.1,
            'activation': 'leaky_relu',
            'l2_lambda': 0.0,
            'lr_schedule': True,
            'lr_decay': 0.95
        },
        {
            'name': 'Deeper Network',
            'layer_sizes': [784, 256, 128, 64, 10],
            'learning_rate': 0.1,
            'activation': 'relu',
            'l2_lambda': 0.001,
            'lr_schedule': True,
            'lr_decay': 0.95
        },
        {
            'name': 'Wider Network',
            'layer_sizes': [784, 256, 256, 10],
            'learning_rate': 0.1,
            'activation': 'relu',
            'l2_lambda': 0.001,
            'lr_schedule': True,
            'lr_decay': 0.95
        }
    ]
    
    # Run benchmarks
    print("\n" + "=" * 70)
    print("NEURAL NETWORK BENCHMARK")
    print("=" * 70)
    
    results = []
    for config in configs:
        result = run_benchmark(X_train, y_train, X_val, y_val, config, epochs=20)
        results.append(result)
    
    # Sort by validation accuracy
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (sorted by validation accuracy)")
    print("=" * 70)
    print(f"{'Name':<25} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss':<12} {'Time':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['train_acc']:<12.4f} {r['val_acc']:<12.4f} {r['val_loss']:<12.4f} {r['time']:<10.2f}")
    
    # Plot results
    plot_benchmark_results(results)
    
    print("\nBest configuration:", results[0]['name'])
    print(f"Validation Accuracy: {results[0]['val_acc']:.4f}")


if __name__ == '__main__':
    main()

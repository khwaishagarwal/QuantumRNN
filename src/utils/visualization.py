import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.visualization import plot_circuit_layout, plot_histogram

def plot_performance_metrics(metrics, title="Model Performance Metrics"):
    plt.figure(figsize=(10, 5))
    for label, values in metrics.items():
        plt.plot(values, label=label)
    
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()
    plt.show()

def compare_model_outputs(classical_outputs, quantum_outputs, title="Model Outputs Comparison"):
    plt.figure(figsize=(10, 5))
    plt.plot(classical_outputs, label='Classical Model Outputs', color='blue')
    plt.plot(quantum_outputs, label='Quantum Model Outputs', color='orange')
    
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid()
    plt.show()

def plot_training_history(train_losses, val_losses, title='Training History'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

def plot_predictions(actual, predicted, title='Predictions vs Actual'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

def visualize_quantum_circuit(qc, backend=None):
    """Visualize a quantum circuit"""
    if backend:
        transpiled_qc = transpile(qc, backend)
        return transpiled_qc.draw(output='mpl')
    else:
        return qc.draw(output='mpl')
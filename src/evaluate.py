import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time
from tqdm import tqdm

# Fix imports to use relative paths
from src.models.classical_rnn import SimpleRNN
from src.models.quantum_rnn import QRNN
from src.data.airline_passengers import prepare_dataloaders
from src.utils.visualization import plot_predictions

def evaluate_model(model_type, model_path, input_size, hidden_size, 
                  n_qubits=5, n_qlayers=1, batch_size=32, seq_length=12):
    """Evaluate either classical RNN or QRNN model"""
    
    # Prepare data
    _, test_loader, scaler = prepare_dataloaders(
        batch_size=1,  # Use batch size of 1 for step-by-step prediction
        seq_length=seq_length
    )
    
    # Initialize model
    if model_type == 'classical':
        model = SimpleRNN(input_size, hidden_size)
        model_name = f'classical_rnn_h{hidden_size}'
    elif model_type == 'quantum':
        model = QRNN(input_size, hidden_size, n_qubits, n_qlayers)
        model_name = f'quantum_rnn_h{hidden_size}_q{n_qubits}_l{n_qlayers}'
    else:
        raise ValueError("model_type must be either 'classical' or 'quantum'")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predictions and timings
    all_targets = []
    all_predictions = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f'Evaluating {model_type} model'):
            all_targets.append(targets.numpy())
            
            # Measure inference time
            start_time = time.time()
            
            if model_type == 'classical':
                # Reshape for classical RNN: [batch, seq, features]
                inputs_reshaped = inputs.squeeze(-1)
                
                outputs = []
                h = None
                
                # Process each time step in the sequence
                for t in range(seq_length):
                    output, h = model(inputs_reshaped[:, t:t+1], h)
                    outputs.append(output)
                
                # Stack outputs for the full sequence
                outputs = torch.stack(outputs, dim=1)
                
            else:  # quantum model
                outputs = []
                h = None
                
                # Process each time step in the sequence
                for t in range(seq_length):
                    output, h = model(inputs[:, t], h)
                    outputs.append(output)
                
                # Stack outputs for the full sequence
                outputs = torch.stack(outputs, dim=1)
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            all_predictions.append(outputs.numpy())
    
    # Convert lists to arrays
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Inverse transform the predictions and targets back to original scale
    all_targets_original = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    all_predictions_original = scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(all_predictions.shape)
    
    # Calculate metrics
    mse = np.mean((all_targets_original - all_predictions_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_targets_original - all_predictions_original))
    
    # Calculate mean and std of inference time
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    # Print metrics
    print(f"\n{model_type.capitalize()} Model Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Inference Time: {mean_inference_time:.6f} s")
    print(f"Std Inference Time: {std_inference_time:.6f} s")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets_original.reshape(-1), label='Actual', alpha=0.7)
    plt.plot(all_predictions_original.reshape(-1), label='Predicted', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Airline Passengers')
    plt.title(f'{model_type.capitalize()} RNN Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt_save_path = os.path.join(os.path.dirname(model_path), f'{model_name}_predictions.png')
    plt.savefig(plt_save_path)
    
    # Save results
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mean_inference_time': mean_inference_time,
        'std_inference_time': std_inference_time,
        'targets': all_targets_original,
        'predictions': all_predictions_original
    }
    
    results_save_path = os.path.join(os.path.dirname(model_path), f'{model_name}_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RNN Models')
    parser.add_argument('--model', type=str, choices=['classical', 'quantum'], 
                       required=True, help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model file')
    parser.add_argument('--input_size', type=int, default=1, 
                       help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=32, 
                       help='Hidden state size')
    parser.add_argument('--n_qubits', type=int, default=5, 
                       help='Number of qubits (for quantum model)')
    parser.add_argument('--n_qlayers', type=int, default=1, 
                       help='Number of quantum layers (for quantum model)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for evaluation')
    parser.add_argument('--seq_length', type=int, default=12, 
                       help='Sequence length for time series')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_type=args.model,
        model_path=args.model_path,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        n_qubits=args.n_qubits,
        n_qlayers=args.n_qlayers,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
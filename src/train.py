import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle
import sys

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix imports to use relative paths
from src.models.classical_rnn import SimpleRNN
from src.models.quantum_rnn import QRNN
from src.data.airline_passengers import prepare_dataloaders
from src.utils.visualization import plot_training_history

def train_model(model_type, input_size, hidden_size, n_qubits=5, n_qlayers=1, 
                batch_size=32, seq_length=12, learning_rate=0.01, 
                epochs=100, save_path='models'):
    """Train either classical RNN or QRNN model"""
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data
    train_loader, test_loader, scaler = prepare_dataloaders(
        batch_size=batch_size, 
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
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                           desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, (inputs, targets) in progress_bar:
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
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
                outputs = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
                
                # Use a linear layer to project from hidden_size to 1
                outputs_projected = outputs[:, :, 0].unsqueeze(-1)  # Just take first dimension of hidden state
                
                # Calculate loss
                loss = criterion(outputs_projected, targets)
                
            else:  # quantum model
                outputs = []
                h = None
                
                # Process each time step in the sequence
                for t in range(seq_length):
                    output, h = model(inputs[:, t], h)
                    outputs.append(output)
                
                # Stack outputs for the full sequence
                outputs = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
                
                # Use a linear layer to project from hidden_size to 1
                outputs_projected = outputs[:, :, 0].unsqueeze(-1)  # Just take first dimension of hidden state
                
                # Calculate loss
                loss = criterion(outputs_projected, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
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
                    
                    # Use a linear layer to project from hidden_size to 1
                    outputs_projected = outputs[:, :, 0].unsqueeze(-1)  # Just take first dimension of hidden state
                    
                    # Calculate loss
                    loss = criterion(outputs_projected, targets)
                    
                else:  # quantum model
                    outputs = []
                    h = None
                    
                    # Process each time step in the sequence
                    for t in range(seq_length):
                        output, h = model(inputs[:, t], h)
                        outputs.append(output)
                    
                    # Stack outputs for the full sequence
                    outputs = torch.stack(outputs, dim=1)
                    
                    # Use a linear layer to project from hidden_size to 1
                    outputs_projected = outputs[:, :, 0].unsqueeze(-1)  # Just take first dimension of hidden state
                    
                    # Calculate loss
                    loss = criterion(outputs_projected, targets)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}')
    
    # Save model and training history
    model_file = os.path.join(save_path, f'{model_name}.pth')
    history_file = os.path.join(save_path, f'{model_name}_history.pkl')
    
    torch.save(model.state_dict(), model_file)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_type.capitalize()} RNN Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_name}_loss.png'))
    
    return model, train_losses, val_losses, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Models')
    parser.add_argument('--model', type=str, choices=['classical', 'quantum'], 
                       required=True, help='Model type to train')
    parser.add_argument('--input_size', type=int, default=1, 
                       help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=32, 
                       help='Hidden state size')
    parser.add_argument('--n_qubits', type=int, default=5, 
                       help='Number of qubits (for quantum model)')
    parser.add_argument('--n_qlayers', type=int, default=1, 
                       help='Number of quantum layers (for quantum model)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=12, 
                       help='Sequence length for time series')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='models', 
                       help='Path to save model and results')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        n_qubits=args.n_qubits,
        n_qlayers=args.n_qlayers,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        epochs=args.epochs,
        save_path=args.save_path
    )
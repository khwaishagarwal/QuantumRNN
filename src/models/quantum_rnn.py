import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter

class QiskitTorchLayer(nn.Module):
    """Custom PyTorch layer that executes a Qiskit quantum circuit"""
    
    def __init__(self, n_qubits, n_layers):
        super(QiskitTorchLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize the trainable parameters for the quantum circuit
        # 3 rotation parameters per qubit per layer plus 1 entanglement parameter per qubit per layer
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 4) * 0.1)
        
        # Create a Qiskit simulator
        self.backend = Aer.get_backend('qasm_simulator')
    
    def _create_circuit(self, inputs):
        """Create a quantum circuit with the given input values and trainable weights"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Encode input data as angles of rotation gates
        for i in range(min(self.n_qubits, len(inputs))):
            qc.rx(inputs[i].item(), i)
        
        # Apply trainable variational layers
        for l in range(self.n_layers):
            # Rotation gates with trainable parameters
            for i in range(self.n_qubits):
                qc.rx(self.weights[l, i, 0].item(), i)
                qc.rz(self.weights[l, i, 1].item(), i)
                qc.rx(self.weights[l, i, 2].item(), i)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qc.cx(i, i+1)
            
            # Connect last qubit to first qubit to create a cyclic entanglement
            qc.cx(self.n_qubits - 1, 0)
            
            # Additional rotation gates with trainable parameters
            for i in range(self.n_qubits):
                qc.rz(self.weights[l, i, 3].item(), i)
        
        # Add measurements
        qc.measure_all()
        
        return qc
        
    def forward(self, x):
        """Forward pass of the quantum layer"""
        batch_size = x.shape[0]
        expectations = torch.zeros(batch_size, self.n_qubits)
        
        for b in range(batch_size):
            inputs = x[b]
            
            # Create and execute the quantum circuit
            qc = self._create_circuit(inputs)
            
            # Execute circuit using the simulator directly
            transpiled_qc = transpile(qc, self.backend)
            job = self.backend.run(transpiled_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            # Convert counts to expectation values
            for i in range(self.n_qubits):
                exp_val = 0
                for bitstring, count in counts.items():
                    # Calculate expectation value for PauliZ measurement
                    # Need to reverse the string as Qiskit reads bits from right to left
                    if len(bitstring) >= i + 1:  # Check if the bitstring is long enough
                        bit = bitstring[-(i+1)]  # Reverse indexing due to Qiskit's ordering
                        exp_val += (-1)**int(bit) * count / 1024
                
                expectations[b, i] = exp_val
                
        return expectations

class QRNN(nn.Module):
    """Quantum Recurrent Neural Network"""
    
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        n_qlayers=1,
        batch_first=True
    ):
        super(QRNN, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        
        # Classical pre-processing layer
        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        
        # Quantum circuit layer
        self.quantum_layer = QiskitTorchLayer(n_qubits, n_qlayers)
        
        # Classical post-processing layer
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)
    
    def forward(self, x, h=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate input and hidden state
        combined = torch.cat((x, h), dim=1)
        
        # Pass through the classical pre-processing layer
        combined = self.clayer_in(combined)
        
        # Apply the quantum circuit layer
        q_out = self.quantum_layer(combined)
        
        # Output layer
        out = self.clayer_out(q_out)
        
        # Update the hidden state
        new_h = out
        
        return out, new_h

class SimpleRNN(nn.Module):
    """Simple Recurrent Neural Network for baseline comparison"""
    
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # RNN cell weights
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, x, h=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # RNN cell computation
        h_next = self.activation(self.input_to_hidden(x) + self.hidden_to_hidden(h))
        
        return h_next, h_next
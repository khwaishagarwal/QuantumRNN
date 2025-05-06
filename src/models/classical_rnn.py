import torch
import torch.nn as nn

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

class ClassicalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ClassicalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
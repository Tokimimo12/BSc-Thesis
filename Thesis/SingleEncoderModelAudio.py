import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class SingleEncoderModelAudio(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout_rate, output_size):
        super(SingleEncoderModelAudio, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate    
        self.output_size = output_size
        
        # GRU layer
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_dim,  # Correct parameter is hidden_size
                          num_layers=self.num_layers, 
                          batch_first=True,  # Input is [batch_size, seq_length, input_size]
                          dropout=self.dropout_rate if self.num_layers > 1 else 0)

        # Fully connected output layer
        self.fc2 = nn.Linear(self.hidden_dim, output_size)
        
        # Tanh activation function to scale outputs between -3 and 3
        self.tanh = nn.Tanh()

    def forward(self, x, lengths):
        batch_size = lengths.size(0)

        # GRU forward pass
        _, hidden = self.gru(x)  # `hidden` shape: (num_layers, batch_size, hidden_dim)
        hidden = hidden[-1]  # Use the last hidden state
        
        # Fully connected layers
        output = self.fc2(hidden)  # Output of shape [batch_size, output_size]
        
        # Apply tanh and scale the output to [-3, 3]
        output = 3 * self.tanh(output)
        
        return output

    def compute_loss(self, batch_pred, y_labels):
        # Use Mean Squared Error loss for regression
        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_pred, y_labels)
        return loss

    def create_optimizer(self, lr):
        # Optimizer (Adam) with the specified learning rate
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

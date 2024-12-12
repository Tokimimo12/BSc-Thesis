import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class SingleEncoderModelText(nn.Module):
    def __init__(self, dic_size, use_glove, encoder_size, num_layers, hidden_dim, dr, output_size):
        super(SingleEncoderModelText, self).__init__()
        
        # Parameters
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.encoder_size = encoder_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dr = dr
        self.output_size = output_size
        
        
                
        if self.use_glove:
            print("Using glove")

        self.embed_dim = 300 if self.use_glove else 128 
    
        # Embedding layer
        self.embedding = nn.Embedding(self.dic_size, self.embed_dim)

            
        # GRU layer
        self.gru = nn.GRU(input_size=self.embed_dim, 
                          hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, 
                          dropout=self.dr, 
                          batch_first=True)
        
        # Fully connected output layers
        # self.fc1 = nn.Linear(self.hidden_dim, self.num_categories)  
        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)
        
        # Tanh activation function to scale outputs between -3 and 3
        self.tanh = nn.Tanh()

    def forward(self, x, lengths):
        batch_size = lengths.size(0)

        if (x.min() < 0 or x.max() >= self.dic_size):
            raise ValueError(
                f"Input indices out of range! Min index: {x.min()}, Max index: {x.max()}, Vocabulary size: {self.dic_size}"
            )
        
        embedded = self.embedding(x)  # [batch_size, seq_length] -> [batch_size, seq_length, embed_dim]
        
        gru_out, hidden = self.gru(embedded)  # gru_out: [batch_size, seq_length, hidden_dim], hidden: [num_layers, batch_size, hidden_dim]
        
        last_hidden = hidden[-1]  # Shape: [batch_size, hidden_dim]
        
        # fc1_out = self.fc1(last_hidden)  # [batch_size, num_categories]
        output = self.fc2(last_hidden)  # [batch_size, output_size]
        
        output = 3 * self.tanh(output)
        
        return output
    
    def compute_loss(self, batch_pred, y_labels):
        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_pred, y_labels)
        return loss

    def create_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

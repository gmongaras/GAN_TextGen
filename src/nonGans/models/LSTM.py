import torch
from torch import nn
from ..blocks.LSTM import LSTM_Module



class LSTM(nn.Module):
    # input_size (E) - The embedding size of the input.
    # hidden_size (H) - The size of the hidden state
    # vocab_size (V) - The size of the vocab to predict from
    # layers - The number of LSTM blocks stacked ontop of each other
    # dropout - Rate to apply dropout to the LSTM block
    # device - Deivce to put the LSTM block on
    def __init__(self, input_size, hidden_size, vocab_size, layers, dropout, device):
        super(LSTM, self).__init__()
        
        # Create the LSTM model
        self.LSTM = LSTM_Module(input_size, hidden_size, layers, dropout, device)
        
        # Output Linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)


    # Forward method to get softmax outputs from
    # a sequence of inputs
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x):
        return self.linear(self.LSTM(x))





class LSTM_torch(nn.Module):
    # input_size (E) - The embedding size of the input.
    # hidden_size (H) - The size of the hidden state
    # vocab_size (V) - The size of the vocab to predict from
    # layers - The number of LSTM blocks stacked ontop of each other
    # dropout - Rate to apply dropout to the LSTM block
    # device - Deivce to put the LSTM block on
    def __init__(self, input_size, hidden_size, vocab_size, layers, dropout, device):
        super(LSTM_torch, self).__init__()
        
        # Create the LSTM model
        self.LSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, batch_first=True).to(device)
        
        # Output Linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)


    # Forward method to get softmax outputs from
    # a sequence of inputs
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x):
        return self.linear(self.LSTM(x)[0])
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
        self.linear = nn.Linear(hidden_size, vocab_size, device=device)


    # Forward method to get outputs from
    # a sequence of inputs. Note, these are not
    # softmax output, rather linear outputs
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    #   context - Optional context which will be the initial context
    #             in the first LSTM block. Shape (layers, N, H)
    #   hidden - Optional hidden state which will be the initial hidden
    #             state in the first LSTM block. Shape (layers, N, H)
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x, context=None, hidden=None):
        return self.linear(self.LSTM(x, context))





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
        self.linear = nn.Linear(hidden_size, vocab_size, device=device)


    # Forward method to get softmax outputs from
    # a sequence of inputs
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    #   context - Optional context which will be the initial context
    #             in the first LSTM block. Shape (layers, N, H)
    #   hidden - Optional hidden state which will be the initial hidden
    #             state in the first LSTM block. Shape (layers, N, H)
    #   retain_output - Optional parameter. If True, context and
    #                   hidden state will be returned. If False,
    #                   only the output will be returned
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x, context=None, hidden=None, retain_output=False):
        if retain_output:
            o = self.LSTM(x, (hidden, context))
            return self.linear(o[0]), o[1]
        return self.linear(self.LSTM(x)[0])
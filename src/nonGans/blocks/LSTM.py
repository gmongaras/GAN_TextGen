import torch
from torch import nn




# Block used for the actual LSTM module.
class LSTM_Block(nn.Module):
    # input_size (E) - The embedding size of the input.
    # hidden_size (H) - The size of the hidden state
    # dropout - Rate to apply dropout to the LSTM block
    # device - Deivce to put the LSTM block on
    def __init__(self, input_size, hidden_size, dropout, device):
        super(LSTM_Block, self).__init__()
        
        # Used for proper weight initialization
        k = torch.sqrt(torch.tensor(1/hidden_size))
        
        # Used to get the output of the forget gate
        self.linear_if = nn.Linear(input_size, hidden_size, device=device)
        self.linear_if.weight.data.uniform_(-k, k)
        self.linear_if.bias.data.uniform_(-k, k)
        self.linear_hf = nn.Linear(hidden_size, hidden_size, device=device)
        
        # Used to get the output of the input gate
        self.linear_ii = nn.Linear(input_size, hidden_size, device=device)
        self.linear_ii.weight.data.uniform_(-k, k)
        self.linear_ii.bias.data.uniform_(-k, k)
        self.linear_hi = nn.Linear(hidden_size, hidden_size, device=device)
        self.linear_ig = nn.Linear(input_size, hidden_size, device=device)
        self.linear_ig.weight.data.uniform_(-k, k)
        self.linear_ig.bias.data.uniform_(-k, k)
        self.linear_hg = nn.Linear(hidden_size, hidden_size, device=device)
        
        # Used to get the output of the 
        self.linear_io = nn.Linear(input_size, hidden_size, device=device)
        self.linear_io.weight.data.uniform_(-k, k)
        self.linear_io.bias.data.uniform_(-k, k)
        self.linear_ho = nn.Linear(hidden_size, hidden_size, device=device)
        
        # Used for dropout
        self.dropout = nn.Dropout(dropout)
    
    
    # Inputs:
    #   x_t - The input at the current timestep of shape (N, E)
    #   h_t1 - The hidden state from the previous LSTM block of shape (N, H)
    #   c_t1 - The context from the previous LSTM block of shape (N, H)
    # Outputs:
    #   h_t - The output or prediction of the LSTM block of shape (N, H)
    #   h_t - The new hidden state matrix of shape (N, H)
    #   c_t - The new context matrix of shape (N, H)
    def forward(self, x_t, h_t1, c_t1):
        # Calculate the forget gate
        # f_t has shape (N, H)
        f_t = torch.sigmoid(self.linear_if(x_t) + self.linear_hf(h_t1))
        f_t = self.dropout(f_t)
        
        # Calculate the input gate
        # i_t has shape (N, H)
        # g_t has shape (N, H)
        i_t = torch.sigmoid(self.linear_ii(x_t) + self.linear_hi(h_t1))
        i_t = self.dropout(i_t)
        g_t = torch.tanh(self.linear_ig(x_t) + self.linear_hg(h_t1))
        g_t = self.dropout(g_t)
        
        # Get the context vector at the current timestep
        # c_t has shape (N, H)
        c_t = f_t*c_t1 + i_t*g_t
        
        # Calculate the output gate
        # o_t has shape (N, H)
        o_t = torch.sigmoid(self.linear_io(x_t) + self.linear_ho(h_t1))
        o_t = self.dropout(o_t)
        
        # Calculate the final output. This is also the hidden
        # state for the next LSTM block
        # h_t has shape (N, H)
        h_t = o_t*torch.tanh(c_t)
        
        # Return the output, context, and hidden state
        return h_t, h_t, c_t





# This class is what will actually be used. It uses the
# LSTM blocks to handle stacked layers and should behave
# like the pytoch impementaiton of the LSTM
class LSTM_Module(nn.Module):
    # input_size (E) - The embedding size of the input.
    # hidden_size (H) - The size of the hidden state
    # layers - The number of LSTM blocks stacked ontop of each other
    # dropout - Rate to apply dropout to the LSTM block
    # device - Deivce to put the LSTM block on
    def __init__(self, input_size, hidden_size, layers, dropout, device):
        super(LSTM_Module, self).__init__()
        
        # Save variables that may be needed later
        self.E = input_size
        self.H = hidden_size
        self.layers = layers
        self.device = device
        
        # Initialize the LSTM blocks
        self.blocks = nn.ParameterList([LSTM_Block(input_size if i == 0 else hidden_size, hidden_size, dropout, device) for i in range(layers)])
    
    
    # Takes an input sequence and outputs the predicted next
    # word at each timestep
    # Inputs:
    #   x - The input sequence to get outputs for of shape (N, S, E)
    #   context - Optional context which will be the initial context
    #             in the first LSTM block. Shape (L, N, H)
    #   hidden - Optional hidden state which will be the initial hidden
    #             state in the first LSTM block. Shape (layers, N, H)
    # Outputs:
    #   Tensor of shape (N, S, H) where each part of the sequence if the
    #   predicted output for the next part of the sequence
    def forward(self, x, context=None, hidden=None):
        # Make sure the input is 3 dimensions
        assert len(x.shape) == 3, "Input should have 3 dimensions: (batch size (N), sequence length (S), embedding size (E))"
        
        # Get the batch size
        N = x.shape[0]
        
        # Initialize the hidden and context matricies to zeros
        if hidden == None:
            h_t = [torch.zeros(N, self.H, device=self.device) for i in range(self.layers)]
        else:
            h_t = hidden.contiguous()
            assert hidden.shape == (self.layers, N, self.H), "Hidden state must be of shape (layers, N, H)"
        if context == None:
            c_t = [torch.zeros(N, self.H, device=self.device) for i in range(self.layers)]
        else:
            c_t = context.contiguous()
            assert context.shape == (self.layers, N, self.H), "Context must be of shape (layers, N, H)"
        
        # The output will be of shape (N, S, H), but to make it easier,
        # it will start as (S, N, H)
        outputs = torch.zeros(x.shape[1], x.shape[0], self.H, device=self.device)
        
        # For each part of the sequence, get an output
        for s in range(0, x.shape[1]):
            x_l0 = x[:, s]
            
            # Iterate over all LSTM blocks
            for b in range(0, self.layers):
                # Feed in the hidden state, context for that layer
                # as well as the output of the previous layer
                x_l0, h_t[b], c_t[b] = self.blocks[b](x_l0, h_t[b], c_t[b])
            
            # Save the final output
            outputs[s] = x_l0
        
        # Return the final output as shape (N, S, H)
        return outputs.permute(1, 0, 2)
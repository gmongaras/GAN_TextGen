import torch
from torch import nn
from .LSTM import LSTM_torch
from ..blocks.Transformer import Transformer_Module





# This model is a combination of both a transformer and a LSTM.
# It is optimized to predict characters, not words.
class BothModel(nn.Module):
    # input_size (E) - The embedding size of the input.
    # vocab_size (V) - The size of the vocab to predict from
    # dropout - Rate to apply dropout to the model
    # device - Deivce to put the model on
    # saveDir - Directory to save model to
    # saveFile - File to save model to
    # num_heads - The number of heads for the Transformer model
    # hidden_size (H) - The size of the hidden state for the LSTM
    # T - Number of transformer blocks
    # L - The number of layers in the LSTM
    # W - max size of each word to encode
    def __init__(self, input_size, vocab, dropout, device, saveDir, saveFile, num_heads, hidden_size, T, L, W):
        super(BothModel, self).__init__()
        
        # Save the needed parameters
        self.input_size = input_size
        self.L = L
        self.W = W
        self.vocab = vocab
        self.V = len(vocab)
        
        # Create the transformer blocks
        self.trans = Transformer_Module(T, input_size, num_heads, input_size, device)
        
        # Linear layer to convert the context from
        # shape (N, E) -> (N, H)
        self.contextLinear = nn.Linear(input_size, hidden_size, device=device)
        
        # Create the LSTM. The LSTM takes
        # as input a tensor of shape (N, S, V)
        # and output a tensor of shape (N, S, V)
        self.LSTM = LSTM_torch(self.V, hidden_size, self.V, L, dropout, device)
        
        # Create the encoder to transform a matrix of shape:
        # (N, S, W, V) -> (N, S, E). This encoder transforms
        # word probabilities to encoded forms of each word.
        self.CharToWord_linear1 = nn.Linear(self.V, 1, device=device)
        self.CharToWord_linear2 = nn.Linear(W, input_size, device=device)
    
    
    
    
    # Input:
    #   x - A tensor of shape (N, S, E) where each sequence is a
    #       sentence of encoded words.
    # Output:
    #   A tensor of shape (N, W, V) which is a batch of new word
    #   predictions. Each word is split into W characters and
    #   each character is a linear encoding across the vocab.
    def forward(self, x):
        # Get the batch size and sequence length
        N = x.shape[0]
        S = x.shape[1]
        
        # Send the input through the transformer blocks
        # Shape: (N, S, E) -> (N, S, E)
        x = self.trans(x)
        
        # Get the context for the LSTM and expand it
        # to the shape (L, N, E)
        context = self.contextLinear(x[:, 0]).unsqueeze(0).expand(self.L, -1, -1)
        hidden = torch.zeros(context.shape, device=context.device)
        
        # Initialize the input into the LSTM as start tokens
        # The input will be of shape (N, 1, E)
        # and will expand to shape (N, S, E)
        x = torch.nn.functional.one_hot(torch.tensor([self.vocab["¶"]], dtype=torch.long), num_classes=self.V).float().unsqueeze(0).expand(N, -1, -1).to(context.device)
        
        # Get a new word from teh LSTM
        for w in range(0, self.W):
            # Get the LSTM output
            out = self.LSTM(x, context, hidden, retain_output=True)
            
            # Save the hidden state and context
            hidden, context = out[1]
            
            # Add the output to the input
            x = torch.cat((x, out[0][:, -1:]), dim=1)
        
        # The start token can be removed as the
        # generator didn't create it. Return this
        # as the output
        return x[:, 1:]
        
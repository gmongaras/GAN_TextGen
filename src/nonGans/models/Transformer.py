import torch
from torch import nn
from ..blocks.Transformer import Transformer_Module





class Transformer(nn.Module):
    # layers - Number of transformer layers to stack
    # E - Input embedding size
    # vocab_size - (V) Size of the vocab to predict from
    # num_heads - Number of heads in each MHA block
    # Linear_embedding - embedding size of the output of the
    #                    Linear block
    # device - Device to put the transformer blocks on
    def __init__(self, layers, E, vocab_size, num_heads, Linear_embedding, device):
        super(Transformer, self).__init__()
        
        # Create the Transformer module
        self.Transformer = Transformer_Module(layers, E, num_heads, Linear_embedding, device)
        
        # Output Linear layer
        self.linear = nn.Linear(E, vocab_size, device=device)
    
    # Forward method to get outputs from
    # a sequence of inputs. Note, these are not
    # softmax output, rather linear outputs
    # Inputs:
    #   x - A batch of sequence of inputs of shape (N, S, E)
    # Outputs:
    #   A tensor of shape (N, S, V) where each output along the
    #   S dimension is the output vector of the next item
    #   in the sequence.
    def forward(self, x):
        return self.linear(self.Transformer(x))
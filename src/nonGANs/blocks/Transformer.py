import torch
from torch import nn
from ..blocks.MHA import MHA





# Block used for the actual Transformer Module
class Transformer_Block(nn.Module):
    # Inputs:
    #   E - Input embedding size
    #   num_heads - Number of heads in each MHA block
    #   Linear_embedding - embedding size of the output of the
    #                      Linear block
    def __init__(self, E, num_heads, Linear_embedding):
        super(Transformer_Block, self).__init__()
        
        # The first MHA module
        self.MHA = MHA(E, E, E, num_heads)
        
        # Feed-foward block after the MHA blocks
        self.FF = nn.Linear(E, Linear_embedding)
        self.ReLU = nn.ReLU()
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E)
        self.LN2 = nn.LayerNorm(Linear_embedding)
    
    
    # Input:
    #   A tensor of the shape (N, S, E) that comes from the input
    #     embeddings we want to encode
    # Output:
    #   A tensor of shape (N, S, E) after transforming the input
    def forward(self, X):
        X_saved = X.clone()
        X = self.MHA(X, X)
        X += X_saved
        X = self.LN1(X.contiguous())
        
        X_saved = X.clone()
        X = self.FF(X)
        X = self.ReLU(X) + 0
        X += X_saved
        X = self.LN2(X.contiguous())
        return X



# This class is the Transformer that will actually be used. It
# uses transformer blocks to stack layers
class Transformer_Module(nn.Module):
    # layers - Number of transformer layers to stack
    # E - Input embedding size
    # num_heads - Number of heads in each MHA block
    # Linear_embedding - embedding size of the output of the
    #                    Linear block
    # device - Device to put the transformer blocks on
    def __init__(self, layers, E, num_heads, Linear_embedding, device):
        super(Transformer_Module, self).__init__()
        
        # Saved parameters
        self.E = E
        self.device = device
        
        # The transformer blocks
        self.blocks = nn.Sequential(
            *[
                Transformer_Block(E, num_heads, Linear_embedding) for i in range(layers)
            ]
        ).to(device)
        
        
    # Takes an input sequence and outputs the predicted next
    # word at each timestep
    # Inputs:
    #   x - The input sequence to get outputs for of shape (N, S, E)
    # Outputs:
    #   Tensor of shape (N, S, E) where each part of the sequence if the
    #   predicted output for the next part of the sequence
    def forward(self, x):
        # Make sure the input is 3 dimensions
        assert len(x.shape) == 3, "Input should have 3 dimensions: (batch size (N), sequence length (S), embedding size (E))"
        
        # Get the batch size
        N = x.shape[0]
        
        # The output will be of shape (N, S, E), but to make it easier,
        # it will start as (S, N, E)
        outputs = torch.zeros(x.shape[1], x.shape[0], self.E, device=self.device)
        
        # For each part of the sequence, get an output
        for s in range(1, x.shape[1]+1):
            
            # Get the output from the transformer blocks
            # Feed the sequence up to the current
            # s value. So, the sequence will start with
            # 1 token and end with all tokens
            
            # The output is the last prediction in the
            # output sequence
            out = self.blocks(x[:, :s, :])[:, -1]
            
            # Save the final output
            outputs[s-1] = out
        
        # Return the final output as shape (N, S, E)
        return outputs.permute(1, 0, 2)
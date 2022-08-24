import imp
import torch
from torch import nn
from ..blocks.MHA import MHA





class inTrans(nn.Module):
    # Inputs:
    #   E - Input embedding size
    #   num_heads - Number of heads in each MHA block
    #   hidden_size - Hidden size of the linear layer
    def __init__(self, E, num_heads, hidden_size=512):
        super(inTrans, self).__init__()
        
        # The first MHA module
        self.MHA = MHA(E, E, E, num_heads)
        
        # Feed-foward block after the MHA blocks
        self.FF1 = nn.Linear(E, hidden_size)
        self.Act = nn.GELU()
        self.FF2 = nn.Linear(hidden_size, E)
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E)
        self.LN2 = nn.LayerNorm(E)
    
    
    # Input:
    #   A tensor of the shape (N, S, E_2) that comes from the input
    #     embeddings we want to encode
    #   Optional tensor of shape (N, S)
    def forward(self, X, masks=None):
        X_saved = X.clone()
        X = self.MHA(X, X, masks)
        X += X_saved
        X = self.LN1(X)
        
        X_saved = X.clone()
        X = self.FF1(X)
        X = self.Act(X) + 0
        X = self.FF2(X)
        X += X_saved
        X = self.LN2(X)
        return X
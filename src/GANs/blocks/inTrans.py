import torch
from torch import nn
from ..blocks.MHA import MHA





class inTrans(nn.Module):
    # Inputs:
    #   E_I - Input embedding size
    #   E_O - Output embedding size
    #   num_heads - Number of heads in each MHA block
    #   hidden_size - Hidden size of the linear layer
    def __init__(self, E_I, E_O, num_heads, hidden_size=512):
        super(inTrans, self).__init__()

        self.E_I = E_I
        self.E_O = E_O
        
        # The first MHA module
        self.MHA = MHA(E_I, E_I, E_I, num_heads)
        
        # Feed-foward block after the MHA blocks
        self.FF1 = nn.Linear(E_I, hidden_size)
        self.Act = nn.GELU()
        self.FF2 = nn.Linear(hidden_size, E_O)
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E_I)
        self.LN2 = nn.LayerNorm(E_O)
    
    
    # Input:
    #   A tensor of the shape (N, S, E_2) that comes from the input
    #     embeddings we want to encode
    #   Optional tensor of shape (N, S)
    def forward(self, X, masks=None):
        X_saved = X.clone()
        X = self.MHA(X, X, masks)
        X = self.LN1(X)
        X += X_saved

        # No residual if the output is different from the input
        if self.E_I != self.E_O:
            X = self.FF1(X)
            X = self.Act(X) + 0
            X = self.FF2(X)
            X = self.LN2(X)
        else:
            X_saved = X.clone()
            X = self.FF1(X)
            X = self.Act(X) + 0
            X = self.FF2(X)
            X = self.LN2(X)
            X += X_saved
        return X
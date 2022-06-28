import imp
import torch
from torch import nn
from blocks.MHA import MHA





class inTrans(nn.Module):
    # Inputs:
    #   E_1 - Input embedding size
    #   num_heads - Number of heads in each MHA block
    #   FF_embedding - embedding size of the output of the
    #                  Feed-forward block
    def __init__(self, E, num_heads, FF_embedding):
        super(inTrans, self).__init__()
        
        # The first MHA module
        self.MHA = MHA(E, E, E, num_heads)
        
        # Feed-foward block after the MHA blocks
        self.FF = nn.Linear(E, FF_embedding)
        self.ReLU = nn.ReLU(inplace=False)
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E)
        self.LN2 = nn.LayerNorm(FF_embedding)
    
    
    # Input:
    #   A tensor of the shape (S, E_2) that comes from the input
    #     embeddings we want to encode
    def forward(self, X):
        X_saved = X.clone()
        X = self.MHA(X, X)
        X += X_saved
        X = self.LN1(X)
        
        X_saved = X.clone()
        X = self.FF(X)
        X = self.ReLU(X)
        X += X_saved
        X = self.LN2(X)
        return X
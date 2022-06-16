import imp
import torch
from torch import nn
from blocks.MHA import MHA





class outTrans(nn.Module):
    # Inputs:
    #   E_1 - Input embedding size that is used for the keys and
    #         values in the second MHA block
    #   E_2 - Input embedding from the input into
    #         the beginning of the transformer
    #   num_heads - Number of heads in each MHA block
    #   FF_embedding - embedding size of the output of the
    #                  Feed-forward block
    def __init__(self, E_1, E_2, num_heads, FF_embedding):
        super(outTrans, self).__init__()
        
        # The first MHA module with a mask
        self.MHA1 = MHA(E_2, E_2, E_2, num_heads, True)
        
        # Second MHA module without a mask and with an
        # input from a different source
        self.MHA2 = MHA(E_1, E_2, E_2, num_heads)
        
        # Feed-foward block after the MHA blocks
        self.FF = nn.Linear(E_2, FF_embedding)
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E_2)
        self.LN2 = nn.LayerNorm(E_2)
        self.LN3 = nn.LayerNorm(FF_embedding)
    
    
    # Input:
    #   A secondary tensor of the shape (S, E_1) that comes from
    #     the output of the input transformer blocks
    #   A primary tensor of the shape (N, S, E_2) that comes from
    #     the output sentence word embeddings
    def forward(self, X_1, X_2):
        X = self.MHA1(X_2, X_2)
        X = self.LN1(X)
        X += X_2
        
        X_saved = X.clone()
        X = self.MHA2(X_1, X)
        X = self.LN2(X)
        X += X_saved
        
        X_saved = X.clone()
        X = self.FF(X)
        X = self.LN3(X)
        return X + X_saved
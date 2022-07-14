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
    #   gausNoise - True to add pure gaussian noise in the B output blocks,
    #               False to not add gaussian noise
    #   num_heads - Number of heads in each MHA block
    #   FF_embedding - embedding size of the output of the
    #                  Feed-forward block
    #   device - Device to put tensors on
    def __init__(self, E_1, E_2, gausNoise, num_heads, FF_embedding, device):
        super(outTrans, self).__init__()
        self.device = device
        self.gausNoise = gausNoise
        
        # The first MHA module with a mask
        self.MHA1 = MHA(E_2, E_2, E_2, num_heads, True).to(device)
        
        # Second MHA module without a mask and with an
        # input from a different source
        self.MHA2 = MHA(E_1, E_2, E_2, num_heads).to(device)
        
        # Feed-foward block after the MHA blocks
        self.FF = nn.Linear(E_2, FF_embedding, device=device)
        self.ReLU = nn.ReLU()
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E_2, device=device)
        self.LN2 = nn.LayerNorm(E_2, device=device)
        self.LN3 = nn.LayerNorm(FF_embedding, device=device)
    
    
    # Input:
    #   A secondary tensor of the shape (S, E_1) that comes from
    #     the output of the input transformer blocks
    #   A primary tensor of the shape (N, S, E_2) that comes from
    #     the output sentence word embeddings
    def forward(self, X_1, X_2):
        X = self.MHA1(X_2, X_2)
        X += X_2
        X = self.LN1(X)
        
        # Add gaussian noise if set to true
        if self.gausNoise:
            X += torch.rand((X.shape), requires_grad=True, device=self.device)
        
        X_saved = X.clone()
        X = self.MHA2(X_1, X)
        X += X_saved
        X = self.LN2(X)
        
        X_saved = X.clone()
        X = self.FF(X)
        X = self.ReLU(X) + 0
        X += X_saved
        X = self.LN3(X)
        return X
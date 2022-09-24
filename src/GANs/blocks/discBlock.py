from ..blocks.inTrans import inTrans
import torch
from torch import nn
import numpy as np





class discBlock(nn.Module):
    # Inputs:
    #   T - The number of transformer blocks before pooling
    #   embedding_size_in - The size of the embeddings of the input into
    #                    this block
    #   embedding_size_out - The size of the embeddings of the output of
    #                    this block
    #   num_heads - Number of heads in the MHA module
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    def __init__(self, T, embedding_size_in, embedding_size_out, num_heads, hiddenSize, pooling):
        super(discBlock, self).__init__()

        # Linearly increase the embedding size
        Es = torch.linspace(embedding_size_in, embedding_size_out, T+1).int().numpy()

        # Maximum number of heads for each transformer block where the max size is
        # num_heads and the min size is 1
        heads = [0 for i in range(0, len(Es))]
        for i in range(0, T+1):
            E = Es[i]
            for v in range(1, min(num_heads+1, E+1)):
                if E%v == 0:
                    heads[i] = v
        
        # The transformer blocks
        self.trans = [inTrans(Es[i], Es[i+1], heads[i], hiddenSize) for i in range(T)]
        self.trans = nn.Sequential(*self.trans)
        
        # Average pooling layer to
        if pooling == "max":
            self.pool = nn.MaxPool1d(kernel_size=2) # Pool across 2 words
        elif pooling == "avg":
            self.pool = nn.AvgPool1d(kernel_size=2)
    
    
    
    # Input:
    #   3-D tensor of shape (N, S, embedding_size)
    #   Optional 3-D tensor of shape (N, S)
    # Output:
    #   3-D tensor of shape (N, S//2, 2)
    def forward(self, X, masks=None):
        if masks != None:
            for b in self.trans:
                X = b(X, masks)
        else:
            X = self.trans(X)
        if hasattr(self, 'pool'):
            # Using pooling only if the length of the
            # sequence is greater than 1
            if X.shape[1] > 1:
                X = self.pool(X.permute(0, 2, 1)).permute(0, 2, 1)
        return X
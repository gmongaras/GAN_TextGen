import torch
from torch import nn
from ..blocks.MHA import MHA





class outTrans(nn.Module):
    # Inputs:
    #   E - Embedding size used in this transformer block
    #   noiseDist - Distribution to sample noise from. Can be one of 
    #               ("norm", "unif", "trunc" (for truncated normal))
    #   num_heads - Number of heads in each MHA block
    #   device - Device to put tensors on
    #   hidden_size - Hidden size of the linear layer
    def __init__(self, E, noiseDist, num_heads, device, hidden_size=512):
        super(outTrans, self).__init__()
        self.device = device
        
        # The first self-MHA
        self.MHA1 = MHA(E, num_heads).to(device)
        
        # Second MHA module without a mask and with an
        # input from a different source
        self.MHA2 = MHA(E, num_heads).to(device)
        
        # Feed-foward block after the MHA blocks
        self.FF1 = nn.Linear(E, hidden_size, device=device)
        self.Act = nn.GELU()
        self.FF2 = nn.Linear(hidden_size, E, device=device)
        
        # Layer normalization blocks
        self.LN1 = nn.LayerNorm(E, device=device)
        self.LN2 = nn.LayerNorm(E, device=device)
        self.LN3 = nn.LayerNorm(E, device=device)

        # Noise distribution for the model
        self.noiseDist = noiseDist
        if noiseDist == "unif":
            self.dist = torch.distributions.uniform.Uniform(-1, 1)
        else:
            self.dist = torch.distributions.normal.Normal(0, 1)
    
    
    # Input:
    #   A primary tensor of the shape (N, S, E) that comes from
    #     the output sentence word embeddings
    #   A secondary tensor of the shape (N, S, E) that comes from
    #     the output of the input transformer blocks
    def forward(self, X_1, X_2):
        X = self.MHA1(X_1, X_1, X_1)
        X = self.LN1(X)
        X += X_1
        
        # Add noise using the given noise distribution
        # if the noise distribution is not None
        if self.noiseDist != None:
            if self.noiseDist == "trunc":
                a = -1.5
                b = 1.5
                noise = torch.nn.init.trunc_normal_(torch.empty(X.shape), a=a, b=b).to(self.device)
            else:
                noise = self.dist.sample(X.shape).float().to(self.device)
            X += noise
        
        X_saved = X.clone()
        X = self.MHA2(X_2, X_2, X)
        X = self.LN2(X)
        X += X_saved
        
        X_saved = X.clone()
        X = self.FF1(X)
        X = self.Act(X) + 0
        X = self.FF2(X)
        X = self.LN3(X)
        X += X_saved
        return X
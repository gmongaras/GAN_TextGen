from ..blocks.inTrans import inTrans
from torch import nn





class discBlock(nn.Module):
    # Inputs:
    #   T - The number of transformer blocks before pooling
    #   embedding_size - The size of the embeddings of the input into
    #                    this block
    #   num_heads - Number of heads in the MHA module
    #   pooling - What pooling mode should be used? ("avg", "max", or "none")
    def __init__(self, T, embedding_size, num_heads, pooling):
        super(discBlock, self).__init__()
        
        # The transformer blocks
        self.trans = [inTrans(embedding_size, num_heads, 512) for i in range(T)]
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
            X = self.pool(X.permute(0, 2, 1)).permute(0, 2, 1)
        return X
from blocks.inTrans import inTrans
from torch import nn





class discBlock(nn.Module):
    # Inputs:
    #   embedding_size - The size of the embeddings of the input into
    #                    this block
    #   sequence_length - The length of the sequence as input
    #   num_heads - Number of heads in the MHA module
    def __init__(self, embedding_size, sequence_length, num_heads):
        super(discBlock, self).__init__()
        
        # The transformer block
        self.trans = inTrans(embedding_size, num_heads, embedding_size)
        
        # Average pooling layer to
        self.pool = nn.AdaptiveAvgPool1d(sequence_length//2)
    
    
    
    # Input:
    #   3-D tensor of shape (N, S, embedding_size)
    # Output:
    #   3-D tensor of shape (N, S//2, 2)
    def forward(self, X):
        X = self.trans(X).permute(0, 2, 1)
        X = self.pool(X)
        return X.permute(0, 2, 1)
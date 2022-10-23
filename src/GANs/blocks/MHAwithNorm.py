from torch import nn
from ..blocks.MHA import MHA




class MHAwithNorm(nn.Module):
    # Inputs:
    #   E_1 - Input embedding size of X_1, the first input tensor
    #   E_2 - Input embedding size of X_2, the second input tensor
    #   output_embedding - Size to encode the embeddings for both
    #                      input tensors.
    #   num_heads - Number of heads in the MHA module
    def __init__(self, E_1, E_2, output_embedding, num_heads):
        super(MHAwithNorm, self).__init__()
        
        
        # MHA block
        self.MHA = MHA(E_1, E_2, output_embedding, num_heads)
        
        # Layer norm block
        self.LN = nn.LayerNorm(output_embedding)
        
    
    # 2 input tensors. The first input tensor X_1 will encode the
    # value and the query. The second tensor X-2 will encode
    # the query.
    # Input:
    #   A 3-D tensor of shape (N, S, E_1)
    #   A 3-D tensor of shape (N, S, E_2)
    # Output:
    #   A 3-D tensor of shape (N, S, output_embedding)
    def forward(self, X_1, X_2):
       # MHA output
       O = self.MHA(X_1, X_2)
       
       # Return the normalized output
       return self.LN(O + X_2)
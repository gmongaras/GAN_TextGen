import torch
from torch import nn






class MHA(nn.Module):
    # Inputs:
    #   E_1 - Input embedding size of X_1, the first input tensor
    #   E_2 - Input embedding size of X_2, the second input tensor
    #   output_embedding - Size to encode the embeddings for both
    #                      input tensors.
    #   num_heads - Number of heads in the MHA module
    #   mask - True to use a mask so the model doesn't look ahead,
    #           False otherwise
    def __init__(self, E_1, E_2, output_embedding, num_heads, mask=False):
        super(MHA, self).__init__()
        
        
        # Saved variables
        self.mask = mask
        
        # Key, query, value weights
        self.value_weights = nn.Linear(E_1, output_embedding)
        self.key_weights = nn.Linear(E_1, output_embedding)
        self.query_weights = nn.Linear(E_2, output_embedding)
        
        # MHA module
        self.MultiHeadAtt = nn.MultiheadAttention(output_embedding, num_heads, batch_first=True)
    
    
    # 2 input tensors. The first input tensor X_1 will encode the
    # value and the query. The second tensor X-2 will encode
    # the query.
    # Input:
    #   A 3-D tensor of shape (N, S, E_1)
    #   A 3-D tensor of shape (N, S, E_2)
    # Output:
    #   A 3-D tensor of shape (N, S, output_embedding)
    def forward(self, X_1, X_2):
        # Get the key, query, value embedings
        X_2 = X_2
        query = self.query_weights(X_2)
        value = torch.broadcast_to(self.value_weights(X_1), query.shape)
        key = torch.broadcast_to(self.key_weights(X_1), query.shape)
        
        # Get the MHA valeu and return it
        if self.mask:
            return self.MultiHeadAtt(query, key, value)[0]
        else:
            return self.MultiHeadAtt(query, key, value)[0]
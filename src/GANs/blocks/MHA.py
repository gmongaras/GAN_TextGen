from torch import nn





class MHA(nn.Module):
    # Inputs:
    #   E - Size to encode the embeddings for both input tensors.
    #   num_heads - Number of heads in the MHA module
    def __init__(self, E, num_heads):
        super(MHA, self).__init__()
        
        # MHA module
        self.MultiHeadAtt = nn.MultiheadAttention(E, num_heads, batch_first=True)
    
    
    # Input:
    #   Q - A 3-D tensor of shape (N, S, E)
    #   K - A 3-D tensor of shape (N, S, E)
    #   V - A 3-D tensor of shape (N, S, E)
    #   masks - Optional 3-D tensor of shape (N, S)
    # Output:
    #   A 3-D tensor of shape (N, S, output_embedding)
    def forward(self, Q, K, V, masks=None):
        # Get the MHA value and return it
        return self.MultiHeadAtt(Q, K, V, attn_mask=masks, need_weights=False)[0]
import torch
from torch import nn





# Attention for a single head
class SelfAttention(nn.Module):
    # Inputs
    #   embedding_dim - Embedding dimension to embed the keys, queries, and values
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()

        # Attention scale
        self.scale = 1/torch.sqrt(embedding_dim)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)


    # Input
    #   Q - Queries of shape (N, S, embedding_dim)
    #   K - Keys of shape (N, S, embedding_dim)
    #   V - Values of shape (N, S, embedding_dim)
    def forward(self, Q, K, V, mask=None):
        if mask == None:
            return torch.matmul(
                self.softmax(
                    (torch.matmul(Q, torch.swapaxes(K, -1, -2)))*self.scale)
                    , V
            )
        return torch.matmul(
            self.softmax(
                (torch.matmul(Q, torch.swapaxes(K, -1, -2)))*self.scale
                + mask)
                , V
        )


class MyMHA(nn.Module):
    # Inputs
    #   input_dim - Embedding dimension of the keys, queries, and values
    #   embedding_dim - Embedding dimension to embed the keys, queries, and values
    #   num_heads - Number of heads in the MHA mechanism
    #   use_bias - True to embed using a bias, False otherwise
    def __init__(self, input_dim, embedding_dim, num_heads, use_bias=True):
        super(MyMHA, self).__init__()

        assert embedding_dim%num_heads == 0, "Embedding dim must be divisible by number of heads"
        self.head_dim = embedding_dim//num_heads

        # Weight matrices to encode the key, query, and value
        # (N, S, input_embedding) -> (N, S, output_embedding)
        self.query_weights = nn.Linear(input_dim, embedding_dim, bias=use_bias)
        self.key_weights = nn.Linear(input_dim, embedding_dim, bias=use_bias)
        self.value_weights = nn.Linear(input_dim, embedding_dim, bias=use_bias)


    # Input
    #   Q - Queries of shape (N, S, input_dim)
    #   K - Keys of shape (N, S, input_dim)
    #   V - Values of shape (N, S, input_dim)
    def forward(self, Q, K, V):
        # Reshape the input to be of shape (N, S, num_heads, input_dim)


        # Project the Q, K, and V
        Q = self.query_weights(Q)
        K = self.key_weights(K)
        V = self.value_weights(V)






class MHA(nn.Module):
    # Inputs:
    #   E_1 - Input embedding size of X_1, the first input tensor
    #   E_2 - Input embedding size of X_2, the second input tensor
    #   output_embedding - Size to encode the embeddings for both
    #                      input tensors.
    #   num_heads - Number of heads in the MHA module
    def __init__(self, E_1, E_2, output_embedding, num_heads):
        super(MHA, self).__init__()
        
        
        # Key, query, value weights
        # self.value_weights = nn.Linear(E_2, output_embedding, bias=False)
        # self.key_weights = nn.Linear(E_1, output_embedding, bias=False)
        # self.query_weights = nn.Linear(E_1, output_embedding, bias=False)
        
        # MHA module
        self.MultiHeadAtt = nn.MultiheadAttention(output_embedding, num_heads, batch_first=True)
    
    
    # 2 input tensors. The first input tensor X_1 will encode the
    # key and the query. The second tensor X-2 will encode
    # the values.
    # Input:
    #   A 3-D tensor of shape (N, S, E_1)
    #   A 3-D tensor of shape (N, S, E_2)
    #   Optional 3-D tensor of shape (N, S)
    # Output:
    #   A 3-D tensor of shape (N, S, output_embedding)
    def forward(self, X_1, X_2, masks=None):
        # Get the key, query, value embedings
        # value = self.value_weights(X_2)
        # query = self.query_weights(X_1)
        # key = self.key_weights(X_1)

        value = X_2
        query = X_1
        key = X_1
        
        # Get the MHA value and return it
        return self.MultiHeadAtt(query, key, value, attn_mask=masks, need_weights=False)[0]
import torch
from torch import nn
import math



# Thanks to the following tutorial for this awesome code!
# https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, embedding_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2)*
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * embedding_size)
        pe[:, 1::2] = torch.cos(position * embedding_size)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
import torch
from torch import nn
from .LSTM import LSTM
from ..blocks.Transformer import Transformer_Module





# This model is a combination of both a transformer and a LSTM.
# It is optimized to predict characters, not words.
class BothModel(nn.Module):
    # T - Number of transformer blocks
    # W - max size of each word to encode
    def __init__(self, T):
        ;
import math
import torch
import torch.nn as nn
from torch import Tensor

# Adapted from here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# TODO: If there is time, use relative positional embeddings instead
#   See: https://arxiv.org/pdf/1803.02155.pdf and https://arxiv.org/pdf/1809.04281.pdf
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe
        return self.dropout(x)
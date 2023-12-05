import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to the inputs to provide information about the
    relative position of tokens.

    Args:
        d_model (int): The embedding dimension size.
        drop_prob (float): Dropout probability to apply after encoding.
        max_len (int): Maximum sequence length for positional encoding.

    Attributes:
        pe (Tensor): The precomputed positional encodings' tensor.

    """
    def __init__(self, d_model: int, drop_prob: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1), :]
        # return self.dropout(x)
        return x

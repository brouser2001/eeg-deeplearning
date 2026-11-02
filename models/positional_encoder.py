import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Input/output:
        x: (B, S, D)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # (S, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # (D/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        # (S, D)
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as (1, S, D) so it broadcasts over batch dimension
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, S, D)
        """
        S = x.size(1)

        x = x + self.pe[:, :S, :]

        return self.dropout(x)

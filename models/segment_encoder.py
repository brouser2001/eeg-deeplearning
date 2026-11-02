import torch.nn as nn
from torch import Tensor

class SegmentEncoder(nn.Module):
    """
    Converts each multi-channel signal window into one D-dimensional token.

    Input:
        x: (B, S, C, W)

    Output:
        (B, S, D)
    """

    def __init__(
        self,
        window_size: int,
        d_model: int,
        in_channels: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                d_model // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv1d(
                d_model // 2,
                d_model,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, S, C, W)

        Returns:
            (B, S, D)
        """

        B, S, C, W = x.shape

        # Treat each window as an independent CNN example.
        #
        # (B, S, C, W)
        #      ↓
        # (B*S, C, W)
        x = x.reshape(B * S, C, W)

        # CNN
        #
        # (B*S, C, W)
        #      ↓
        # (B*S, D, 1)
        x = self.net(x)

        # (B*S, D, 1)
        #      ↓
        # (B*S, D)
        x = x.squeeze(-1)

        # (B*S, D)
        #      ↓
        # (B, S, D)
        x = x.reshape(B, S, -1)

        return x

import torch.nn as nn
from torch import Tensor
from .segment_encoder import SegmentEncoder
from .positional_encoder import PositionalEncoding

class SignalTransformerClassifier(nn.Module):
    """
    Transformer classifier for variable-length sequences of
    signal windows.

    Everything outside the internal Conv1d uses:

        (B, S, D)

    Input:
        x:
            (B, S, W)

        pad_mask:
            (B, S)

            False = real window
            True  = padding window

    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        in_channels: int = 1,
    ):
        super().__init__()

        self.d_model = d_model

        # --------------------------------------------------
        # 1. Convert each window into a token
        # --------------------------------------------------

        self.segment_encoder = SegmentEncoder(
            window_size=window_size,
            d_model=d_model,
            in_channels=in_channels,
        )

        # --------------------------------------------------
        # 2. Add position information
        # --------------------------------------------------

        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_seq_len,
        )

        # --------------------------------------------------
        # 3. Transformer
        # --------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # --------------------------------------------------
        # 4. Classification head
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: Tensor,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:
                (B, S, W)

            pad_mask:
                (B, S)
                False = valid token
                True = padding token

        Returns:
            logits:
                (B, num_classes)
        """

        # --------------------------------------------------
        # Encode windows
        # --------------------------------------------------

        # (B, S, W)
        #      ↓
        # (B, S, D)
        x = self.segment_encoder(x)

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------

        # (B, S, D) -> (B, S, D)
        x = self.pos_encoding(x)

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        # (B, S, D) -> (B, S, D)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=pad_mask,
        )

        # --------------------------------------------------
        # Masked mean pooling
        # --------------------------------------------------

        if pad_mask is None:

            # All tokens are valid
            pooled = x.mean(dim=1)

        else:

            # pad_mask:
            # (B, S)
            #
            # True  = padding
            # False = valid

            valid_mask = (~pad_mask).unsqueeze(-1)

            # valid_mask:
            # (B, S, 1)

            x_masked = x * valid_mask

            # Number of valid tokens for each sample
            lengths = valid_mask.sum(dim=1).clamp(min=1)

            # Sum valid tokens and divide by number
            pooled = x_masked.sum(dim=1) / lengths

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        # (B, D) -> (B, num_classes)
        logits = self.classifier(pooled)

        return logits

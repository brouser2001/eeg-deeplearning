import torch.nn as nn
from torch import Tensor
from .segment_encoder import SegmentEncoder
from .positional_encoder import PositionalEncoding

class SignalTransformerClassifierMulticlass(nn.Module):
    """
    Multi-task Transformer classifier for variable-length sequences
    of signal windows.

    A single shared encoder (segment encoder + positional encoding +
    Transformer) feeds into one or more task-specific classification heads.

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
        logits: dict[str, Tensor]
            One (B, num_classes) tensor per registered task.
    """

    def __init__(
        self,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        tasks: dict[str, int] | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        in_channels: int = 1,
    ):
        """
        Args:
            tasks:
                Mapping of task name → number of output classes.
                e.g. {"emotion": 5, "speech_act": 12, "speaker": 8}

                If `None`, a single head named "default" with 3 classes
                is created (preserves old behaviour).
        """
        super().__init__()

        if tasks is None:
            tasks = {"default": 3}

        self.tasks = tasks
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
        # 3. Transformer (shared backbone)
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
        # 4. Task-specific classification heads
        # --------------------------------------------------

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                )
                for name, num_classes in tasks.items()
            }
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _masked_pool(self, x: Tensor, pad_mask: Tensor | None) -> Tensor:
        """Masked mean-pool over the sequence dimension.

        Args:
            x:        (B, S, D)
            pad_mask: (B, S)  True = padding, False = valid

        Returns:
            (B, D)
        """
        if pad_mask is None:
            return x.mean(dim=1)

        valid_mask = (~pad_mask).unsqueeze(-1)  # (B, S, 1)
        x_masked = x * valid_mask
        lengths = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        return x_masked.sum(dim=1) / lengths  # (B, D)

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        pad_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            x:
                (B, S, W)

            pad_mask:
                (B, S)
                False = valid token
                True  = padding token

        Returns:
            logits:
                dict[str, Tensor] with shape (B, num_classes) per task.
        """

        # --------------------------------------------------
        # Encode windows
        # --------------------------------------------------

        # (B, S, W) → (B, S, D)
        x = self.segment_encoder(x)

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------

        # (B, S, D) → (B, S, D)
        x = self.pos_encoding(x)

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        # (B, S, D) → (B, S, D)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=pad_mask,
        )

        # --------------------------------------------------
        # Masked mean pooling
        # --------------------------------------------------

        # (B, S, D) → (B, D)
        pooled = self._masked_pool(x, pad_mask)

        # --------------------------------------------------
        # Task-specific classification heads
        # --------------------------------------------------

        logits = {name: head(pooled) for name, head in self.heads.items()}

        return logits

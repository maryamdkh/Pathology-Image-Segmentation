import torch
import torch.nn as nn
from typing import Optional


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dim: Optional[int] = None,
    ):
        """
        Classification model using a pretrained encoder.

        Args:
            encoder (nn.Module): Feature extractor (TimmUniversalEncoder)
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
            hidden_dim (Optional[int]): If set, adds an extra FC layer
        """
        super().__init__()
        self.encoder = encoder

        # Last encoder stage channels
        encoder_out_channels = encoder.out_channels[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)

        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(encoder_out_channels, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoder_out_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder returns multi-scale features
        features = self.encoder(x)

        # Use the deepest feature map
        x = features[-1]

        x = self.pool(x)
        x = self.classifier(x)

        return x

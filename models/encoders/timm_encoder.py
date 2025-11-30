import torch.nn as nn
from timm import create_model


class TIMMEncoder(nn.Module):
    """
    Generic encoder wrapper for any TIMM backbone that supports features_only=True.
    Automatically adapts feature maps for SMP decoders (e.g., UNet++).
    """
    def __init__(self, model_name: str, pretrained: bool = True, **kwargs):
        super().__init__()
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )
        self.output_stride = 32  # typical max downsampling

        # Required by SMP decoders: list of channel sizes per stage.
        self.out_channels = self.backbone.feature_info.channels()

        # UNet++ expects 5 stages. Pad if fewer.
        if len(self.out_channels) < 5:
            pad = 5 - len(self.out_channels)
            self.out_channels = [0] * pad + self.out_channels

    def set_in_channels(self, in_channels, **kwargs):
        # no-op: SMP calls this, but our encoder already supports 3 channels
        pass
    def forward(self, x):
        feats = self.backbone(x)

        # Ensure 5-level output for SMP decoders
        if len(feats) < 5:
            pad = 5 - len(feats)
            feats = [None] * pad + feats

        return feats

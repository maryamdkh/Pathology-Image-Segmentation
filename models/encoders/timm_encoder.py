
from typing import Any

import timm
import torch
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    various model architectures, including traditional-style and transformer-style models.

    Features:
        - Supports configurable depth and output stride.
        - Ensures consistent multi-level feature extraction across diverse models.
        - Compatible with convolutional and transformer-like backbones.
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the encoder.

        Args:
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
        # At the moment we do not support models with more than 5 stages,
        # but can be reconfigured in the future.
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__()
        self.name = name

        # Default model configuration for feature extraction
        common_kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # Ｎot all models support output stride argument, drop it by default
        if output_stride == 32:
            common_kwargs.pop("output_stride")

        # Load a temporary model to analyze its feature hierarchy
        try:
            with torch.device("meta"):
                tmp_model = timm.create_model(name, features_only=True)
        except Exception:
            tmp_model = timm.create_model(name, features_only=True)

        # Check if model output is in channel-last format (NHWC)
        self._is_channel_last = getattr(tmp_model, "output_fmt", None) == "NHWC"

        # Determine the model's downsampling pattern and set hierarchy flags
        encoder_stage = len(tmp_model.feature_info.reduction())
        reduction_scales = list(tmp_model.feature_info.reduction())

        if reduction_scales == [2 ** (i + 2) for i in range(encoder_stage)]:
            # Transformer-style downsampling: scales (4, 8, 16, 32)
            self._is_transformer_style = True
            self._is_vgg_style = False
        elif reduction_scales == [2 ** (i + 1) for i in range(encoder_stage)]:
            # Traditional-style downsampling: scales (2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = False
        elif reduction_scales == [2**i for i in range(encoder_stage)]:
            # Vgg-style models including scale 1: scales (1, 2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = True
        else:
            raise ValueError("Unsupported model downsampling pattern.")

        if self._is_transformer_style:
            # Transformer-like models (start at scale 4)
            if "tresnet" in name:
                # 'tresnet' models start feature extraction at stage 1,
                # so out_indices=(1, 2, 3, 4) for depth=5.
                common_kwargs["out_indices"] = tuple(range(1, depth))
            else:
                # Most transformer-like models use out_indices=(0, 1, 2, 3) for depth=5.
                common_kwargs["out_indices"] = tuple(range(depth - 1))

            timm_model_kwargs = _merge_kwargs_no_duplicates(common_kwargs, kwargs)
            self.model = timm.create_model(name, **timm_model_kwargs)

            # FIX: Create meaningful features instead of zero channels
            # Use a convolution to create proper features for the missing scale
            self.dummy_conv = nn.Conv2d(
                in_channels,
                64,  # Or any reasonable number like 32, 64, 96
                kernel_size=3,
                stride=2,
                padding=1
            )

            # Update out_channels to include the real channel count
            self._out_channels = (
                [in_channels] + [64] + self.model.feature_info.channels()  # 64 instead of 0
            )
        else:
            if "dla" in name:
                # For 'dla' models, out_indices starts at 0 and matches the input size.
                common_kwargs["out_indices"] = tuple(range(1, depth + 1))
            if self._is_vgg_style:
                common_kwargs["out_indices"] = tuple(range(depth + 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )

            if self._is_vgg_style:
                self._out_channels = self.model.feature_info.channels()
            else:
                self._out_channels = [in_channels] + self.model.feature_info.channels()

        self._in_channels = in_channels
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            list[torch.Tensor]: List of feature maps at different scales.
        """
        features = self.model(x)

        # Convert NHWC to NCHW if needed
        if self._is_channel_last:
            features = [
                feature.permute(0, 3, 1, 2).contiguous() for feature in features
            ]

        # FIX: Replace dummy zero-channel feature with real features
        if self._is_transformer_style:
            # Create meaningful features for the missing scale instead of empty tensor
            dummy_features = self.dummy_conv(x)  # This creates [B, 64, H//2, W//2]
            features = [dummy_features] + features  # Use real features instead of empty

        # Add input tensor as scale 1 feature if `self._is_vgg_style` is False
        if not self._is_vgg_style:
            features = [x] + features

        return features


    @property
    def out_channels(self) -> list[int]:
        """
        Returns the number of output channels for each feature stage.

        Returns:
            list[int]: A list of channel dimensions at each scale.
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """
        Returns the effective output stride based on the model depth.

        Returns:
            int: The effective output stride.
        """
        return int(min(self._output_stride, 2**self._depth))

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if not key.startswith("model."):
                    new_key = "model." + key
                if "gernet" in self.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)


def _merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries, ensuring no duplicate keys exist.

    Args:
        a (dict): Base dictionary.
        b (dict): Additional parameters to merge.

    Returns:
        dict: A merged dictionary.
    """
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)

_EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, 1280),
    "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, 1792),
    "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT, 2048),
    "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT, 2304),
    "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, 2560),
}

_DEFAULT_VARIANT = "efficientnet_b3"


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for geolocation feature extraction."""

    def __init__(
        self, pretrained: bool = True, freeze: bool = False,
        variant: str = _DEFAULT_VARIANT,
    ):
        super().__init__()
        if variant not in _EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Unknown EfficientNet variant: {variant}. "
                f"Choose from: {list(_EFFICIENTNET_VARIANTS.keys())}"
            )
        model_fn, default_weights, feature_dim = _EFFICIENTNET_VARIANTS[variant]

        weights = default_weights if pretrained else None
        full_model = model_fn(weights=weights)
        self.features = full_model.features
        self.avgpool = full_model.avgpool
        self.feature_dim = feature_dim

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def unfreeze_last_n_blocks(self, n: int) -> None:
        total_blocks = len(self.features)
        for i, child in enumerate(self.features.children()):
            if i >= total_blocks - n:
                for param in child.parameters():
                    param.requires_grad = True

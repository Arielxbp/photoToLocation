from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B1 backbone for geolocation feature extraction."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        else:
            weights = None
        full_model = efficientnet_b1(weights=weights)
        self.features = full_model.features
        self.avgpool = full_model.avgpool
        self.feature_dim = 1280

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last N MBConv blocks for fine-tuning."""
        total_blocks = len(self.features)
        for i, child in enumerate(self.features.children()):
            if i >= total_blocks - n:
                for param in child.parameters():
                    param.requires_grad = True


def create_backbone(pretrained: bool = True, freeze: bool = False) -> EfficientNetBackbone:
    return EfficientNetBackbone(pretrained=pretrained, freeze=freeze)

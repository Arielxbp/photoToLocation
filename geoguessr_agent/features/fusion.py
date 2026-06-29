"""
Feature fusion module — combines EfficientNet image features with
CLIP-extracted clue features for improved geolocation prediction.
"""

import torch
import torch.nn as nn


class ClueEncoder(nn.Module):
    """Encodes a flat clue probability vector into a compact embedding."""

    def __init__(self, clue_dim: int, embed_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clue_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, clue_features: torch.Tensor) -> torch.Tensor:
        return self.net(clue_features)


class FeatureFusion(nn.Module):
    """
    Fuses vision backbone features with encoded clue embeddings
    and projects back to the dimension expected by the heads.
    """

    def __init__(
        self,
        image_dim: int,
        clue_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clue_encoder = ClueEncoder(clue_dim, embed_dim, dropout)
        fused_dim = image_dim + embed_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        clue_features: torch.Tensor,
    ) -> torch.Tensor:
        clue_embed = self.clue_encoder(clue_features)
        fused = torch.cat([image_features, clue_embed], dim=-1)
        return self.projection(fused)

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import EfficientNetBackbone, _DEFAULT_VARIANT
from .heads import ContinentHead, CoordinateHead, CountryHead, RegionHead


class GeoLocator(nn.Module):
    """
    Full geolocation model combining EfficientNet backbone with
    hierarchical classification heads (country, region, continent).
    """

    def __init__(
        self,
        num_countries: int = 95,
        num_regions: int = 1500,
        num_continents: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        backbone_name: str = _DEFAULT_VARIANT,
    ):
        super().__init__()
        self.backbone = EfficientNetBackbone(
            pretrained=pretrained, freeze=freeze_backbone, variant=backbone_name,
        )
        in_dim = self.backbone.feature_dim

        self.country_head = CountryHead(in_dim, num_countries, dropout)
        self.region_head = RegionHead(in_dim, num_regions, dropout)
        self.continent_head = ContinentHead(in_dim, num_continents, dropout)
        self.coord_head = CoordinateHead(in_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "country_logits": self.country_head(features),
            "region_logits": self.region_head(features),
            "continent_logits": self.continent_head(features),
            "coord_xyz": self.coord_head(features),
            "features": features,
        }

    def predict_country(self, x: torch.Tensor, top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k country indices and probabilities."""
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.softmax(outputs["country_logits"], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        return top_indices, top_probs

    def predict_region(self, x: torch.Tensor, top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k region indices and probabilities."""
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.softmax(outputs["region_logits"], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        return top_indices, top_probs

    def predict_full(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full prediction with all heads."""
        with torch.no_grad():
            outputs = self.forward(x)
            return {
                "country_probs": torch.softmax(outputs["country_logits"], dim=-1),
                "region_probs": torch.softmax(outputs["region_logits"], dim=-1),
                "continent_probs": torch.softmax(outputs["continent_logits"], dim=-1),
                "coord_xyz": outputs["coord_xyz"],
                "features": outputs["features"],
            }

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "num_countries": self.country_head.fc[-1].out_features,
                    "num_regions": self.region_head.fc[-1].out_features,
                    "num_continents": self.continent_head.fc[-1].out_features,
                    "backbone_name": self.backbone.feature_dim,  # stored for reference
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GeoLocator":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            num_countries=config["num_countries"],
            num_regions=config["num_regions"],
            num_continents=config["num_continents"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

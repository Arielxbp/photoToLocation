from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..features.fusion import FeatureFusion
from .backbone import _DEFAULT_VARIANT, EfficientNetBackbone
from .heads import ContinentHead, CoordinateHead, CountryHead, RegionHead


class GeoLocator(nn.Module):
    """
    Full geolocation model combining EfficientNet backbone with
    hierarchical classification heads (country, region, continent).

    Optionally fuses CLIP-extracted clue features via a FeatureFusion
    module for improved prediction. When ``clue_features`` is None,
    the model falls back to the image-only path.
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
        clue_feature_dim: Optional[int] = None,
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

        self.has_fusion = clue_feature_dim is not None
        if self.has_fusion:
            self.fusion = FeatureFusion(
                image_dim=in_dim,
                clue_dim=clue_feature_dim,
                output_dim=in_dim,
            )
        else:
            self.fusion = None  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        clue_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        if self.has_fusion and clue_features is not None:
            features = self.fusion(features, clue_features)
        return {
            "country_logits": self.country_head(features),
            "region_logits": self.region_head(features),
            "continent_logits": self.continent_head(features),
            "coord_xyz": self.coord_head(features),
            "features": features,
        }

    def predict_country(
        self,
        x: torch.Tensor,
        top_k: int = 5,
        clue_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k country indices and probabilities."""
        with torch.no_grad():
            outputs = self.forward(x, clue_features)
            probs = torch.softmax(outputs["country_logits"], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        return top_indices, top_probs

    def predict_region(
        self,
        x: torch.Tensor,
        top_k: int = 5,
        clue_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k region indices and probabilities."""
        with torch.no_grad():
            outputs = self.forward(x, clue_features)
            probs = torch.softmax(outputs["region_logits"], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        return top_indices, top_probs

    def predict_full(
        self,
        x: torch.Tensor,
        clue_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full prediction with all heads."""
        with torch.no_grad():
            outputs = self.forward(x, clue_features)
            return {
                "country_probs": torch.softmax(outputs["country_logits"], dim=-1),
                "region_probs": torch.softmax(outputs["region_logits"], dim=-1),
                "continent_probs": torch.softmax(outputs["continent_logits"], dim=-1),
                "coord_xyz": outputs["coord_xyz"],
                "features": outputs["features"],
            }

    def save(self, path: str) -> None:
        config = {
            "num_countries": self.country_head.fc[-1].out_features,
            "num_regions": self.region_head.fc[-1].out_features,
            "num_continents": self.continent_head.fc[-1].out_features,
            "backbone_name": self.backbone.feature_dim,
            "has_fusion": self.has_fusion,
        }
        if self.has_fusion and self.fusion is not None:
            config["clue_feature_dim"] = self.fusion.clue_encoder.net[0].in_features
        torch.save({"state_dict": self.state_dict(), "config": config}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GeoLocator":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        model = cls(
            num_countries=cfg["num_countries"],
            num_regions=cfg["num_regions"],
            num_continents=cfg["num_continents"],
            clue_feature_dim=cfg.get("clue_feature_dim"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

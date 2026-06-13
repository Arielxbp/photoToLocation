from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..model.geolocator import GeoLocator
from ..model.losses import latlng_to_rad
from ..data.panorama import (
    generate_view_crops,
    equirect_to_perspective,
    stitch_panorama,
    PanoramaInfo,
)


class InferencePipeline:
    """End-to-end inference: image → (lat, lng, country, region, continent)."""

    def __init__(
        self,
        model: GeoLocator,
        region_centroids: torch.Tensor,
        country_index: dict[str, int],
        idx_to_country: dict[int, str],
        region_index: Optional[dict[str, int]] = None,
        idx_to_region: Optional[dict[int, str]] = None,
        idx_to_continent: Optional[dict[int, str]] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.region_centroids = region_centroids.to(device)
        self.country_index = country_index
        self.idx_to_country = idx_to_country
        self.region_index = region_index or {}
        self.idx_to_region = idx_to_region or {}
        self.idx_to_continent = idx_to_continent or {}
        self.device = device

    def preprocess(self, image: Image.Image | np.ndarray | bytes | str | Path) -> torch.Tensor:
        """Preprocess an image from various input formats."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        img = img.resize((320, 180), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        tensor = torch.from_numpy((arr - mean) / std).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image | np.ndarray | bytes | str | Path) -> dict:
        """
        Full prediction from an image.
        Returns lat, lng, country name, country confidence, top-5 countries.
        """
        tensor = self.preprocess(image)
        outputs = self.model(tensor)

        country_probs = torch.softmax(outputs["country_logits"], dim=-1)
        top5_conf, top5_idx = torch.topk(country_probs, k=5, dim=-1)
        top5_conf = top5_conf.squeeze(0).cpu().tolist()
        top5_idx = top5_idx.squeeze(0).cpu().tolist()

        top5_countries = [
            {
                "country": self.idx_to_country.get(i, f"unknown_{i}"),
                "confidence": float(c),
            }
            for i, c in zip(top5_idx, top5_conf)
        ]

        region_probs = torch.softmax(outputs["region_logits"], dim=-1)
        top_region_probs, top_region_indices = torch.topk(region_probs, k=5, dim=-1)

        weighted_coords = (
            top_region_probs @ self.region_centroids[top_region_indices.squeeze(0)]
        )
        coords_rad = weighted_coords.squeeze(0)
        lat = float(torch.rad2deg(coords_rad[0]).item())
        lng = float(torch.rad2deg(coords_rad[1]).item())

        continent_probs = torch.softmax(outputs["continent_logits"], dim=-1)
        continent_idx = continent_probs.argmax(-1).item()
        continent = self.idx_to_continent.get(continent_idx, f"unknown_{continent_idx}")

        return {
            "latitude": max(-90.0, min(90.0, lat)),
            "longitude": max(-180.0, min(180.0, lng)),
            "country": top5_countries[0]["country"],
            "country_confidence": top5_countries[0]["confidence"],
            "continent": continent,
            "top5_countries": top5_countries,
        }

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """Batch prediction for multiple images."""
        tensors = torch.cat([self.preprocess(img) for img in images], dim=0).to(self.device)
        outputs = self.model(tensors)

        country_probs = torch.softmax(outputs["country_logits"], dim=-1)
        region_probs = torch.softmax(outputs["region_logits"], dim=-1)
        continent_probs = torch.softmax(outputs["continent_logits"], dim=-1)

        _, top_country_idx = country_probs.max(dim=-1)
        _, top_continent_idx = continent_probs.max(dim=-1)

        top_region_probs, top_region_indices = torch.topk(region_probs, k=5, dim=-1)

        batch = country_probs.shape[0]
        results = []

        for i in range(batch):
            weighted_coords = (
                top_region_probs[i] @ self.region_centroids[top_region_indices[i]]
            )
            lat = float(torch.rad2deg(weighted_coords[0]).item())
            lng = float(torch.rad2deg(weighted_coords[1]).item())

            top5_idx = torch.topk(country_probs[i], k=5).indices
            top5_conf = torch.topk(country_probs[i], k=5).values

            results.append({
                "latitude": max(-90.0, min(90.0, lat)),
                "longitude": max(-180.0, min(180.0, lng)),
                "country": self.idx_to_country.get(
                    top_country_idx[i].item(), "unknown"
                ),
                "country_confidence": float(country_probs[i].max().item()),
                "continent": self.idx_to_continent.get(
                    top_continent_idx[i].item(), "unknown"
                ),
            })

        return results

    @torch.no_grad()
    def predict_multi_view(
        self,
        crops: list[Image.Image],
    ) -> dict:
        """
        Multi-view fusion: run inference on multiple perspective crops and
        fuse the country logits by averaging probabilities across views.

        Args:
            crops: List of perspective-corrected PIL Images (same scene,
                   different headings).

        Returns:
            Prediction dict with fused country, coordinates, and per-view detail.
        """
        if len(crops) == 1:
            return self.predict(crops[0])

        tensors = torch.cat(
            [self.preprocess(crop) for crop in crops], dim=0
        ).to(self.device)
        outputs = self.model(tensors)

        country_logits = outputs["country_logits"]
        country_probs = torch.softmax(country_logits, dim=-1)

        avg_country_probs = country_probs.mean(dim=0)

        top5_conf, top5_idx = torch.topk(avg_country_probs, k=5)
        top5_conf = top5_conf.cpu().tolist()
        top5_idx = top5_idx.cpu().tolist()

        top5_countries = [
            {
                "country": self.idx_to_country.get(i, f"unknown_{i}"),
                "confidence": float(c),
            }
            for i, c in zip(top5_idx, top5_conf)
        ]

        top_view_conf, top_view_idx = country_probs.max(dim=-1)
        best_view = top_view_conf.argmax().item()

        region_probs = torch.softmax(outputs["region_logits"], dim=-1)
        top_region_probs, top_region_indices = torch.topk(
            region_probs[best_view], k=5, dim=-1
        )

        weighted_coords = (
            top_region_probs.unsqueeze(0)
            @ self.region_centroids[top_region_indices]
        )
        coords_rad = weighted_coords.squeeze(0)
        lat = float(torch.rad2deg(coords_rad[0]).item())
        lng = float(torch.rad2deg(coords_rad[1]).item())

        continent_probs = torch.softmax(outputs["continent_logits"], dim=-1)
        continent_idx = continent_probs[best_view].argmax(-1).item()
        continent = self.idx_to_continent.get(
            continent_idx, f"unknown_{continent_idx}"
        )

        per_view = [
            {
                "country": self.idx_to_country.get(
                    top_view_idx[i].item(), "unknown"
                ),
                "confidence": float(top_view_conf[i].item()),
            }
            for i in range(len(crops))
        ]

        return {
            "latitude": max(-90.0, min(90.0, lat)),
            "longitude": max(-180.0, min(180.0, lng)),
            "country": top5_countries[0]["country"],
            "country_confidence": top5_countries[0]["confidence"],
            "continent": continent,
            "top5_countries": top5_countries,
            "per_view": per_view,
            "best_view_index": best_view,
            "n_views": len(crops),
        }

    @torch.no_grad()
    def predict_panorama(
        self,
        pano: PanoramaInfo,
        base_heading: float = 0.0,
        n_crops: int = 5,
    ) -> Optional[dict]:
        """
        Full pipeline: stitch panorama tiles → perspective crops → multi-view
        inference.

        Args:
            pano: PanoramaInfo with collected tiles.
            base_heading: Initial camera heading in degrees.
            n_crops: Number of horizontal perspective crops to extract.

        Returns:
            Prediction dict or None if stitching fails.
        """
        equi = stitch_panorama(pano)
        if equi is None:
            return None

        crops = generate_view_crops(
            equi,
            base_heading=base_heading,
            n_horizontal=n_crops,
            fov_deg=90.0,
            out_size=(640, 640),
        )

        crop_images = [c[0] for c in crops]
        return self.predict_multi_view(crop_images)

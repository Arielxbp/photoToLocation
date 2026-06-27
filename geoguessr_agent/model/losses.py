from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def haversine_distance(
    pred_latlng: torch.Tensor, true_latlng: torch.Tensor
) -> torch.Tensor:
    """
    Compute Haversine distance (in km) between predicted and true (lat, lng) in radians.
    pred_latlng: (N, 2) — first column lat, second lng
    true_latlng: (N, 2)
    """
    R = 6371.0
    dlat = true_latlng[:, 0] - pred_latlng[:, 0]
    dlng = true_latlng[:, 1] - pred_latlng[:, 1]
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(true_latlng[:, 0])
        * torch.cos(pred_latlng[:, 0])
        * torch.sin(dlng / 2) ** 2
    )
    a = torch.clamp(a, 0.0, 1.0)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


def haversine_smoothed_labels(
    region_centroids: torch.Tensor,
    true_coords: torch.Tensor,
    true_region_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute PIGEON-style smoothed labels for region classification.

    For each sample, compute a soft label distribution over all regions,
    where probability mass is assigned based on Haversine distance to the true location.

    Args:
        region_centroids: (num_regions, 2) — centroid (lat_rad, lng_rad) for each region
        true_coords: (batch, 2) — true (lat_rad, lng_rad)
        true_region_idx: (batch,) — index of the true region
        temperature: tau parameter for softening

    Returns:
        smoothed_labels: (batch, num_regions) soft label distribution
    """
    batch = true_coords.shape[0]
    num_regions = region_centroids.shape[0]

    smoothed = []
    for i in range(batch):
        centroid_i = region_centroids[true_region_idx[i]]
        d_i = haversine_distance(
            region_centroids, centroid_i.unsqueeze(0).expand(num_regions, 2)
        )
        logits = -d_i / temperature
        logits = logits - logits.max()
        y_smooth = torch.exp(logits)
        y_smooth = y_smooth / (y_smooth.sum() + 1e-12)
        smoothed.append(y_smooth)
    return torch.stack(smoothed)


class HaversineSmoothCrossEntropy(nn.Module):
    """
    PIGEON-style smoothed cross-entropy loss for region classification.
    Combines haversine distance-based label smoothing with cross-entropy.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        region_centroids: torch.Tensor,
        true_coords: torch.Tensor,
        true_region_idx: torch.Tensor,
    ) -> torch.Tensor:
        smoothed = haversine_smoothed_labels(
            region_centroids, true_coords, true_region_idx, self.temperature
        )
        log_probs = F.log_softmax(logits, dim=-1)
        return -(smoothed * log_probs).sum(dim=-1).mean()


class HierarchicalLoss(nn.Module):
    """
    Combined loss for country + region + continent prediction.

    Supports:
        - label smoothing for country classification
        - MixUp: accepts optional lam + shuffled indices for mixed-target loss
    """

    def __init__(
        self,
        region_centroids: torch.Tensor,
        loss_country_weight: float = 0.6,
        loss_region_weight: float = 0.3,
        loss_continent_weight: float = 0.1,
        haversine_temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("region_centroids", region_centroids)
        self.w_country = loss_country_weight
        self.w_region = loss_region_weight
        self.w_continent = loss_continent_weight
        self.label_smoothing = label_smoothing
        self.country_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.continent_loss = nn.CrossEntropyLoss()
        self.region_loss = HaversineSmoothCrossEntropy(temperature=haversine_temperature)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        l_region = self.region_loss(
            outputs["region_logits"],
            self.region_centroids,
            targets["true_coords"],
            targets["region_idx"],
        )

        lam = targets.get("mixup_lam", 1.0)
        if lam >= 1.0:
            l_country = self.country_loss(outputs["country_logits"], targets["country_idx"])
            l_continent = self.continent_loss(outputs["continent_logits"], targets["continent_idx"])
        else:
            l_country = lam * self.country_loss(
                outputs["country_logits"], targets["country_idx"]
            ) + (1 - lam) * self.country_loss(
                outputs["country_logits"], targets["country_idx_shuffled"]
            )
            l_continent = lam * self.continent_loss(
                outputs["continent_logits"], targets["continent_idx"]
            ) + (1 - lam) * self.continent_loss(
                outputs["continent_logits"], targets["continent_idx_shuffled"]
            )

        total = (
            self.w_country * l_country
            + self.w_region * l_region
            + self.w_continent * l_continent
        )

        return {
            "total": total,
            "country": l_country,
            "region": l_region,
            "continent": l_continent,
        }

    def set_label_smoothing(self, value: float) -> None:
        self.label_smoothing = value
        self.country_loss = nn.CrossEntropyLoss(label_smoothing=value)


class DPOGeoLoss(nn.Module):
    """
    Direct Preference Optimization loss adapted for geolocation.

    Given two guesses for the same image (better and worse, as measured by
    Haversine distance to ground truth), the model should prefer the better guess.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        ref_region_logits: torch.Tensor,
        model_region_logits_better: torch.Tensor,
        model_region_logits_worse: torch.Tensor,
        region_centroids: torch.Tensor,
        true_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_region_logits: logits from reference (frozen) model
            model_region_logits_better: logits from current model for better guess
            model_region_logits_worse: logits from current model for worse guess
            region_centroids: (num_regions, 2) centroids
            true_coords: (batch, 2) true locations
        """
        ref_probs = torch.softmax(ref_region_logits, dim=-1)
        model_probs_better = torch.softmax(model_region_logits_better, dim=-1)
        model_probs_worse = torch.softmax(model_region_logits_worse, dim=-1)

        ref_coords_better = ref_probs @ region_centroids
        ref_coords_worse = ref_probs @ region_centroids
        model_coords_better = model_probs_better @ region_centroids
        model_coords_worse = model_probs_worse @ region_centroids

        d_ref_better = haversine_distance(ref_coords_better, true_coords)
        d_ref_worse = haversine_distance(ref_coords_worse, true_coords)
        d_model_better = haversine_distance(model_coords_better, true_coords)
        d_model_worse = haversine_distance(model_coords_worse, true_coords)

        r_ref_better = -d_ref_better
        r_ref_worse = -d_ref_worse
        r_model_better = -d_model_better
        r_model_worse = -d_model_worse

        advantage_better = r_model_better - r_ref_better
        advantage_worse = r_model_worse - r_ref_worse

        log_ratio = self.beta * (advantage_better - advantage_worse)
        loss = -torch.log(torch.sigmoid(log_ratio)).mean()

        return loss


def latlng_to_rad(lat_deg: torch.Tensor, lng_deg: torch.Tensor) -> torch.Tensor:
    """Convert (lat_deg, lng_deg) to radians."""
    return torch.stack([
        torch.deg2rad(lat_deg),
        torch.deg2rad(lng_deg),
    ], dim=-1)


def rad_to_latlng(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians back to (lat_deg, lng_deg)."""
    return torch.stack([
        torch.rad2deg(rad[:, 0]),
        torch.rad2deg(rad[:, 1]),
    ], dim=-1)

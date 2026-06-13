from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..model.losses import haversine_distance, latlng_to_rad


def compute_accuracy(
    output: torch.Tensor, target: torch.Tensor, top_k: tuple[int, ...] = (1, 3, 5)
) -> dict[str, float]:
    with torch.no_grad():
        maxk = max(top_k)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = {}
        for k in top_k:
            results[f"top{k}"] = correct[:k].float().sum(0).mean().item() * 100.0
    return results


def compute_distance_metrics(
    pred_coords: torch.Tensor, true_coords: torch.Tensor
) -> dict[str, float]:
    """Compute mean/median haversine distance in km."""
    d = haversine_distance(
        latlng_to_rad(pred_coords[:, 0], pred_coords[:, 1]),
        latlng_to_rad(true_coords[:, 0], true_coords[:, 1]),
    )
    return {
        "mean_distance_km": d.mean().item(),
        "median_distance_km": d.median().item(),
    }


def compute_per_country_accuracy(
    country_preds: list[int],
    country_targets: list[int],
    country_names: list[str],
) -> dict[str, float]:
    """Compute accuracy per country."""
    from collections import defaultdict

    correct = defaultdict(int)
    total = defaultdict(int)
    for p, t, n in zip(country_preds, country_targets, country_names):
        total[n] += 1
        if p == t:
            correct[n] += 1

    return {n: correct[n] / total[n] * 100.0 for n in total}

#!/usr/bin/env python3
"""
GeoGuessr Agent — Infer geolocation from a single Street View photo.

See scripts/ for data preparation and training:
  scripts/scrape_plonkit.py   Build the Plonkit clue knowledge base
  scripts/build_index.py      Build country/region indices from Kaggle dataset
  scripts/train.py            Train the geolocation model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geoguessr_agent.config import load_config
from geoguessr_agent.model.geolocator import GeoLocator


def main():
    parser = argparse.ArgumentParser(
        description="Infer geolocation from a single Street View photo"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--indices", default="data/indices.json")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.indices) as f:
        indices = json.load(f)
    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}
    idx_to_continent = {int(k): v for k, v in indices["idx_to_continent"].items()}

    region_centroids = torch.load(
        str(Path(args.indices).with_suffix(".centroids.pt")),
        map_location="cpu", weights_only=False,
    )
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]

    model = GeoLocator(
        num_countries=indices["num_countries"],
        num_regions=indices["num_regions"],
        num_continents=indices["num_continents"],
        pretrained=False,
        #backbone_name=config.model.backbone,
    )
    model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"]
    )
    model.to(config.device)
    model.eval()

    from geoguessr_agent.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        model=model,
        region_centroids=region_centroids,
        country_index=indices["country_index"],
        idx_to_country=idx_to_country,
        idx_to_continent=idx_to_continent,
        device=config.device,
        use_tta=config.training.use_tta,
    )

    result = pipeline.predict(args.image)
    print(f"\nImage: {args.image}")
    print(f"  Location: ({result['latitude']:.4f}, {result['longitude']:.4f})")
    print(f"  Country:  {result['country']} (confidence: {result['country_confidence']:.2%})")
    print(f"  Continent: {result['continent']}")
    print(f"  Top-5:")
    for i, entry in enumerate(result["top5_countries"]):
        print(f"    {i+1}. {entry['country']:30s} {entry['confidence']:.2%}")


if __name__ == "__main__":
    main()

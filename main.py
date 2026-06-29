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
    parser.add_argument(
        "--extract-features", action="store_true",
        help="Extract clue features with CLIP and show per-category predictions",
    )
    parser.add_argument(
        "--fuse-features", action="store_true",
        help="Use extracted clue features for fused prediction (requires model with fusion)",
    )
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
        clue_feature_dim=len(indices.get("clue_feature_indices", {})) or None,
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
    print("  Top-5:")
    for i, entry in enumerate(result["top5_countries"]):
        print(f"    {i+1}. {entry['country']:30s} {entry['confidence']:.2%}")

    if args.extract_features or args.fuse_features:
        from geoguessr_agent.features import CATEGORY_PROMPTS, get_feature_dim
        from geoguessr_agent.features.extractor import ClueFeatureExtractor

        clip_device = config.features.device or config.device
        extractor = ClueFeatureExtractor(
            model_name=config.features.clip_model,
            device=clip_device,
            use_streetclip=config.features.use_streetclip,
        )

        extracted = extractor.extract_with_labels(args.image)
        clue_vector = extracted["vector"]

        print(f"\n  Clue features ({get_feature_dim()} dims):")
        for cat in CATEGORY_PROMPTS:
            tops = extracted["labels"].get(cat.key, [])
            if tops:
                items = ", ".join(
                    f"{label}={conf:.2f}" for label, conf in tops
                )
                print(f"    {cat.label:20s} {items}")

        if args.fuse_features and model.has_fusion:
            import numpy as np
            from PIL import Image

            from geoguessr_agent.constants import IMAGE_NET_MEAN, IMAGE_NET_STD

            img = Image.open(args.image).convert("RGB").resize((320, 180), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mean = IMAGE_NET_MEAN.reshape(3, 1, 1)
            std = IMAGE_NET_STD.reshape(3, 1, 1)
            tensor = torch.from_numpy((arr - mean) / std).float().unsqueeze(0).to(config.device)
            clue_tensor = torch.from_numpy(clue_vector).float().unsqueeze(0).to(config.device)

            outputs = model(tensor, clue_tensor)
            country_probs = torch.softmax(outputs["country_logits"], dim=-1)
            top5_conf, top5_idx = torch.topk(country_probs, k=5, dim=-1)

            print("\n  Fused prediction:")
            print("    Top-5:")
            for i in range(5):
                idx = top5_idx[0, i].item()
                conf = top5_conf[0, i].item()
                name = idx_to_country.get(idx, f"unknown_{idx}")
                print(f"      {i+1}. {name:30s} {conf:.2%}")


if __name__ == "__main__":
    main()

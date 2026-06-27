#!/usr/bin/env python3
"""Train the geolocation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from geoguessr_agent.config import load_config
from geoguessr_agent.data.loader import create_dataloaders
from geoguessr_agent.data.mapper import build_balanced_split
from geoguessr_agent.model.geolocator import GeoLocator
from geoguessr_agent.model.losses import HierarchicalLoss
from geoguessr_agent.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the geolocation model")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--data-dir", default="data/kaggle/data")
    parser.add_argument("--indices", default="data/indices.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--resume", default=None, help="Resume from a training_state.pt checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.indices) as f:
        indices = json.load(f)
    country_index = indices["country_index"]
    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}
    continent_index = indices["continent_index"]
    idx_to_continent = {int(k): v for k, v in indices["idx_to_continent"].items()}
    country_to_continent = {
        k: int(v) for k, v in indices["country_to_continent"].items()
    }

    region_centroids = torch.load(
        str(Path(args.indices).with_suffix(".centroids.pt")),
        map_location="cpu", weights_only=False,
    )
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]

    s2_level = indices.get("s2_level", 6)
    cell_to_idx_path = Path(args.indices).with_suffix(".cell_to_idx.pt")
    if cell_to_idx_path.exists():
        from geoguessr_agent.geoutils import load_cell_to_index
        cell_to_idx = load_cell_to_index(cell_to_idx_path)
    else:
        cell_to_idx = {}

    num_countries = len(country_index)
    num_regions = region_centroids.shape[0]
    num_continents = len(continent_index)

    print(f"Countries: {num_countries}, Regions: {num_regions}, Continents: {num_continents}")

    region_index = {"cell_" + str(i): i for i in range(num_regions)}

    file_list = None
    if args.balance:
        file_list = build_balanced_split(
            args.data_dir,
            min_per_country=config.data.min_images_per_country,
            seed=config.training.seed,
        )
        print(f"Balanced file list: {len(file_list)} images")

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        country_index=country_index,
        region_index=region_index,
        continent_index=continent_index,
        country_to_continent=country_to_continent,
        image_size=config.model.image_size,
        batch_size=config.training.batch_size,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
        file_list=file_list,
        balance=args.balance,
        filter_low_variance=config.data.filter_low_variance,
        variance_threshold=config.data.variance_threshold,
        laplacian_threshold=config.data.laplacian_threshold,
        s2_level=s2_level,
        cell_to_idx=cell_to_idx,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = GeoLocator(
        num_countries=num_countries,
        num_regions=num_regions,
        num_continents=num_continents,
        pretrained=config.model.pretrained,
        freeze_backbone=False,
        dropout=config.model.dropout,
        backbone_name=config.model.backbone,
    )

    loss_fn = HierarchicalLoss(
        region_centroids=region_centroids,
        loss_country_weight=config.training.loss_country_weight,
        loss_region_weight=config.training.loss_region_weight,
        loss_continent_weight=config.training.loss_continent_weight,
        haversine_temperature=config.training.haversine_temperature,
        label_smoothing=config.training.label_smoothing,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=config.device,
    )

    checkpoint_dir = args.checkpoint_dir or config.data.checkpoint_dir
    trainer.fit(checkpoint_dir, resume_from=args.resume)

    model.save(f"{checkpoint_dir}/geolocator_final.pth")
    print(f"Training complete. Model saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()

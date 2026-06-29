#!/usr/bin/env python3
"""Fine-tune the geolocation model with DPO on the replay buffer (Phase 3).

Loads a replay buffer collected by ``scripts/selfplay.py`` and runs offline
Direct Preference Optimization, preferring the better (closer) guess over the
worse one for each image. The frozen reference model is initialised from the
same checkpoint as the trainable model.

Optionally evaluates on a Kaggle holdout set before and after DPO to detect
model drift, and versions checkpoints per iteration.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from geoguessr_agent.config import load_config
from geoguessr_agent.data.loader import create_dataloaders
from geoguessr_agent.model.geolocator import GeoLocator
from geoguessr_agent.training.dpo_trainer import DPOTrainer


def _build_model(indices: dict, model_path: str) -> GeoLocator:
    model = GeoLocator(
        num_countries=indices["num_countries"],
        num_regions=indices["num_regions"],
        num_continents=indices["num_continents"],
        pretrained=False,
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
    )
    return model


def _make_val_loader(indices: dict, data_dir: str, config) -> tuple:
    country_index = indices["country_index"]
    continent_index = indices["continent_index"]
    country_to_continent = {
        k: int(v) for k, v in indices["country_to_continent"].items()
    }
    num_regions = indices["num_regions"]
    region_index = {"cell_" + str(i): i for i in range(num_regions)}

    s2_level = indices.get("s2_level", 6)
    cell_to_idx_path = Path(data_dir).parent / "indices.json"
    cell_to_idx_path = Path(str(cell_to_idx_path).replace(".json", ".cell_to_idx.pt"))
    cell_to_idx = {}
    if cell_to_idx_path.exists():
        from geoguessr_agent.geoutils import load_cell_to_index
        cell_to_idx = load_cell_to_index(cell_to_idx_path)

    _, val_loader = create_dataloaders(
        data_dir=data_dir,
        country_index=country_index,
        region_index=region_index,
        continent_index=continent_index,
        country_to_continent=country_to_continent,
        image_size=config.model.image_size,
        batch_size=config.training.batch_size,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
        balance=False,
        filter_low_variance=config.data.filter_low_variance,
        variance_threshold=config.data.variance_threshold,
        laplacian_threshold=config.data.laplacian_threshold,
        s2_level=s2_level,
        cell_to_idx=cell_to_idx,
    )
    return val_loader


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tune on the replay buffer")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--indices", default="data/indices.json")
    parser.add_argument("--buffer", default="data/replay_buffer.pkl")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number for checkpoint versioning")
    parser.add_argument(
        "--val-data", default=None,
        help="Path to Kaggle data dir for holdout validation (e.g. data/kaggle/data)",
    )
    parser.add_argument("--val-indices", default=None,
                        help="Path to indices JSON for validation (defaults to --indices)")
    parser.add_argument("--no-val", action="store_true",
                        help="Disable validation (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)

    indices_path = args.val_indices or args.indices
    with open(indices_path) as f:
        indices = json.load(f)

    model_indices_path = args.indices
    with open(model_indices_path) as f:
        model_indices = json.load(f)

    region_centroids = torch.load(
        str(Path(model_indices_path).with_suffix(".centroids.pt")),
        map_location="cpu", weights_only=False,
    )
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]

    cell_to_idx_path = Path(str(Path(model_indices_path).with_suffix(".centroids.pt")).replace(
        ".centroids.pt", ".cell_to_idx.pt"
    ))
    cell_to_idx = {}
    if cell_to_idx_path.exists():
        from geoguessr_agent.geoutils import load_cell_to_index
        cell_to_idx = load_cell_to_index(cell_to_idx_path)

    s2_level = model_indices.get("s2_level", 6)

    model = _build_model(model_indices, args.model)
    ref_model = _build_model(model_indices, args.model)

    buffer_path = Path(args.buffer)
    if not buffer_path.exists():
        print(f"Replay buffer not found: {buffer_path}")
        print("Run scripts/selfplay.py first to collect data.")
        sys.exit(1)

    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    print(f"Replay buffer: {len(buffer)} entries")

    val_loader = None
    validation_enabled = config.dpo.validation_enabled and not args.no_val
    if validation_enabled and args.val_data:
        print(f"Loading validation holdout from {args.val_data} ...")
        val_loader = _make_val_loader(indices, args.val_data, config)
        print(f"  Validation batches: {len(val_loader)}")
    elif validation_enabled and not args.val_data:
        print("  [INFO] No --val-data provided; skipping holdout validation.")
        print("  Pass --val-data data/kaggle/data to enable drift detection.")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        region_centroids=region_centroids,
        beta=config.dpo.beta,
        learning_rate=config.dpo.learning_rate,
        device=config.device,
    )

    result = trainer.train_on_buffer(
        buffer=buffer,
        batch_size=config.dpo.batch_size,
        epochs=config.dpo.dpo_epochs,
        checkpoint_dir=args.checkpoint_dir or config.data.checkpoint_dir,
        iteration=args.iteration,
        val_loader=val_loader,
        max_degradation_pct=config.dpo.max_degradation_pct,
        cell_to_idx=cell_to_idx,
        s2_level=s2_level,
        country_index=indices.get("country_index"),
        dpo_loss_weight=config.dpo.dpo_loss_weight,
        sft_loss_weight=config.dpo.sft_loss_weight,
    )

    print(f"DPO fine-tuning complete. Loss: {result['loss']:.4f}")
    if result.get("degraded"):
        print("  [WARN] Holdout accuracy degraded — consider rolling back to previous iteration.")
    if result.get("checkpoint_path"):
        print(f"  Checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()

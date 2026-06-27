#!/usr/bin/env python3
"""Fine-tune the geolocation model with DPO on the replay buffer (Phase 3).

Loads a replay buffer collected by ``scripts/selfplay.py`` and runs offline
Direct Preference Optimization, preferring the better (closer) guess over the
worse one for each image. The frozen reference model is initialised from the
same checkpoint as the trainable model.
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


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tune on the replay buffer")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--indices", default="data/indices.json")
    parser.add_argument("--buffer", default="data/replay_buffer.pkl")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.indices) as f:
        indices = json.load(f)

    region_centroids = torch.load(
        str(Path(args.indices).with_suffix(".centroids.pt")),
        map_location="cpu", weights_only=False,
    )
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]

    model = _build_model(indices, args.model)
    ref_model = _build_model(indices, args.model)

    buffer_path = Path(args.buffer)
    if not buffer_path.exists():
        print(f"Replay buffer not found: {buffer_path}")
        print("Run scripts/selfplay.py first to collect data.")
        sys.exit(1)

    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    print(f"Replay buffer: {len(buffer)} entries")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        region_centroids=region_centroids,
        beta=config.dpo.beta,
        learning_rate=config.dpo.learning_rate,
        device=config.device,
    )

    loss = trainer.train_on_buffer(
        buffer=buffer,
        batch_size=config.dpo.batch_size,
        epochs=config.dpo.dpo_epochs,
        checkpoint_dir=args.checkpoint_dir or config.data.checkpoint_dir,
    )

    print(f"DPO fine-tuning complete. Final loss: {loss:.4f}")


if __name__ == "__main__":
    main()

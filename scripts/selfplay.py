#!/usr/bin/env python3
"""Run the self-play loop (Phase 2).

Drives a live OpenGuessr browser session: for each round it screenshots the
Street View, runs the geolocation model, clicks the minimap and reads the
score. Every round is stored in a replay buffer which is persisted to disk so
it can later be consumed by ``scripts/dpo_finetune.py``.

Requires Playwright with a Chromium browser:
    pip install -e '.[selfplay]'
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from geoguessr_agent.config import load_config
from geoguessr_agent.geoutils import get_capital_coordinates, get_country_centroids
from geoguessr_agent.model.geolocator import GeoLocator
from geoguessr_agent.plonkit.kb import ClueKnowledgeBase
from geoguessr_agent.self_play.loop import SelfPlayLoop


def main():
    parser = argparse.ArgumentParser(description="Run the self-play loop")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--indices", default="data/indices.json")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--kb-dir", default="data/plonkit_cache")
    parser.add_argument("--buffer", default="data/replay_buffer.pkl",
                        help="Where to persist the collected replay buffer")
    parser.add_argument("--heatmaps", action="store_true", default=False)
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Exploration epsilon (overrides config)")
    parser.add_argument("--use-capitals", action="store_true", default=False,
                        help="Click minimap on country capitals instead of geographic centres")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epsilon is not None:
        config.dpo.exploration_epsilon = args.epsilon

    with open(args.indices) as f:
        indices = json.load(f)
    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}

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
    )
    model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"]
    )

    kb = None
    if args.kb_dir and Path(args.kb_dir).exists():
        kb = ClueKnowledgeBase(args.kb_dir)

    if args.use_capitals:
        country_centroids = get_capital_coordinates(indices["country_index"])
    else:
        country_centroids = get_country_centroids(indices["country_index"])

    loop = SelfPlayLoop(
        config=config,
        model=model,
        region_centroids=region_centroids,
        country_index=indices["country_index"],
        idx_to_country=idx_to_country,
        country_centroids=country_centroids,
        kb=kb,
        enable_heatmaps=args.heatmaps,
    )

    stats = asyncio.run(loop.run_session(num_rounds=args.rounds))

    buffer_path = Path(args.buffer)
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(buffer_path, "wb") as f:
        pickle.dump(loop.buffer, f)
    print(f"Saved replay buffer ({len(loop.buffer)} entries) → {buffer_path}")
    print(f"Session stats: {stats}")


if __name__ == "__main__":
    main()

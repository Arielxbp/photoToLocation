#!/usr/bin/env python3
"""
GeoGuessr Agent — Autonomous AI agent for playing GeoGuessr via OpenGuessr.

Commands:
  scrape-plonkit    Build the Plonkit clue knowledge base
  build-index       Build country/region indices from Kaggle dataset
  train             Train the geolocation model (Phase 1)
  infer             Run inference on a single image
  self-play         Run the self-play loop (Phase 2)
  dpo-finetune      Fine-tune with DPO on the replay buffer (Phase 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from geoguessr_agent.config import Config, load_config
from geoguessr_agent.data.loader import GeoguessrDataset, create_dataloaders
from geoguessr_agent.data.mapper import CountryMapper, build_balanced_split
from geoguessr_agent.geoutils import (
    generate_s2_region_centroids,
    get_country_centroids,
    save_region_centroids,
)
from geoguessr_agent.model.geolocator import GeoLocator
from geoguessr_agent.model.losses import HierarchicalLoss
from geoguessr_agent.plonkit.kb import ClueKnowledgeBase
from geoguessr_agent.plonkit.scraper import build_clue_kb
from geoguessr_agent.training.dpo_trainer import DPOTrainer
from geoguessr_agent.training.trainer import Trainer


def cmd_scrape_plonkit(args):
    """Build the Plonkit clue knowledge base."""
    cache_dir = args.cache_dir or "data/plonkit_cache"
    print(f"Building Plonkit KB → {cache_dir}")
    kb = build_clue_kb(cache_dir=cache_dir)
    print(f"Collected {len(kb)} countries")
    for code in sorted(kb)[:5]:
        name = kb[code].get("name", code)
        clues = [k for k in kb[code] if k not in ("name", "slug")]
        print(f"  {code} ({name}): {len(clues)} clue categories")


def cmd_build_index(args):
    """Build country/region indices from the Kaggle dataset."""
    data_dir = args.data_dir or "data/kaggle"
    output = args.output or "data/indices.json"

    print(f"Building indices from {data_dir}...")
    mapper = CountryMapper()
    country_index = mapper.build_from_files(data_dir, min_samples=args.min_samples)

    print(f"Found {len(country_index)} countries with ≥{args.min_samples} samples")

    region_centroids = generate_s2_region_centroids(level=args.s2_level)
    print(f"Generated {region_centroids.shape[0]} S2 L{args.s2_level} region centroids")

    continent_index = mapper.continent_to_idx

    indices = {
        "country_index": country_index,
        "idx_to_country": {str(k): v for k, v in mapper.idx_to_country.items()},
        "continent_index": continent_index,
        "idx_to_continent": {str(k): v for k, v in mapper.idx_to_continent.items()},
        "country_to_continent": {
            str(k): v for k, v in mapper.country_to_continent.items()
        },
        "num_countries": len(country_index),
        "num_regions": region_centroids.shape[0],
        "num_continents": len(continent_index),
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(indices, f, indent=2)

    centroids_path = Path(output).with_suffix(".centroids.pt")
    save_region_centroids(region_centroids, centroids_path)

    print(f"Saved indices → {output}")
    print(f"Saved centroids → {centroids_path}")


def cmd_train(args):
    """Train the geolocation model (Phase 1 supervised)."""
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

    region_centroids = torch.load(str(Path(args.indices).with_suffix(".centroids.pt")),
                                 map_location="cpu", weights_only=False)
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]

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
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = GeoLocator(
        num_countries=num_countries,
        num_regions=num_regions,
        num_continents=num_continents,
        pretrained=config.model.pretrained,
        freeze_backbone=False,
        dropout=config.model.dropout,
    )

    loss_fn = HierarchicalLoss(
        region_centroids=region_centroids,
        loss_country_weight=config.training.loss_country_weight,
        loss_region_weight=config.training.loss_region_weight,
        loss_continent_weight=config.training.loss_continent_weight,
        haversine_temperature=config.training.haversine_temperature,
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
    trainer.fit(checkpoint_dir)

    model.save(f"{checkpoint_dir}/geolocator_final.pth")
    print(f"Training complete. Model saved to {checkpoint_dir}/")


def cmd_infer(args):
    """Run inference on a single image."""
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
    )

    result = pipeline.predict(args.image)
    print(f"\nImage: {args.image}")
    print(f"  Location: ({result['latitude']:.4f}, {result['longitude']:.4f})")
    print(f"  Country:  {result['country']} (confidence: {result['country_confidence']:.2%})")
    print(f"  Continent: {result['continent']}")
    print(f"  Top-5:")
    for i, entry in enumerate(result["top5_countries"]):
        print(f"    {i+1}. {entry['country']:30s} {entry['confidence']:.2%}")


def cmd_self_play(args):
    """Run the self-play loop."""
    config = load_config(args.config)

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

    from geoguessr_agent.geoutils import get_country_centroids
    from geoguessr_agent.self_play.loop import SelfPlayLoop

    country_centroids = get_country_centroids(indices["country_index"])

    loop = SelfPlayLoop(
        config=config,
        model=model,
        region_centroids=region_centroids,
        country_index=indices["country_index"],
        idx_to_country=idx_to_country,
        country_centroids=country_centroids,
        kb=kb,
    )

    asyncio_runner(loop.run_session, num_rounds=args.rounds)


def cmd_dpo_finetune(args):
    """Fine-tune with DPO on the replay buffer."""
    import pickle

    config = load_config(args.config)

    with open(args.indices) as f:
        indices = json.load(f)

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

    ref_model = GeoLocator(
        num_countries=indices["num_countries"],
        num_regions=indices["num_regions"],
        num_continents=indices["num_continents"],
        pretrained=False,
    )
    ref_model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"]
    )

    buffer_path = args.buffer or "data/replay_buffer.pkl"
    if not Path(buffer_path).exists():
        print(f"Replay buffer not found: {buffer_path}")
        print("Run self-play first to collect data.")
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


def asyncio_runner(coro_func, **kwargs):
    """Helper to run async functions from sync code."""
    import asyncio
    asyncio.run(coro_func(**kwargs))


def main():
    parser = argparse.ArgumentParser(
        description="GeoGuessr Agent — Autonomous AI geolocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to YAML config file"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_scrape = sub.add_parser("scrape-plonkit", help="Build Plonkit clue KB")
    p_scrape.add_argument("--cache-dir", default="data/plonkit_cache")

    p_build = sub.add_parser("build-index", help="Build country/region indices")
    p_build.add_argument("--data-dir", default="data/kaggle")
    p_build.add_argument("--output", default="data/indices.json")
    p_build.add_argument("--min-samples", type=int, default=50)
    p_build.add_argument("--s2-level", type=int, default=6)

    p_train = sub.add_parser("train", help="Train geolocation model")
    p_train.add_argument("--data-dir", default="data/kaggle")
    p_train.add_argument("--indices", default="data/indices.json")
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.add_argument("--balance", action="store_true", default=True)

    p_infer = sub.add_parser("infer", help="Run inference on an image")
    p_infer.add_argument("image", help="Path to image file")
    p_infer.add_argument("--model", default="checkpoints/best_model.pth")
    p_infer.add_argument("--indices", default="data/indices.json")

    p_play = sub.add_parser("self-play", help="Run self-play loop")
    p_play.add_argument("--model", default="checkpoints/best_model.pth")
    p_play.add_argument("--indices", default="data/indices.json")
    p_play.add_argument("--rounds", type=int, default=50)
    p_play.add_argument("--kb-dir", default="data/plonkit_cache")

    p_dpo = sub.add_parser("dpo-finetune", help="DPO fine-tune on replay buffer")
    p_dpo.add_argument("--model", default="checkpoints/best_model.pth")
    p_dpo.add_argument("--indices", default="data/indices.json")
    p_dpo.add_argument("--buffer", default="data/replay_buffer.pkl")
    p_dpo.add_argument("--checkpoint-dir", default="checkpoints")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    command_map = {
        "scrape-plonkit": cmd_scrape_plonkit,
        "build-index": cmd_build_index,
        "train": cmd_train,
        "infer": cmd_infer,
        "self-play": cmd_self_play,
        "dpo-finetune": cmd_dpo_finetune,
    }

    cmd_func = command_map.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

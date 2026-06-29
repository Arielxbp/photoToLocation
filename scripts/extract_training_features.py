#!/usr/bin/env python3
"""
Pre-extract CLIP clue features for all training images.

Runs the ClueFeatureExtractor on every image in the Kaggle dataset and saves
a mapping {image_path: numpy_array} to data/clue_features.pt.

Usage:
  python scripts/extract_training_features.py --data-dir data/kaggle/data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from geoguessr_agent.config import load_config
from geoguessr_agent.features import CATEGORY_PROMPTS, get_feature_dim
from geoguessr_agent.features.extractor import ClueFeatureExtractor


def discover_images(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    images = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith((".jpg", ".jpeg", ".png")):
                json_path = os.path.join(root, f"{Path(f).stem}.json")
                if os.path.exists(json_path):
                    images.append(os.path.join(root, f))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract CLIP clue features")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--data-dir", default="data/kaggle/data")
    parser.add_argument("--output", default="data/clue_features.pt")
    parser.add_argument("--model", default=None, help="CLIP model name (overrides config)")
    parser.add_argument("--use-streetclip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit to N images for testing")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = args.model or config.features.clip_model
    use_streetclip = args.use_streetclip or config.features.use_streetclip
    device = config.features.device or config.device or "cuda"

    images = discover_images(args.data_dir)
    if args.limit:
        images = images[: args.limit]

    dim = get_feature_dim()
    print(f"Images: {len(images)}")
    print(f"Feature dim: {dim} ({len(CATEGORY_PROMPTS)} categories)")
    print(f"CLIP model: {model_name}")

    extractor = ClueFeatureExtractor(
        model_name=model_name,
        device=device,
        use_streetclip=use_streetclip,
    )

    from PIL import Image

    features = {}
    for img_path in tqdm(images, desc="Extracting features"):
        try:
            img = Image.open(img_path).convert("RGB")
            vec = extractor.extract(img)
            features[img_path] = vec.astype(np.float32)
        except Exception as e:
            print(f"  [SKIP] {img_path}: {e}")
            continue

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    torch.save(
        {"features": features, "feature_dim": dim, "paths": sorted(features.keys())},
        str(output_path),
    )

    coverage = len(features) / len(images) * 100 if images else 0
    print(f"Saved {len(features)}/{len(images)} ({coverage:.0f}%) → {output_path}")


if __name__ == "__main__":
    main()

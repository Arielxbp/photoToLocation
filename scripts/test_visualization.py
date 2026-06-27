#!/usr/bin/env python3
"""Predict country from a Street View image with Grad-CAM visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from geoguessr_agent.constants import IMAGE_NET_MEAN, IMAGE_NET_STD
from geoguessr_agent.inference.gradcam import (
    GradCAM,
    draw_boxes,
    find_activation_boxes,
    get_target_layer,
    overlay_heatmap,
)
from geoguessr_agent.model.geolocator import GeoLocator


def preprocess_for_model(image: Image.Image, device: str) -> tuple[torch.Tensor, Image.Image]:
    img_resized = image.convert("RGB").resize((320, 180), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = IMAGE_NET_MEAN.reshape(3, 1, 1)
    std = IMAGE_NET_STD.reshape(3, 1, 1)
    tensor = torch.from_numpy((arr - mean) / std).float()
    return tensor.unsqueeze(0).to(device), img_resized


def main():
    parser = argparse.ArgumentParser(
        description="Predict country from a Street View image with Grad-CAM visualization"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--indices", default="data/indices.json",
                        help="Path to country/region indices JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for annotated image (default: <input>_annotated.jpg)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Heatmap threshold for bounding boxes (0-1)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Heatmap overlay opacity (0-1)")
    parser.add_argument("--max-boxes", type=int, default=5,
                        help="Maximum number of bounding boxes to draw")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    indices_path = Path(args.indices)
    if not indices_path.exists():
        print(f"Error: indices not found: {args.indices}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {args.model}")
        sys.exit(1)

    with open(indices_path) as f:
        indices = json.load(f)
    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}
    idx_to_continent = {int(k): v for k, v in indices["idx_to_continent"].items()}

    region_centroids = torch.load(
        str(indices_path.with_suffix(".centroids.pt")),
        map_location="cpu", weights_only=False,
    )
    if isinstance(region_centroids, dict):
        region_centroids = region_centroids["centroids"]
    region_centroids = region_centroids.to(args.device)

    print(f"Device: {args.device}")
    print(f"Model:  {args.model}")
    print(f"Image:  {args.image}")

    model = GeoLocator.load(str(model_path), device=args.device)
    model.eval()

    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    print(f"Original size: {original_size[0]}x{original_size[1]}")

    input_tensor, resized_image = preprocess_for_model(original_image, args.device)

    with torch.no_grad():
        outputs = model(input_tensor)

    country_logits = outputs["country_logits"]
    country_probs = torch.softmax(country_logits, dim=-1)
    top5_conf, top5_idx = torch.topk(country_probs[0], k=5)

    region_logits = outputs["region_logits"]
    region_probs = torch.softmax(region_logits, dim=-1)
    top_region_probs, top_region_indices = torch.topk(region_probs, k=5, dim=-1)

    weighted_coords = top_region_probs @ region_centroids[top_region_indices.squeeze(0)]
    coords_rad = weighted_coords.squeeze(0)
    lat = float(torch.rad2deg(coords_rad[0]).item())
    lng = float(torch.rad2deg(coords_rad[1]).item())
    lat = max(-90.0, min(90.0, lat))
    lng = max(-180.0, min(180.0, lng))

    continent_logits = outputs["continent_logits"]
    continent_probs = torch.softmax(continent_logits, dim=-1)
    continent_idx = continent_probs.argmax(-1).item()
    continent = idx_to_continent.get(continent_idx, f"unknown_{continent_idx}")

    top_country = idx_to_country.get(top5_idx[0].item(), "unknown")
    top_conf = float(top5_conf[0].item())

    print(f"\n{'='*50}")
    print(f"  Predicted Country:  {top_country}")
    print(f"  Confidence:         {top_conf:.2%}")
    print(f"  Continent:          {continent}")
    print(f"  Location:           ({lat:.4f}, {lng:.4f})")
    print(f"\n  Top-5 Countries:")
    for i, (idx, conf) in enumerate(zip(top5_idx.tolist(), top5_conf.tolist())):
        marker = " <--" if i == 0 else ""
        print(f"    {i+1}. {idx_to_country.get(idx, '?'):30s} {conf:.2%}{marker}")
    print(f"{'='*50}")

    print("\nComputing Grad-CAM heatmap...")
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    heatmap = gradcam.compute(input_tensor, target_class=top5_idx[0].item())
    gradcam.remove_hooks()

    heatmap_t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_t = F.interpolate(
        heatmap_t, size=(original_size[1], original_size[0]),
        mode="bilinear", align_corners=False,
    )
    heatmap_full = heatmap_t.squeeze().numpy()

    boxes = find_activation_boxes(
        heatmap_full, threshold=args.threshold, min_area_ratio=0.005,
    )

    blended = overlay_heatmap(original_image, heatmap_full, alpha=args.alpha)
    annotated = draw_boxes(blended, boxes, max_boxes=args.max_boxes)

    draw = ImageDraw.Draw(annotated)
    lines = [
        f"Country: {top_country} ({top_conf:.1%})",
        f"Continent: {continent}",
        f"Location: ({lat:.2f}, {lng:.2f})",
    ]
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 20), line, fill=(255, 255, 255))

    output_path = args.output or str(image_path.parent / f"{image_path.stem}_annotated.jpg")
    annotated.save(output_path, quality=95)
    print(f"\nAnnotated image saved to: {output_path}")

    box_count = min(len(boxes), args.max_boxes)
    print(f"Drew {box_count} bounding box(es) on high-activation regions")
    print(f"Threshold used: {args.threshold}")


if __name__ == "__main__":
    main()

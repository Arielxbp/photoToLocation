#!/usr/bin/env python3
"""
Test program: Given a Google Street View photo, predict the country using the
trained GeoLocator model and save an annotated copy showing what the model
focused on (Grad-CAM heatmap + bounding boxes on high-activation regions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import label as connected_components_label

from geoguessr_agent.model.geolocator import GeoLocator


def _get_target_layer(model: GeoLocator) -> torch.nn.Module:
    """Return the last convolutional layer of the EfficientNet backbone."""
    return model.backbone.features[-1]


class GradCAM:
    """Grad-CAM for model introspection on the country classification head."""

    def __init__(self, model: GeoLocator, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_module, _input, output):
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.enable_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W].
            target_class: Target class index. If None, uses the argmax of logits.

        Returns:
            Heatmap as a 2D numpy array (H, W) in [0, 1], same spatial size as input.
        """
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.detach().clone().requires_grad_(True)

        outputs = self.model(input_tensor)
        logits = outputs["country_logits"]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()

        score = logits[0, target_class]
        score.backward()

        activations = self.activations.detach()  # [1, C, h, w]
        gradients = self.gradients.detach()      # [1, C, h, w]

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (weights * activations).sum(dim=1)  # [1, h, w]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.squeeze(0).cpu().numpy()  # [h, w]

        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = self._resize_cam(cam, (input_h, input_w))

        return cam

    @staticmethod
    def _resize_cam(cam: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
        cam_t = F.interpolate(cam_t, size=target_shape, mode="bilinear", align_corners=False)
        return cam_t.squeeze().numpy()


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    """Convert a [0,1] heatmap to an RGB colour array (blue→green→red)."""
    h = heatmap
    r = np.clip(h * 2.0 - 1.0, 0, 1) if h.max() > 0.5 else np.zeros_like(h)
    g = np.clip(h * 2.0, 0, 1) - np.clip(h * 2.0 - 1.0, 0, 1)
    b = np.clip(1.0 - h * 2.0, 0, 1)
    return np.stack([r, g, b], axis=-1)  # [H, W, 3]


def apply_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay heatmap on PIL image with given alpha blending."""
    img_arr = np.array(image).astype(np.float32) / 255.0
    heatmap_rgb = heatmap_to_color(heatmap)
    blended = img_arr * (1 - alpha) + heatmap_rgb * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def find_activation_boxes(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_area_ratio: float = 0.01,
) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of high-activation regions in the heatmap.

    Args:
        heatmap: 2D numpy array [H, W] in [0, 1].
        threshold: Value above which pixels are considered "activated".
        min_area_ratio: Minimum box area relative to image area.

    Returns:
        List of (x1, y1, x2, y2) boxes in pixel coordinates.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    labeled, num_features = connected_components_label(binary)

    h, w = heatmap.shape
    min_area = int(h * w * min_area_ratio)

    boxes = []
    for i in range(1, num_features + 1):
        yy, xx = np.where(labeled == i)
        if len(yy) < min_area:
            continue
        y1, y2 = int(yy.min()), int(yy.max())
        x1, x2 = int(xx.min()), int(xx.max())
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def draw_boxes(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (255, 255, 0),
    width: int = 3,
    max_boxes: int = 5,
) -> Image.Image:
    """Draw bounding boxes on a PIL image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes[:max_boxes]):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def preprocess_for_model(image: Image.Image, device: str) -> tuple[torch.Tensor, Image.Image]:
    """
    Preprocess image for the model: resize to 320x180, normalise.
    Returns (tensor_for_model, resized_pil_image).
    """
    img_resized = image.convert("RGB").resize((320, 180), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
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

    # --- Load indices ---
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

    # --- Load model ---
    model = GeoLocator.load(str(model_path), device=args.device)
    model.eval()

    # --- Load and preprocess image ---
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    print(f"Original size: {original_size[0]}x{original_size[1]}")

    input_tensor, resized_image = preprocess_for_model(original_image, args.device)

    # --- Run inference ---
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
        marker = " ←" if i == 0 else ""
        print(f"    {i+1}. {idx_to_country.get(idx, '?'):30s} {conf:.2%}{marker}")
    print(f"{'='*50}")

    # --- Grad-CAM ---
    print("\nComputing Grad-CAM heatmap...")
    target_layer = _get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    heatmap = gradcam.compute(input_tensor, target_class=top5_idx[0].item())
    gradcam.remove_hooks()

    # --- Resize heatmap to original image size ---
    heatmap_t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_t = F.interpolate(
        heatmap_t, size=(original_size[1], original_size[0]),
        mode="bilinear", align_corners=False,
    )
    heatmap_full = heatmap_t.squeeze().numpy()

    # --- Find activation boxes ---
    boxes = find_activation_boxes(
        heatmap_full,
        threshold=args.threshold,
        min_area_ratio=0.005,
    )

    # --- Build annotated image ---
    blended = apply_heatmap(original_image, heatmap_full, alpha=args.alpha)
    annotated = draw_boxes(blended, boxes, max_boxes=args.max_boxes)

    # --- Add text overlay with prediction ---
    draw = ImageDraw.Draw(annotated)
    lines = [
        f"Country: {top_country} ({top_conf:.1%})",
        f"Continent: {continent}",
        f"Location: ({lat:.2f}, {lng:.2f})",
    ]
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 20), line, fill=(255, 255, 255))

    # --- Save ---
    output_path = args.output or str(image_path.parent / f"{image_path.stem}_annotated.jpg")
    annotated.save(output_path, quality=95)
    print(f"\nAnnotated image saved to: {output_path}")

    box_count = min(len(boxes), args.max_boxes)
    print(f"Drew {box_count} bounding box(es) on high-activation regions")
    print(f"Threshold used: {args.threshold}")


if __name__ == "__main__":
    main()

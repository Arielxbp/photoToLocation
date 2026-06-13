from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import label as connected_components_label

from ..model.geolocator import GeoLocator


def get_target_layer(model: GeoLocator) -> torch.nn.Module:
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

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.squeeze(0).cpu().numpy()

        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
        cam_t = F.interpolate(cam_t, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return cam_t.squeeze().numpy()


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    h = heatmap
    r = np.clip(h * 2.0 - 1.0, 0, 1) if h.max() > 0.5 else np.zeros_like(h)
    g = np.clip(h * 2.0, 0, 1) - np.clip(h * 2.0 - 1.0, 0, 1)
    b = np.clip(1.0 - h * 2.0, 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
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
    binary = (heatmap >= threshold).astype(np.uint8)
    labeled, num_features = connected_components_label(binary)
    h, w = heatmap.shape
    min_area = int(h * w * min_area_ratio)
    boxes = []
    for i in range(1, num_features + 1):
        yy, xx = np.where(labeled == i)
        if len(yy) < min_area:
            continue
        boxes.append((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def draw_boxes(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (255, 255, 0),
    width: int = 3,
    max_boxes: int = 5,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes[:max_boxes]:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def generate_annotated_heatmap(
    model: GeoLocator,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: int,
    country_name: str,
    confidence: float,
    continent: str,
    lat: float,
    lng: float,
    device: str = "cuda",
) -> Image.Image | None:
    """
    Generate an annotated image with Grad-CAM heatmap overlay, bounding
    boxes, and prediction text.  Returns None if scipy is unavailable.
    """
    try:
        import scipy  # noqa: F401
    except ImportError:
        return None

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.compute(input_tensor.to(device), target_class=target_class)
    gradcam.remove_hooks()

    orig_w, orig_h = original_image.size
    heatmap_t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_t = F.interpolate(
        heatmap_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False,
    )
    heatmap_full = heatmap_t.squeeze().numpy()

    blended = overlay_heatmap(original_image, heatmap_full, alpha=0.4)
    boxes = find_activation_boxes(heatmap_full, threshold=0.5, min_area_ratio=0.005)
    annotated = draw_boxes(blended, boxes, max_boxes=5)

    draw = ImageDraw.Draw(annotated)
    lines = [
        f"Country: {country_name} ({confidence:.1%})",
        f"Continent: {continent}",
        f"Location: ({lat:.2f}, {lng:.2f})",
    ]
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 20), line, fill=(255, 255, 255))

    return annotated

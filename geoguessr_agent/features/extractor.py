"""
CLIP-based feature extractor for street photo clues.

Uses a pretrained CLIP model (default: geolocal/StreetCLIP or
openai/clip-vit-base-patch32) to perform zero-shot classification
across Plonkit-aligned clue categories.

Text embeddings are precomputed and cached for speed. On each image:
  1. Encode image once with CLIP vision encoder
  2. For each category, compute cosine similarity against precomputed
     text embeddings for that category's prompts
  3. Softmax within category to get probability distribution
  4. Concatenate all category distributions → flat clue feature vector
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .categories import CATEGORY_PROMPTS, get_feature_dim

_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
_STREETCLIP_MODEL = "geolocal/StreetCLIP"


class _WarmupModel(torch.nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, **kwargs):
        return self.clip(**kwargs)


class ClueFeatureExtractor:

    def __init__(
        self,
        model_name: str = _DEFAULT_CLIP_MODEL,
        device: str = "cuda",
        use_streetclip: bool = False,
    ):
        self.device = device
        self.model_name = _STREETCLIP_MODEL if use_streetclip else model_name

        from transformers import CLIPModel, CLIPProcessor

        print(f"[ClueExtractor] Loading CLIP: {self.model_name} ...")
        clip = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = _WarmupModel(clip).to(device).eval()

        self.feature_dim = get_feature_dim()
        self.category_boundaries: list[tuple[int, int]] = []
        self._build_text_cache()

    def _build_text_cache(self) -> None:
        all_texts: list[str] = []
        boundaries = []
        offset = 0
        for cat in CATEGORY_PROMPTS:
            n = len(cat.prompts)
            boundaries.append((offset, offset + n))
            for _, prompt_text in cat.prompts:
                all_texts.append(prompt_text)
            offset += n

        self.category_boundaries = boundaries

        print(
            f"[ClueExtractor] Encoding {len(all_texts)} text prompts "
            f"across {len(CATEGORY_PROMPTS)} categories ..."
        )

        inputs = self.processor(
            text=all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_outputs = self.model.clip.text_model(**{
                k: v for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")
            })
            text_features = self.model.clip.text_projection(
                text_outputs[1]
            )
            self.text_features = F.normalize(text_features, dim=-1)

        print("[ClueExtractor] Text cache built.")

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract clue feature vector from a PIL image.
        Returns numpy array of shape (feature_dim,) — probability
        distributions concatenated across all categories.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_outputs = self.model.clip.vision_model(**{
            k: v for k, v in inputs.items()
            if k == "pixel_values"
        })
        image_features = self.model.clip.visual_projection(
            vision_outputs[1]
        )
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.text_features.T
        logits = logits.squeeze(0)

        category_probs = []
        for start, end in self.category_boundaries:
            cat_logits = logits[start:end]
            cat_probs = torch.softmax(cat_logits, dim=-1)
            category_probs.append(cat_probs.cpu().numpy())

        return np.concatenate(category_probs)

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        feature_vecs = [self.extract(img) for img in images]
        return np.stack(feature_vecs, axis=0)

    @torch.no_grad()
    def extract_with_labels(self, image: Image.Image) -> dict:
        """
        Extract clue features with human-readable per-category top labels.
        Returns dict with 'vector' (np.ndarray) and 'labels' (dict).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_outputs = self.model.clip.vision_model(**{
            k: v for k, v in inputs.items()
            if k == "pixel_values"
        })
        image_features = self.model.clip.visual_projection(
            vision_outputs[1]
        )
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.text_features.T
        logits = logits.squeeze(0)

        category_probs = []
        labels: dict[str, list[tuple[str, float]]] = {}
        for cat_idx, cat in enumerate(CATEGORY_PROMPTS):
            start, end = self.category_boundaries[cat_idx]
            cat_logits = logits[start:end]
            cat_probs = torch.softmax(cat_logits, dim=-1)
            category_probs.append(cat_probs.cpu().numpy())

            probs_np = cat_probs.cpu().numpy()
            sorted_idx = np.argsort(-probs_np)
            labels[cat.key] = [
                (cat.prompts[i][0], float(probs_np[i]))
                for i in sorted_idx[:3]
                if probs_np[i] > 0.1
            ]

        return {
            "vector": np.concatenate(category_probs),
            "labels": labels,
        }

    def to(self, device: str) -> "ClueFeatureExtractor":
        self.device = device
        self.model = self.model.to(device)
        self.text_features = self.text_features.to(device)
        return self

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

from geoguessr_agent.data.mapper import _normalize_country
from geoguessr_agent.geoutils import latlng_to_region_idx

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class GeoguessrDataset(Dataset):
    """Dataset for GeoGuessr street view images with classification labels."""

    def __init__(
        self,
        data_dir: str | Path,
        country_index: dict[str, int],
        region_index: dict[str, int],
        continent_index: dict[str, int],
        country_to_continent: dict[str, int],
        image_size: tuple[int, int] = (320, 180),
        file_list: Optional[list[str]] = None,
        filter_low_variance: bool = True,
        variance_threshold: float = 15.0,
        laplacian_threshold: float = 50.0,
        transform: Optional[Callable] = None,
        s2_level: int = 6,
        cell_to_idx: Optional[dict[int, int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.country_index = country_index
        self.region_index = region_index
        self.continent_index = continent_index
        self.country_to_continent = country_to_continent
        self.image_size = image_size
        self.filter_low_variance = filter_low_variance
        self.variance_threshold = variance_threshold
        self.laplacian_threshold = laplacian_threshold
        self.transform = transform
        self.s2_level = s2_level
        self.cell_to_idx = cell_to_idx or {}

        if file_list is not None:
            self.samples = file_list
        else:
            self.samples = self._discover_samples()

    def _discover_samples(self) -> list[str]:
        samples = []
        for root, _dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith((".jpg", ".jpeg", ".png")):
                    json_path = Path(root) / f"{Path(f).stem}.json"
                    if json_path.exists():
                        samples.append(os.path.join(root, f))
        return samples

    def _compute_image_stats(self, img: Image.Image) -> tuple[float, float]:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        max_std = max(arr[..., 0].std(), arr[..., 1].std(), arr[..., 2].std())

        gray = np.array(img.convert("L"), dtype=np.float32)
        from scipy import ndimage
        laplacian = ndimage.laplace(gray)
        lap_var = laplacian.var() if laplacian.size > 0 else 0.0

        return max_std, lap_var

    def _is_valid_image(self, img: Image.Image) -> bool:
        if not self.filter_low_variance:
            return True
        color_var, lap_var = self._compute_image_stats(img)
        if color_var < self.variance_threshold:
            return False
        if lap_var < self.laplacian_threshold:
            return False
        return True

    def _load_metadata(self, img_path: str) -> dict:
        json_path = str(Path(img_path).with_suffix(".json"))
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)
        parent_json = Path(img_path).parent / "metadata.json"
        if parent_json.exists():
            with open(parent_json) as f:
                all_meta = json.load(f)
            fname = Path(img_path).name
            return all_meta.get(fname, {})
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)

        if not self._is_valid_image(img):
            img = Image.new("RGB", self.image_size, (128, 128, 128))

        if self.transform:
            img = self.transform(img)
        else:
            from ..constants import IMAGE_NET_MEAN_T, IMAGE_NET_STD_T
            img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            mean = IMAGE_NET_MEAN_T.view(3, 1, 1)
            std = IMAGE_NET_STD_T.view(3, 1, 1)
            img = (img - mean) / std

        meta = self._load_metadata(img_path)
        country = _normalize_country(meta.get("country_name") or meta.get("country", "Unknown"))
        region = meta.get("region", "Unknown")
        continent = meta.get("continent", "Unknown")
        coords = meta.get("coordinates", [0.0, 0.0])
        lat = coords[0] if coords else float(meta.get("lat", 0.0))
        lng = coords[1] if len(coords) > 1 else float(meta.get("lng", 0.0))

        country_idx = self.country_index.get(country, -1)

        # Compute region_idx from (lat, lng) via S2 cells — the correct
        # mapping that the old code was missing (it did a string-dict lookup
        # that always returned 0, making the region head untrainable).
        if self.cell_to_idx and lat != 0.0 and lng != 0.0:
            region_idx = latlng_to_region_idx(lat, lng, self.s2_level, self.cell_to_idx)
        else:
            region_idx = self.region_index.get(region, 0)

        continent_idx = self.continent_index.get(
            continent, self.country_to_continent.get(country, 0)
        )

        return {
            "image": img,
            "country_idx": torch.tensor(country_idx, dtype=torch.long),
            "region_idx": torch.tensor(region_idx, dtype=torch.long),
            "continent_idx": torch.tensor(continent_idx, dtype=torch.long),
            "lat": torch.tensor(lat, dtype=torch.float32),
            "lng": torch.tensor(lng, dtype=torch.float32),
            "country_name": country,
            "path": img_path,
        }


def create_dataloaders(
    data_dir: str | Path,
    country_index: dict[str, int],
    region_index: dict[str, int],
    continent_index: dict[str, int],
    country_to_continent: dict[str, int],
    image_size: tuple[int, int] = (320, 180),
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    file_list: Optional[list[str]] = None,
    balance: bool = True,
    filter_low_variance: bool = True,
    variance_threshold: float = 15.0,
    laplacian_threshold: float = 50.0,
    s2_level: int = 6,
    cell_to_idx: Optional[dict[int, int]] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""

    full_dataset = GeoguessrDataset(
        data_dir=data_dir,
        country_index=country_index,
        region_index=region_index,
        continent_index=continent_index,
        country_to_continent=country_to_continent,
        image_size=image_size,
        file_list=file_list,
        filter_low_variance=filter_low_variance,
        variance_threshold=variance_threshold,
        laplacian_threshold=laplacian_threshold,
        s2_level=s2_level,
        cell_to_idx=cell_to_idx,
    )

    generator = torch.Generator().manual_seed(seed)
    total = len(full_dataset)
    val_count = max(1, int(total * val_split))
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    if balance:
        train_countries = [
            full_dataset.samples[i] for i in train_indices
        ]
        country_counts = {}
        for i in train_indices:
            sample = full_dataset.samples[i]
            meta_path = str(Path(sample).with_suffix(".json"))
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                c = _normalize_country(meta.get("country_name") or meta.get("country", "Unknown"))
            except Exception:
                c = "Unknown"
            country_counts[c] = country_counts.get(c, 0) + 1

        weights = []
        for i in train_indices:
            sample = full_dataset.samples[i]
            meta_path = str(Path(sample).with_suffix(".json"))
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                c = _normalize_country(meta.get("country_name") or meta.get("country", "Unknown"))
            except Exception:
                c = "Unknown"
            weights.append(1.0 / country_counts.get(c, 1))

        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_indices), replacement=True
        )
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

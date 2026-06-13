from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ReplayEntry:
    """Single round entry in the replay buffer."""

    def __init__(
        self,
        image: Image.Image,
        true_lat: float,
        true_lng: float,
        true_country: str,
        guess_lat: float,
        guess_lng: float,
        distance_km: float,
        round_id: Optional[str] = None,
    ):
        self.image = image
        self.true_lat = true_lat
        self.true_lng = true_lng
        self.true_country = true_country
        self.guess_lat = guess_lat
        self.guess_lng = guess_lng
        self.distance_km = distance_km
        self.round_id = round_id


class ReplayBufferDataset(Dataset):
    """PyTorch Dataset wrapping the replay buffer."""

    def __init__(
        self,
        entries: list[ReplayEntry],
        image_size: tuple[int, int] = (320, 180),
    ):
        self.entries = entries
        self.image_size = image_size
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.register_buffers = {"mean": mean, "std": std}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        img = entry.image.resize(self.image_size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        img_tensor = (img_tensor - self.register_buffers["mean"]) / self.register_buffers["std"]

        jitter = random.uniform(-0.5, 0.5)
        better_lat = entry.true_lat + jitter * 0.01
        better_lng = entry.true_lng + jitter * 0.01
        worse_lat = entry.guess_lat + jitter * 0.05
        worse_lng = entry.guess_lng + jitter * 0.05

        return {
            "image": img_tensor,
            "lat": torch.tensor(entry.true_lat, dtype=torch.float32),
            "lng": torch.tensor(entry.true_lng, dtype=torch.float32),
            "country": entry.true_country,
            "better_lat": torch.tensor(better_lat, dtype=torch.float32),
            "better_lng": torch.tensor(better_lng, dtype=torch.float32),
            "worse_lat": torch.tensor(worse_lat, dtype=torch.float32),
            "worse_lng": torch.tensor(worse_lng, dtype=torch.float32),
        }


class ReplayBuffer:
    """FIFO replay buffer for self-play experience."""

    def __init__(self, max_size: int = 10_000):
        self._buffer: deque[ReplayEntry] = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, entry: ReplayEntry) -> None:
        self._buffer.append(entry)

    def sample(self, n: int) -> list[ReplayEntry]:
        n = min(n, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        image_size: tuple[int, int] = (320, 180),
        num_workers: int = 2,
    ) -> DataLoader:
        dataset = ReplayBufferDataset(list(self._buffer), image_size=image_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_best_entries(self, n: int) -> list[ReplayEntry]:
        """Return the N entries with lowest distance (best guesses)."""
        sorted_entries = sorted(self._buffer, key=lambda e: e.distance_km)
        return sorted_entries[:n]

    def get_worst_entries(self, n: int) -> list[ReplayEntry]:
        """Return the N entries with highest distance (worst guesses)."""
        sorted_entries = sorted(self._buffer, key=lambda e: -e.distance_km)
        return sorted_entries[:n]

    def get_by_country(self, country: str) -> list[ReplayEntry]:
        return [e for e in self._buffer if e.true_country == country]

    @property
    def stats(self) -> dict:
        if not self._buffer:
            return {"count": 0}
        distances = [e.distance_km for e in self._buffer]
        countries = set(e.true_country for e in self._buffer)
        return {
            "count": len(self._buffer),
            "mean_distance_km": np.mean(distances),
            "median_distance_km": np.median(distances),
            "min_distance_km": np.min(distances),
            "max_distance_km": np.max(distances),
            "unique_countries": len(countries),
        }

    def __len__(self) -> int:
        return len(self._buffer)

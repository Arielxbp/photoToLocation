from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from ..model.geolocator import GeoLocator
from ..model.losses import DPOGeoLoss, latlng_to_rad
from ..self_play.buffer import ReplayBuffer


class DPOTrainer:
    """
    Offline Direct Preference Optimization trainer for geolocation.

    Fine-tunes the model by comparing better vs. worse guesses
    for the same images, as determined by Haversine distance to ground truth.
    """

    def __init__(
        self,
        model: GeoLocator,
        ref_model: GeoLocator,
        region_centroids: torch.Tensor,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.region_centroids = region_centroids.to(device)
        self.loss_fn = DPOGeoLoss(beta=beta).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        self.device = device

    def train_on_buffer(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 64,
        epochs: int = 3,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> float:
        """Run DPO fine-tuning on the replay buffer."""
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            loader = buffer.get_dataloader(batch_size=batch_size, shuffle=True)
            pbar = tqdm(loader, desc=f"DPO epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                images = batch["image"].to(self.device)
                true_coords = latlng_to_rad(
                    batch["lat"].to(self.device), batch["lng"].to(self.device)
                )
                better_coords = latlng_to_rad(
                    batch["better_lat"].to(self.device),
                    batch["better_lng"].to(self.device),
                )
                worse_coords = latlng_to_rad(
                    batch["worse_lat"].to(self.device),
                    batch["worse_lng"].to(self.device),
                )

                with torch.no_grad():
                    ref_outputs = self.ref_model(images)
                    ref_region_logits = ref_outputs["region_logits"]

                outputs_better = self.model(images)
                outputs_worse = self.model(images)

                better_region_logits = torch.softmax(outputs_better["region_logits"], dim=-1)
                worse_region_logits = torch.softmax(outputs_worse["region_logits"], dim=-1)

                better_pred_coords = better_region_logits @ self.region_centroids
                worse_pred_coords = worse_region_logits @ self.region_centroids

                from ..model.losses import haversine_distance
                d_better = haversine_distance(better_pred_coords, true_coords)
                d_worse = haversine_distance(worse_pred_coords, true_coords)

                loss = self.loss_fn(
                    ref_region_logits,
                    outputs_better["region_logits"],
                    outputs_worse["region_logits"],
                    self.region_centroids,
                    true_coords,
                )

                if d_better.mean() > d_worse.mean():
                    loss = self.loss_fn(
                        ref_region_logits,
                        outputs_worse["region_logits"],
                        outputs_better["region_logits"],
                        self.region_centroids,
                        true_coords,
                    )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(n_batches, 1)

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(checkpoint_dir / "dpo_finetuned.pth"))

        return avg_loss

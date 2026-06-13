from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from ..data.loader import GeoguessrDataset
from ..model.geolocator import GeoLocator
from ..model.losses import HierarchicalLoss, latlng_to_rad


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, top_k: tuple = (1, 3, 5)) -> dict[str, float]:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(top_k)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = {}
        for k in top_k:
            results[f"top{k}"] = correct[:k].float().sum(0).mean().item() * 100.0
    return results


class Trainer:
    """Supervised training loop for the geolocation model (Phase 1)."""

    def __init__(
        self,
        model: GeoLocator,
        loss_fn: HierarchicalLoss,
        train_loader,
        val_loader,
        config,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )
        self.scaler = GradScaler() if config.mixed_precision else None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_history: list[dict] = []

    def train_epoch(self) -> dict:
        self.model.train()
        losses = defaultdict(float)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            country_idx = batch["country_idx"].to(self.device)
            region_idx = batch["region_idx"].to(self.device)
            continent_idx = batch["continent_idx"].to(self.device)
            lat = batch["lat"].to(self.device)
            lng = batch["lng"].to(self.device)
            true_coords = latlng_to_rad(lat, lng)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast(self.device):
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(
                        outputs,
                        {
                            "country_idx": country_idx,
                            "region_idx": region_idx,
                            "continent_idx": continent_idx,
                            "true_coords": true_coords,
                        },
                    )
                self.scaler.scale(loss_dict["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(
                    outputs,
                    {
                        "country_idx": country_idx,
                        "region_idx": region_idx,
                        "continent_idx": continent_idx,
                        "true_coords": true_coords,
                    },
                )
                loss_dict["total"].backward()
                self.optimizer.step()

            for k, v in loss_dict.items():
                losses[k] += v.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total'].item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        losses = defaultdict(float)
        all_country_preds = []
        all_country_targets = []
        all_region_preds = []
        all_region_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch["image"].to(self.device)
            country_idx = batch["country_idx"].to(self.device)
            region_idx = batch["region_idx"].to(self.device)
            continent_idx = batch["continent_idx"].to(self.device)
            lat = batch["lat"].to(self.device)
            lng = batch["lng"].to(self.device)
            true_coords = latlng_to_rad(lat, lng)

            outputs = self.model(images)
            loss_dict = self.loss_fn(
                outputs,
                {
                    "country_idx": country_idx,
                    "region_idx": region_idx,
                    "continent_idx": continent_idx,
                    "true_coords": true_coords,
                },
            )
            for k, v in loss_dict.items():
                losses[k] += v.item()

            all_country_preds.append(outputs["country_logits"].cpu())
            all_country_targets.append(country_idx.cpu())
            all_region_preds.append(outputs["region_logits"].cpu())
            all_region_targets.append(region_idx.cpu())

        n_batches = len(self.val_loader)
        avg_losses = {k: v / n_batches for k, v in losses.items()}

        country_logits = torch.cat(all_country_preds)
        country_targets = torch.cat(all_country_targets)
        country_preds = country_logits.argmax(-1)
        balanced_acc = balanced_accuracy_score(country_targets.numpy(), country_preds.numpy()) * 100.0

        accuracies = compute_accuracy(country_logits, country_targets)

        return {
            **avg_losses,
            "balanced_accuracy": balanced_acc,
            **accuracies,
        }

    def fit(self, checkpoint_dir: str | Path) -> dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch
            start = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            elapsed = time.time() - start

            epoch_metrics = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "time": elapsed,
            }
            self.metrics_history.append(epoch_metrics)

            self.scheduler.step(val_metrics["total"])

            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.model.save(str(checkpoint_dir / "best_model.pth"))
                print(f"  ✓ Saved best model (val_loss={val_metrics['total']:.4f})")

            self._log_epoch(epoch_metrics)

            if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:
                self.model.save(str(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"))

        self.model.save(str(checkpoint_dir / "final_model.pth"))
        return epoch_metrics

    def _log_epoch(self, metrics: dict) -> None:
        print(
            f"Epoch {metrics['epoch']:3d} | "
            f"train_loss={metrics['train_total']:.4f} | "
            f"val_loss={metrics['val_total']:.4f} | "
            f"val_bal_acc={metrics.get('val_balanced_accuracy', 0):.2f}% | "
            f"val_top1={metrics.get('val_top1', 0):.2f}% | "
            f"time={metrics['time']:.1f}s"
        )

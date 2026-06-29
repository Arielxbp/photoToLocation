from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

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


def _mixup_data(
    images: torch.Tensor,
    country_idx: torch.Tensor,
    region_idx: torch.Tensor,
    continent_idx: torch.Tensor,
    alpha: float = 0.2,
    clue_features: Optional[torch.Tensor] = None,
) -> tuple:
    if alpha <= 0:
        return images, country_idx, None, region_idx, None, continent_idx, None, 1.0, None
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]

    mixed_clue = None
    if clue_features is not None:
        mixed_clue = lam * clue_features + (1 - lam) * clue_features[index]

    return (
        mixed_images,
        country_idx, country_idx[index],
        region_idx, region_idx[index],
        continent_idx, continent_idx[index],
        lam,
        mixed_clue,
    )


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

        self._build_scheduler()

        self.scaler = GradScaler() if config.mixed_precision else None
        self.mixup_alpha = getattr(config, "mixup_alpha", 0.0)

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_history: list[dict] = []

    def _build_scheduler(self) -> None:
        scheduler_type = getattr(self.cfg, "scheduler_type", "plateau")
        warmup_epochs = getattr(self.cfg, "warmup_epochs", 0)

        if scheduler_type == "cosine":
            total_epochs = self.cfg.epochs
            if warmup_epochs > 0:
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return (epoch + 1) / warmup_epochs
                    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                    return 0.5 * (1 + math.cos(math.pi * progress))
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_epochs,
                )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.cfg.lr_scheduler_factor,
                patience=self.cfg.lr_scheduler_patience,
            )

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
            clue_features = batch.get("clue_features")
            if clue_features is not None:
                clue_features = clue_features.to(self.device)

            images, c_a, c_b, r_a, r_b, ct_a, ct_b, lam, mixed_clue = _mixup_data(
                images, country_idx, region_idx, continent_idx, self.mixup_alpha,
                clue_features=clue_features,
            )

            self.optimizer.zero_grad()

            target_dict = {
                "country_idx": c_a,
                "region_idx": r_a,
                "continent_idx": ct_a,
                "true_coords": true_coords,
                "mixup_lam": lam,
            }
            if c_b is not None:
                target_dict["country_idx_shuffled"] = c_b
                target_dict["region_idx_shuffled"] = r_b
                target_dict["continent_idx_shuffled"] = ct_b

            if self.scaler:
                with autocast(self.device):
                    outputs = self.model(images, mixed_clue)
                    loss_dict = self.loss_fn(outputs, target_dict)
                self.scaler.scale(loss_dict["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, mixed_clue)
                loss_dict = self.loss_fn(outputs, target_dict)
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
            clue_features = batch.get("clue_features")
            if clue_features is not None:
                clue_features = clue_features.to(self.device)

            outputs = self.model(images, clue_features)
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

    def save_checkpoint(self, path: str | Path) -> None:
        scaler_state = self.scaler.state_dict() if self.scaler else None
        model_config = {
            "num_countries": self.model.country_head.fc[-1].out_features,
            "num_regions": self.model.region_head.fc[-1].out_features,
            "num_continents": self.model.continent_head.fc[-1].out_features,
        }
        if self.model.has_fusion and self.model.fusion is not None:
            model_config["clue_feature_dim"] = (
                self.model.fusion.clue_encoder.net[0].in_features
            )
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": scaler_state,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_history,
            "model_config": model_config,
            "training_config": {
                "epochs": self.cfg.epochs,
                "batch_size": self.cfg.batch_size,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "scheduler_type": getattr(self.cfg, "scheduler_type", "plateau"),
                "warmup_epochs": getattr(self.cfg, "warmup_epochs", 0),
                "mixup_alpha": getattr(self.cfg, "mixup_alpha", 0.0),
            },
        }
        torch.save(checkpoint, path)

    def load_state(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.scaler and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        return self.current_epoch

    def fit(self, checkpoint_dir: str | Path, resume_from: str | Path | None = None) -> dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_state(resume_from)
            print(f"Resumed from epoch {start_epoch} (best val_loss={self.best_val_loss:.4f})")

        for epoch in range(start_epoch, self.cfg.epochs):
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

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["total"])
            else:
                self.scheduler.step()

            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.model.save(str(checkpoint_dir / "best_model.pth"))
                print(f"  \u2713 Saved best model (val_loss={val_metrics['total']:.4f})")

            self.save_checkpoint(str(checkpoint_dir / "training_state.pt"))
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

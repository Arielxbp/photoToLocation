from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.geolocator import GeoLocator
from ..model.losses import DPOGeoLoss, haversine_distance, latlng_to_rad
from ..self_play.buffer import ReplayBuffer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@torch.no_grad()
def validate_model(
    model: GeoLocator,
    val_loader: DataLoader,
    region_centroids: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a model on a Kaggle-style validation dataloader.

    Returns metrics: country top-1 accuracy, balanced accuracy,
    mean/median haversine distance (km), and loss.
    """
    model.eval()
    all_country_preds = []
    all_country_targets = []
    all_distances: list[float] = []
    total_loss = 0.0
    n_batches = 0

    region_centroids = region_centroids.to(device)

    for batch in val_loader:
        images = batch["image"].to(device)
        country_idx = batch["country_idx"].to(device)
        lat = batch["lat"].to(device)
        lng = batch["lng"].to(device)

        outputs = model(images)
        all_country_preds.append(outputs["country_logits"].cpu())
        all_country_targets.append(country_idx.cpu())

        region_probs = torch.softmax(outputs["region_logits"], dim=-1)
        pred_coords = region_probs @ region_centroids
        true_coords = latlng_to_rad(lat, lng)
        d = haversine_distance(pred_coords, true_coords)
        all_distances.extend(d.cpu().tolist())

        total_loss += 0.0
        n_batches += 1

    country_logits = torch.cat(all_country_preds)
    country_targets = torch.cat(all_country_targets)
    country_preds = country_logits.argmax(-1)

    balanced_acc = float(balanced_accuracy_score(
        country_targets.numpy(), country_preds.numpy()
    ) * 100.0)

    top1_correct = (country_preds == country_targets).float().mean().item() * 100.0

    distances = np.array(all_distances)
    return {
        "country_top1_pct": top1_correct,
        "balanced_accuracy_pct": balanced_acc,
        "mean_distance_km": float(np.mean(distances)),
        "median_distance_km": float(np.median(distances)),
    }


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
        epochs: int = 1,
        checkpoint_dir: Optional[str | Path] = None,
        iteration: int = 0,
        val_loader: Optional[DataLoader] = None,
        max_degradation_pct: float = 5.0,
        cell_to_idx: dict[int, int] | None = None,
        s2_level: int = 6,
        country_index: dict[str, int] | None = None,
        dpo_loss_weight: float = 0.1,
        sft_loss_weight: float = 1.0,
    ) -> dict:
        """
        Run DPO fine-tuning on the replay buffer.

        Returns a dict with keys: loss, val_before, val_after, degraded, checkpoint_path.
        """
        val_before = None
        val_after = None

        has_val = val_loader is not None
        if has_val:
            self.ref_model.cpu()
            torch.cuda.empty_cache()
            val_before = validate_model(self.model, val_loader, self.region_centroids, self.device)
            print(f"  Validation before DPO: top1={val_before['country_top1_pct']:.1f}%  "
                  f"bal_acc={val_before['balanced_accuracy_pct']:.1f}%  "
                  f"mean_dist={val_before['mean_distance_km']:.0f} km")
            torch.cuda.empty_cache()
            self.ref_model.to(self.device)
            torch.cuda.empty_cache()

        self.model.train()
        country_ce = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            total_dpo = 0.0
            total_sft = 0.0
            n_batches = 0
            pref_correct = 0
            pref_total = 0

            loader = buffer.get_dataloader(
                batch_size=batch_size,
                shuffle=True,
                cell_to_idx=cell_to_idx,
                s2_level=s2_level,
                country_index=country_index,
            )
            pbar = tqdm(loader, desc=f"DPO epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                images = batch["image"].to(self.device)

                if "preferred_region_idx" not in batch:
                    raise RuntimeError(
                        "ReplayBufferDataset did not produce preference indices. "
                        "Pass cell_to_idx to buffer.get_dataloader()."
                    )
                preferred_idx = batch["preferred_region_idx"].to(self.device)
                dispreferred_idx = batch["dispreferred_region_idx"].to(self.device)

                with torch.no_grad():
                    ref_outputs = self.ref_model(images)
                    ref_region_logits = ref_outputs["region_logits"].detach()
                del ref_outputs

                outputs = self.model(images)
                model_region_logits = outputs["region_logits"]

                dpo_loss = self.loss_fn(
                    ref_region_logits,
                    model_region_logits,
                    preferred_idx,
                    dispreferred_idx,
                )

                sft_loss = torch.tensor(0.0, device=self.device)
                if country_index is not None and "country_idx" in batch:
                    country_targets = batch["country_idx"].to(self.device)
                    sft_loss = country_ce(outputs["country_logits"], country_targets)

                loss = dpo_loss_weight * dpo_loss + sft_loss_weight * sft_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    log_probs = F.log_softmax(model_region_logits, dim=-1)
                    model_pref = log_probs.gather(1, preferred_idx.unsqueeze(-1)).squeeze(-1)
                    model_disf = log_probs.gather(1, dispreferred_idx.unsqueeze(-1)).squeeze(-1)
                    pref_correct += (model_pref > model_disf).sum().item()
                    pref_total += preferred_idx.numel()

                del outputs

                total_loss += loss.item()
                total_dpo += dpo_loss.item()
                total_sft += sft_loss.item()
                n_batches += 1

                pref_acc = pref_correct / max(pref_total, 1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dpo": f"{dpo_loss.item():.4f}",
                    "sft": f"{sft_loss.item():.4f}",
                    "pref_acc": f"{pref_acc:.1%}",
                })

            avg_loss = total_loss / max(n_batches, 1)
            pref_acc = pref_correct / max(pref_total, 1)
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  "
                  f"dpo={total_dpo/max(n_batches,1):.4f}  "
                  f"sft={total_sft/max(n_batches,1):.4f}  "
                  f"pref_accuracy={pref_acc:.1%}")

        degraded = False
        if has_val:
            self.ref_model.cpu()
            torch.cuda.empty_cache()
            val_after = validate_model(self.model, val_loader, self.region_centroids, self.device)
            print(f"  Validation after DPO:  top1={val_after['country_top1_pct']:.1f}%  "
                  f"bal_acc={val_after['balanced_accuracy_pct']:.1f}%  "
                  f"mean_dist={val_after['mean_distance_km']:.0f} km")

            if val_before is not None:
                dist_delta = val_after["mean_distance_km"] - val_before["mean_distance_km"]
                top1_delta = val_after["country_top1_pct"] - val_before["country_top1_pct"]
                if dist_delta > 0:
                    pct_increase = (dist_delta / max(val_before["mean_distance_km"], 0.001)) * 100
                    if pct_increase > max_degradation_pct:
                        print(
                            f"  [WARN] Mean distance increased by {pct_increase:.1f}% "
                            f"(> {max_degradation_pct}% threshold) "
                            f"— DPO may have degraded accuracy"
                        )
                        degraded = True
                if top1_delta < 0:
                    print(
                        f"  Top1 accuracy changed by {top1_delta:+.1f}%  "
                        f"(from {val_before['country_top1_pct']:.1f}%"
                        f" to {val_after['country_top1_pct']:.1f}%)"
                    )

        checkpoint_path = None
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            ckpt_name = f"dpo_finetuned_iter{iteration}.pth"
            ckpt_path = checkpoint_dir / ckpt_name
            self.model.save(str(ckpt_path))
            checkpoint_path = str(ckpt_path)

            latest_path = checkpoint_dir / "dpo_finetuned_latest.pth"
            self.model.save(str(latest_path))

            meta = {
                "iteration": iteration,
                "loss": avg_loss,
                "buffer_size": len(buffer),
                "buffer_stats": buffer.stats,
                "val_before": val_before,
                "val_after": val_after,
                "degraded": degraded,
                "checkpoint": checkpoint_path,
            }
            meta_path = checkpoint_dir / f"dpo_meta_iter{iteration}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            if degraded:
                warn_path = checkpoint_dir / f"dpo_finetuned_iter{iteration}_degraded_warning.pth"
                self.model.save(str(warn_path))

        return {
            "loss": avg_loss,
            "val_before": val_before,
            "val_after": val_after,
            "degraded": degraded,
            "checkpoint_path": checkpoint_path,
        }

#!/usr/bin/env python3
"""Compare trained GeoLocator model vs StreetCLIP on the same self-play rounds.

StreetCLIP (geolocal/StreetCLIP) is a CLIP ViT-L/14 model fine-tuned for
zero-shot geolocalization via image-text matching.  For each round both
models see the identical screenshot: the trained model runs its EfficientNet
heads to predict a country centroid, and StreetCLIP matches the image against
all country names.  The trained model's guess is clicked on the minimap so
the round can complete and yield ground-truth coordinates.  Haversine
distances are then computed for both predictions and compared.

Requires Playwright with a Chromium browser and the ``transformers`` library:
    pip install -e '.[selfplay]' transformers
    playwright install chromium

Usage examples:

    python scripts/compare_models.py
    python scripts/compare_models.py --model checkpoints/dpo_finetuned_iter5.pth
    python scripts/compare_models.py --rounds 100 --output data/comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from geoguessr_agent.config import load_config
from geoguessr_agent.game.actions import (
    click_guess_on_map,
    dismiss_cookie_banner,
    go_to_next_round,
    hover_minimap_to_expand,
    start_game,
    submit_guess,
)
from geoguessr_agent.game.browser import OpenGuessrBrowser
from geoguessr_agent.game.state_machine import GameState, GameStateMachine
from geoguessr_agent.geoutils import get_country_centroids
from geoguessr_agent.model.geolocator import GeoLocator


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _preprocess_for_efficientnet(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((320, 180), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = torch.from_numpy((arr - mean) / std).float()
    return tensor.unsqueeze(0)


@torch.no_grad()
def _geolocator_predict(
    model: GeoLocator,
    tensor: torch.Tensor,
    country_centroids: torch.Tensor,
    idx_to_country: dict[int, str],
) -> dict:
    outputs = model(tensor)

    country_probs = torch.softmax(outputs["country_logits"], dim=-1)
    top5_conf, top5_idx = torch.topk(country_probs, k=5, dim=-1)
    country_conf, country_idx = country_probs.max(dim=-1)
    country_idx = country_idx.item()
    country_conf = country_conf.item()

    centroid = country_centroids[country_idx]
    lat = float(torch.rad2deg(centroid[0]).item())
    lng = float(torch.rad2deg(centroid[1]).item())
    if lat != 0.0 or lng != 0.0:
        lat += random.uniform(-3.0, 3.0)
        lng += random.uniform(-3.0, 3.0)
    lat = max(-85.0, min(85.0, lat))
    lng = max(-180.0, min(180.0, lng))

    return {
        "latitude": lat,
        "longitude": lng,
        "country_idx": country_idx,
        "country_conf": country_conf,
        "country_name": idx_to_country.get(country_idx, "Unknown"),
        "top5_conf": top5_conf.squeeze(0).cpu().tolist(),
        "top5_idx": top5_idx.squeeze(0).cpu().tolist(),
    }


class _WarmupModel(torch.nn.Module):
    """Dummy module so the StreetCLIP ``CLIPModel`` can be moved to a device.

    ``CLIPModel`` inherits from ``PreTrainedModel`` which overrides ``.to()``
    and calls ``super().to(*args, **kwargs)`` — but when ``super()`` is not a
    ``nn.Module`` this silently produces a plain Python object instead of a
    device-placed model.  Wrapping in a basic ``nn.Module`` fixes it.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, **kwargs):
        return self.clip(**kwargs)


def _load_streetclip(device: str):
    """Load geolocal/StreetCLIP via HuggingFace ``transformers``."""
    from transformers import CLIPModel, CLIPProcessor

    print("Loading StreetCLIP from geolocal/StreetCLIP ...")
    clip = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    wrapped = _WarmupModel(clip).to(device).eval()
    return wrapped, processor


@torch.no_grad()
def _streetclip_predict(
    streetclip: torch.nn.Module,
    processor,
    img_bytes: bytes,
    country_names: list[str],
    country_centroids: torch.Tensor,
    country_index: dict[str, int],
) -> dict:
    """Run StreetCLIP zero-shot country classification.

    Passes the image against all *country_names* as text labels and returns
    the top-scoring country's centroid coordinates.
    """
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(
        text=country_names, images=pil_img, return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(next(streetclip.parameters()).device) for k, v in inputs.items()}

    outputs = streetclip(**inputs)
    logits = outputs.logits_per_image  # (1, num_texts)
    probs = logits.softmax(dim=-1)

    best_idx = probs.argmax(dim=-1).item()
    best_country = country_names[best_idx]
    best_conf = probs[0, best_idx].item()

    # Map country name → centroid index
    centroid_idx = country_index.get(best_country)
    if centroid_idx is not None:
        centroid = country_centroids[centroid_idx]
        lat = float(torch.rad2deg(centroid[0]).item())
        lng = float(torch.rad2deg(centroid[1]).item())
    else:
        lat, lng = 0.0, 0.0

    if lat != 0.0 or lng != 0.0:
        lat += random.uniform(-3.0, 3.0)
        lng += random.uniform(-3.0, 3.0)
    lat = max(-85.0, min(85.0, lat))
    lng = max(-180.0, min(180.0, lng))

    top5_probs, top5_indices = probs[0].topk(min(5, len(country_names)))
    top5_conf = top5_probs.cpu().tolist()
    top5_idx = top5_indices.cpu().tolist()

    return {
        "latitude": lat,
        "longitude": lng,
        "country_idx": centroid_idx if centroid_idx is not None else 0,
        "country_conf": best_conf,
        "country_name": best_country,
        "top5_conf": top5_conf,
        "top5_idx": top5_idx,
    }


class ComparativeSelfPlayLoop:
    """Runs self-play rounds evaluating both the trained GeoLocator and
    StreetCLIP on the same screenshots.

    The trained model's guess is clicked on the map so the round can
    complete; StreetCLIP's prediction is recorded silently.
    """

    def __init__(
        self,
        config,
        trained_model: GeoLocator,
        streetclip_model,
        streetclip_processor,
        country_index: dict[str, int],
        idx_to_country: dict[int, str],
        country_names: list[str],
        country_centroids: torch.Tensor,
        exploration_epsilon: float = 0.1,
    ):
        self.cfg = config
        self.trained_model = trained_model.to(config.device).eval()
        self.streetclip = streetclip_model
        self.streetclip_processor = streetclip_processor
        self.country_index = country_index
        self.idx_to_country = idx_to_country
        self.country_names = country_names
        self.country_centroids = country_centroids.to(config.device)
        self.exploration_epsilon = exploration_epsilon

        self.browser = OpenGuessrBrowser(
            url=config.game.url,
            headless=config.game.headless,
            stealth=config.game.stealth,
            viewport_width=config.game.viewport_width,
            viewport_height=config.game.viewport_height,
        )
        self.state_machine = GameStateMachine()

        self.screenshot_dir = Path(config.game.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict] = []
        self._rounds_played = 0

    async def _has_street_view(self) -> bool:
        try:
            return await self.browser.page.evaluate("""
            () => {
                const sel = '#panorama-iframe, [class*="panorama"], ' +
                    '[class*="street-view"], [class*="StreetView"], .gm-style';
                const el = document.querySelector(sel);
                if (el && el.offsetParent !== null) return true;
                const mapEl = document.querySelector('.leaflet-container, #map');
                return mapEl && mapEl.offsetParent !== null;
            }
            """)
        except Exception:
            return False

    async def _wait_for_street_view(self, timeout_ms: int = 15_000) -> bool:
        try:
            await self.browser.page.wait_for_selector(
                '#panorama-iframe, .gm-style, [class*="panorama"], '
                '[class*="street-view"], [class*="StreetView"], '
                '.leaflet-container, #map',
                timeout=timeout_ms,
            )
            return True
        except Exception:
            return False

    async def _wait_for_loading_clear(self, timeout_sec: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            loading = await self.browser.page.evaluate("""
            () => {
                const spinners = document.querySelectorAll(
                    '[class*="loading"], [class*="spinner"], '
                    + '.compass_spinner, [role="progressbar"]'
                );
                for (const s of spinners) {
                    if (s.offsetParent !== null) return true;
                }
                return false;
            }
            """)
            if not loading:
                return
            await asyncio.sleep(1.0)

    async def _play_round(self, round_num: int) -> Optional[dict]:
        print(f"  Round {round_num}...")
        await asyncio.sleep(random.uniform(0.5, 1.0))

        self.state_machine.clear_intercepted_data()
        await self.browser.clear_tiles()

        state = await self.state_machine.detect_state(self.browser.page)
        if state not in (GameState.ROUND_ACTIVE,):
            await self.state_machine.wait_for_state(
                self.browser.page, GameState.ROUND_ACTIVE, timeout_ms=10000
            )

        if not await self._has_street_view():
            if not await self._wait_for_street_view(timeout_ms=10_000):
                print("  [WARN] No Street View in round — skipping")
                return None
            await asyncio.sleep(3)

        await self._wait_for_loading_clear(timeout_sec=10.0)

        capture_dir = self.screenshot_dir / f"round_{round_num}"
        capture_dir.mkdir(parents=True, exist_ok=True)
        img_bytes = await self.browser.page.screenshot(path=str(capture_dir / "streetview.png"))

        pano_truth = await self.browser.read_pano_location()

        # ---- Trained GeoLocator ----
        tensor = _preprocess_for_efficientnet(img_bytes).to(self.cfg.device)
        with torch.no_grad():
            trained_pred = _geolocator_predict(
                self.trained_model, tensor,
                self.country_centroids, self.idx_to_country,
            )

        # ---- StreetCLIP baseline ----
        streetclip_pred = _streetclip_predict(
            self.streetclip, self.streetclip_processor, img_bytes,
            self.country_names, self.country_centroids, self.country_index,
        )

        # Exploration epsilon for the *clicked* prediction
        click_pred = dict(trained_pred)
        if self.exploration_epsilon > 0 and random.random() < self.exploration_epsilon:
            rnd_idx = random.randint(0, len(self.idx_to_country) - 1)
            centroid = self.country_centroids[rnd_idx]
            click_pred["latitude"] = float(torch.rad2deg(centroid[0]).item())
            click_pred["longitude"] = float(torch.rad2deg(centroid[1]).item())
            if click_pred["latitude"] != 0.0 or click_pred["longitude"] != 0.0:
                click_pred["latitude"] += random.uniform(-3.0, 3.0)
                click_pred["longitude"] += random.uniform(-3.0, 3.0)

        print(f"    GeoLocator: ({trained_pred['latitude']:.2f}, {trained_pred['longitude']:.2f})"
              f" → {trained_pred['country_name']} (conf={trained_pred['country_conf']:.2f})")
        print(f"    StreetCLIP: ({streetclip_pred['latitude']:.2f}, {streetclip_pred['longitude']:.2f})"
              f" → {streetclip_pred['country_name']} (conf={streetclip_pred['country_conf']:.2f})")

        # ---- click with the trained model ----
        if not await hover_minimap_to_expand(self.browser.page):
            print("    [WARN] Could not hover minimap")
        await asyncio.sleep(0.3)
        await click_guess_on_map(self.browser.page, click_pred["latitude"], click_pred["longitude"])
        await asyncio.sleep(random.uniform(0.3, 0.6))

        submitted = await submit_guess(self.browser.page)
        if not submitted:
            await self.browser.page.keyboard.press("Enter")
            await asyncio.sleep(random.uniform(0.5, 1.0))

        await asyncio.sleep(0.8)
        await self.browser.page.screenshot(path=str(capture_dir / "results.png"))

        results = await self.state_machine.poll_results_content(
            self.browser.page, timeout_ms=3000, poll_ms=100
        )

        true_coords = pano_truth
        if not true_coords and results and results.get("lat") is not None:
            true_coords = (float(results["lat"]), float(results["lng"]))
        if not true_coords:
            true_coords = await self.state_machine.read_true_coordinates(self.browser.page)
        if not true_coords:
            for data in self.state_machine._intercepted_data:
                coords = self.state_machine._extract_coords_from_state(data)
                if coords:
                    true_coords = coords
                    break
        if not true_coords:
            print("    [WARN] Could not read true coordinates — skipping round")
            return None

        true_lat, true_lng = true_coords

        trained_dist = _haversine_km(
            trained_pred["latitude"], trained_pred["longitude"],
            true_lat, true_lng,
        )
        streetclip_dist = _haversine_km(
            streetclip_pred["latitude"], streetclip_pred["longitude"],
            true_lat, true_lng,
        )

        print(f"    True: ({true_lat:.4f}, {true_lng:.4f})  "
              f"GeoLocator={trained_dist:.0f} km  StreetCLIP={streetclip_dist:.0f} km")

        return {
            "round": round_num,
            "true_lat": true_lat,
            "true_lng": true_lng,
            "geo_lat": trained_pred["latitude"],
            "geo_lng": trained_pred["longitude"],
            "geo_country": trained_pred["country_name"],
            "geo_conf": trained_pred["country_conf"],
            "geo_distance_km": trained_dist,
            "streetclip_lat": streetclip_pred["latitude"],
            "streetclip_lng": streetclip_pred["longitude"],
            "streetclip_country": streetclip_pred["country_name"],
            "streetclip_conf": streetclip_pred["country_conf"],
            "streetclip_distance_km": streetclip_dist,
        }

    async def run_session(self, num_rounds: int = 50) -> list[dict]:
        print(f"\n{'='*60}")
        print(f"GeoLocator vs StreetCLIP  —  {num_rounds} rounds")
        print(f"{'='*60}")

        try:
            await self.browser.start()
            print("Browser started")
            await asyncio.sleep(2)

            dismissed = await dismiss_cookie_banner(self.browser.page)
            if dismissed:
                print("Cookie consent dismissed")
                await asyncio.sleep(1)

            await self.state_machine.setup_response_interception(self.browser.page)

            state = await self.state_machine.detect_state(self.browser.page)
            if state == GameState.MAIN_MENU:
                started = await start_game(self.browser.page)
                if not started:
                    print("[ERROR] Could not start game")
                    return []
                await asyncio.sleep(2)

            if not await self._has_street_view():
                clicked = await submit_guess(self.browser.page)
                if not clicked:
                    print("[ERROR] Could not enter game")
                    return []
                print("  Started game")
                await asyncio.sleep(2)
                if not await self._wait_for_street_view(timeout_ms=15_000):
                    print("[ERROR] Street View did not appear")
                    return []
                await asyncio.sleep(2)

            for r in range(1, num_rounds + 1):
                try:
                    record = await self._play_round(r)
                    if record:
                        self.records.append(record)
                        self._rounds_played += 1
                    await go_to_next_round(self.browser.page)
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                except Exception as e:
                    print(f"  [ERROR] Round {r}: {e}")
                    continue

        finally:
            await self.browser.stop()

        return self.records

    def stats(self) -> dict:
        if not self.records:
            return {"rounds": 0}

        geo_dists = [r["geo_distance_km"] for r in self.records]
        sc_dists = [r["streetclip_distance_km"] for r in self.records]

        geo_wins = sum(1 for g, s in zip(geo_dists, sc_dists) if g < s)
        sc_wins = sum(1 for s, g in zip(sc_dists, geo_dists) if s < g)
        ties = len(self.records) - geo_wins - sc_wins

        return {
            "rounds": len(self.records),
            "geolocator": {
                "mean_km": np.mean(geo_dists),
                "median_km": np.median(geo_dists),
                "min_km": np.min(geo_dists),
                "max_km": np.max(geo_dists),
                "std_km": np.std(geo_dists),
            },
            "streetclip": {
                "mean_km": np.mean(sc_dists),
                "median_km": np.median(sc_dists),
                "min_km": np.min(sc_dists),
                "max_km": np.max(sc_dists),
                "std_km": np.std(sc_dists),
            },
            "head_to_head": {
                "geolocator_wins": geo_wins,
                "streetclip_wins": sc_wins,
                "ties": ties,
            },
        }


def _print_stats(stats: dict) -> None:
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    if stats["rounds"] == 0:
        print("No rounds completed.")
        return

    print(f"\nTotal rounds: {stats['rounds']}")
    print(f"\n{'Metric':<16} {'GeoLocator':>10} {'StreetCLIP':>10} {'Δ (G−S)':>10}")
    print(f"{'-'*16} {'-'*10} {'-'*10} {'-'*10}")

    for metric in ("mean_km", "median_km", "min_km", "max_km", "std_km"):
        g = stats["geolocator"][metric]
        s = stats["streetclip"][metric]
        delta = g - s
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<16} {g:10.1f} {s:10.1f} {sign}{delta:9.1f}")

    h2h = stats["head_to_head"]
    total = h2h["geolocator_wins"] + h2h["streetclip_wins"] + h2h["ties"]
    print(f"\nHead-to-head (lower distance wins):")
    print(f"  GeoLocator wins: {h2h['geolocator_wins']:3d} ({100*h2h['geolocator_wins']/total:.0f}%)")
    print(f"  StreetCLIP wins: {h2h['streetclip_wins']:3d} ({100*h2h['streetclip_wins']/total:.0f}%)")
    print(f"  Ties:            {h2h['ties']:3d} ({100*h2h['ties']/total:.0f}%)")

    mean_diff = stats["geolocator"]["mean_km"] - stats["streetclip"]["mean_km"]
    direction = "better" if mean_diff < 0 else "worse"
    print(f"\nGeoLocator is {abs(mean_diff):.0f} km {direction} on average vs StreetCLIP.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained GeoLocator vs StreetCLIP on self-play rounds"
    )
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--model", default="checkpoints/best_model.pth",
        help="Path to the trained GeoLocator checkpoint (.pth)",
    )
    parser.add_argument(
        "--indices", default="data/indices.json",
        help="Path to indices JSON (country_index, idx_to_country, etc.)",
    )
    parser.add_argument("--rounds", type=int, default=50, help="Number of comparison rounds")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Exploration epsilon (overrides config)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save comparison records as JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    epsilon = args.epsilon if args.epsilon is not None else config.dpo.exploration_epsilon

    with open(args.indices) as f:
        indices = json.load(f)
    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}
    country_centroids = get_country_centroids(indices["country_index"])

    # Build ordered list of country names for StreetCLIP text labels and
    # a lookup from country name → centroid index.
    max_idx = max(idx_to_country.keys())
    country_names = [idx_to_country.get(i, "Unknown") for i in range(max_idx + 1)]

    # ---- Load trained GeoLocator ----
    print(f"Loading trained GeoLocator: {args.model}")
    trained = GeoLocator(
        num_countries=indices["num_countries"],
        num_regions=indices["num_regions"],
        num_continents=indices["num_continents"],
        pretrained=False,
    )
    trained.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"]
    )

    # ---- Load StreetCLIP baseline ----
    sc_model, sc_processor = _load_streetclip(config.device)

    loop = ComparativeSelfPlayLoop(
        config=config,
        trained_model=trained,
        streetclip_model=sc_model,
        streetclip_processor=sc_processor,
        country_index=indices["country_index"],
        idx_to_country=idx_to_country,
        country_names=country_names,
        country_centroids=country_centroids,
        exploration_epsilon=epsilon,
    )

    records = asyncio.run(loop.run_session(num_rounds=args.rounds))

    stats = loop.stats()
    _print_stats(stats)

    if args.output and records:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for r in records:
            serializable.append({
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()
            })
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nComparison records saved → {output_path}")


if __name__ == "__main__":
    main()

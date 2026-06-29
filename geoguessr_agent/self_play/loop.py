from __future__ import annotations

import asyncio
import io
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import re
import httpx
import numpy as np
import torch
from PIL import Image

from ..config import Config
from ..game.actions import (
    click_guess_on_map,
    dismiss_cookie_banner,
    go_to_next_round,
    hover_minimap_to_expand,
    start_game,
    submit_guess,
)
from ..game.browser import OpenGuessrBrowser
from ..game.state_machine import GameState, GameStateMachine
from ..inference.gradcam import generate_annotated_heatmap
from ..model.geolocator import GeoLocator
from ..plonkit.kb import ClueKnowledgeBase
from ..self_play.buffer import ReplayBuffer, ReplayEntry
from ..self_play.reward import RewardCalculator

API_KEY=os.environ.get("MAPS_API", "")
IMAGE_SIZE="1280x720"
PANO_REGEX = re.compile(r"panoid=([^&]+)")

class SelfPlayLoop:
    """
    Orchestrates the self-play loop:

    1. Start OpenGuessr session
    2. For each round: screenshot Street View → inference → hover minimap
       → click map → read score
    3. Store (image, guess, true_location, score) in replay buffer
    4. Periodically trigger DPO fine-tuning

    Guess coordinates come from a per-country centroid table (with a small
    jitter); the table covers every predictable country, so no guess lands at
    (0, 0). The trained region head is used only if no table is supplied.
    """

    def __init__(
        self,
        config: Config,
        model: GeoLocator,
        region_centroids: torch.Tensor,
        country_index: dict[str, int],
        idx_to_country: dict[int, str],
        country_centroids: Optional[torch.Tensor] = None,
        kb: Optional[ClueKnowledgeBase] = None,
        enable_heatmaps: bool = False,
    ):
        self.cfg = config
        self.model = model.to(config.device)
        self.model.eval()
        self.region_centroids = region_centroids.to(config.device)
        self.country_index = country_index
        self.idx_to_country = idx_to_country
        if country_centroids is not None:
            self.country_centroids = country_centroids.to(config.device)
        else:
            self.country_centroids = None
        self.kb = kb
        self.enable_heatmaps = enable_heatmaps
        self.exploration_epsilon = config.dpo.exploration_epsilon

        self.browser = OpenGuessrBrowser(
            url=config.game.url,
            headless=config.game.headless,
            stealth=config.game.stealth,
            viewport_width=config.game.viewport_width,
            viewport_height=config.game.viewport_height,
        )

        self.state_machine = GameStateMachine()
        self.buffer = ReplayBuffer(max_size=config.dpo.buffer_size)
        self.reward_calc = RewardCalculator()

        self.screenshot_dir = Path(config.game.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._rounds_played = 0
        self._daily_rounds = 0
        self._total_score = 0.0
        self._session_start = datetime.now()

    def _preprocess_image(self, img_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((320, 180), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        tensor = torch.from_numpy((arr - mean) / std).float()
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def _infer_location(self, img_bytes: bytes) -> dict:
        """
        Run inference on a screenshot.

        Coordinates come from the country-centroid table (one fixed point per
        country, plus a small ±3° jitter so repeated guesses for the same
        country are not pixel-identical). The table covers every country the
        model can predict, so no guess collapses to (0, 0). If no centroid
        table was supplied, the trained region head is used as a fallback.

        Returns dict with keys:
          latitude, longitude, country_idx, country_conf,
          top5_conf, top5_idx, continent_idx, outputs
        """
        tensor = self._preprocess_image(img_bytes).to(self.cfg.device)
        outputs = self.model(tensor)

        country_probs = torch.softmax(outputs["country_logits"], dim=-1)
        top5_conf, top5_idx = torch.topk(country_probs, k=5, dim=-1)
        country_conf, country_idx = country_probs.max(dim=-1)
        country_idx = country_idx.item()
        country_conf = country_conf.item()

        if self.exploration_epsilon > 0 and random.random() < self.exploration_epsilon:
            country_idx = random.randint(0, len(self.idx_to_country) - 1)
            country_conf = 0.0

        # Coordinates: country centroid (primary) + small jitter.
        if self.country_centroids is not None:
            centroid = self.country_centroids[country_idx]
            lat = float(torch.rad2deg(centroid[0]).item())
            lng = float(torch.rad2deg(centroid[1]).item())
            if lat != 0.0 or lng != 0.0:
                lat += random.uniform(-3.0, 3.0)
                lng += random.uniform(-3.0, 3.0)
        else:
            # No centroid table — fall back to the trained region head.
            region_probs = torch.softmax(outputs["region_logits"], dim=-1)
            top_probs, top_idx = torch.topk(region_probs, k=5, dim=-1)
            weighted = top_probs @ self.region_centroids[top_idx.squeeze(0)]
            coords_rad = weighted.squeeze(0)
            lat = float(torch.rad2deg(coords_rad[0]).item())
            lng = float(torch.rad2deg(coords_rad[1]).item())

        lat = max(-85.0, min(85.0, lat))
        lng = max(-180.0, min(180.0, lng))

        continent_probs = torch.softmax(outputs["continent_logits"], dim=-1)
        continent_idx = continent_probs.argmax(-1).item()

        return {
            "latitude": lat,
            "longitude": lng,
            "country_idx": country_idx,
            "country_conf": country_conf,
            "top5_conf": top5_conf.squeeze(0).cpu().tolist(),
            "top5_idx": top5_idx.squeeze(0).cpu().tolist(),
            "continent_idx": continent_idx,
            "outputs": outputs,
        }

    async def _ensure_game_started(self) -> bool:
        """Click 'Guess' on config screen and wait for Street View to appear."""
        try:
            if await self._has_street_view():
                return True

            clicked = await submit_guess(self.browser.page)
            if not clicked:
                return False

            print("  Started game (clicked Guess on config screen)")
            await asyncio.sleep(2)

            if await self._wait_for_street_view(timeout_ms=15_000):
                await asyncio.sleep(2)
                return True

            print("  [WARN] Street View did not appear after starting game")
            return False
        except Exception as e:
            print(f"  [WARN] Game start error: {e}")
            return False

    async def _has_street_view(self) -> bool:
        """Check if the Street View iframe or a map element is visible."""
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
        """Wait for the Street View iframe or map to appear."""
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
        """Wait for loading spinners to disappear."""
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

    async def _play_round(self, round_num: int) -> Optional[ReplayEntry]:
        """Play a single round of OpenGuessr using screenshot-based capture."""
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

        # Wait for loading screen to clear so the screenshot has clean content
        await self._wait_for_loading_clear(timeout_sec=10.0)

        capture_dir = self.screenshot_dir / f"round_{round_num}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        intercept_pano_id = None

        for frame in self.browser.page.frames:
            try:
                # Execute a script inside every frame context to check its performance log
                cached_url = await frame.evaluate("""
                    () => {
                        const resources = performance.getEntriesByType("resource");
                        for (const res of resources) {
                            if (res.name.includes("streetviewpixels") && res.name.includes("panoid=")) {
                                return res.name;
                            }
                        }
                        return null;
                    }
                """)
                if cached_url:
                    match = PANO_REGEX.search(cached_url)
                    if match:
                        intercept_pano_id = match.group(1)
                        break
            except Exception:
                continue
        
        screenshot_path = capture_dir / "streetview.png"
        if intercept_pano_id is not None and API_KEY != "":
            print(f"Downloading image with PANO_ID: {intercept_pano_id} from Maps API")
            url = f"https://maps.googleapis.com/maps/api/streetview?size={IMAGE_SIZE}&pano={intercept_pano_id}&key={API_KEY}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    print(f"Error fetching API: {response.status_code}")
                    return None
                img_bytes = response.content
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_image.save(screenshot_path, quality=95)
        else:
            print(f"Taking screenshot of browser page")
            img_bytes = await self.browser.page.screenshot(path=str(screenshot_path))
        
        print(f"    Img: {len(img_bytes)} bytes")

        # Capture ground truth NOW, while the panorama is still on screen.
        # OpenGuessr embeds the round's true location in the Street View
        # iframe's `location=` parameter; it disappears once we submit.
        pano_truth = await self.browser.read_pano_location()
        if pano_truth:
            print(f"    True location (iframe): ({pano_truth[0]:.4f}, {pano_truth[1]:.4f})")

        # ---- inference ----
        result = self._infer_location(img_bytes)
        pred_lat = result["latitude"]
        pred_lng = result["longitude"]
        country_idx = result["country_idx"]
        conf = result["country_conf"]
        continent_idx = result["continent_idx"]

        pred_country = self.idx_to_country.get(country_idx, "Unknown")
        print(
            f"    Predicted: ({pred_lat:.4f}, {pred_lng:.4f})"
            f" -> {pred_country} (conf={conf:.2f})"
        )

        # ---- Grad-CAM heatmap (optional) ----
        if self.enable_heatmaps:
            try:
                input_tensor = self._preprocess_image(img_bytes)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                annotated = generate_annotated_heatmap(
                    model=self.model,
                    input_tensor=input_tensor,
                    original_image=pil_image,
                    target_class=country_idx,
                    country_name=pred_country,
                    confidence=conf,
                    continent=self.idx_to_country.get(continent_idx, "?"),
                    lat=pred_lat,
                    lng=pred_lng,
                    device=self.cfg.device,
                )
                if annotated:
                    heatmap_path = capture_dir / "heatmap.jpg"
                    annotated.save(str(heatmap_path), quality=92)
                    print(f"    Heatmap saved → {heatmap_path}")
            except Exception as e:
                print(f"    [WARN] Heatmap generation failed: {e}")

        # ---- hover minimap to expand → click ----
        if not await hover_minimap_to_expand(self.browser.page):
            print("    [WARN] Could not hover minimap")

        await asyncio.sleep(0.3)
        await click_guess_on_map(self.browser.page, pred_lat, pred_lng)
        await asyncio.sleep(random.uniform(0.3, 0.6))

        # ---- submit ----
        submitted = await submit_guess(self.browser.page)
        if not submitted:
            print("    [WARN] Could not submit guess, trying keyboard Enter")
            await self.browser.page.keyboard.press("Enter")
            await asyncio.sleep(random.uniform(0.5, 1.0))

        await asyncio.sleep(0.8)
        await self.browser.page.screenshot(path=str(capture_dir / "results.png"))

        results = await self.state_machine.poll_results_content(
            self.browser.page, timeout_ms=3000, poll_ms=100
        )

        if results and results.get("distance") is not None:
            distance = float(results["distance"])
            print(f"    Distance: {distance:.1f} km")
        else:
            distance = 10000.0
            print("    [WARN] Could not read score")

        true_coords = pano_truth
        if (not true_coords
                and results
                and results.get("lat") is not None
                and results.get("lng") is not None):
            true_coords = (float(results["lat"]), float(results["lng"]))
        if not true_coords:
            true_coords = await self.state_machine.read_true_coordinates(self.browser.page)
        if not true_coords:
            for data in self.state_machine._intercepted_data:
                coords = self.state_machine._extract_coords_from_state(data)
                if coords:
                    true_coords = coords
                    break
        if true_coords:
            true_lat, true_lng = true_coords
        else:
            true_lat, true_lng = pred_lat, pred_lng
            print("    [WARN] Could not read true coordinates")

        primary_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        entry = ReplayEntry(
            image=primary_img,
            true_lat=true_lat,
            true_lng=true_lng,
            true_country="Unknown",
            guess_lat=pred_lat,
            guess_lng=pred_lng,
            distance_km=distance,
            round_id=f"round_{round_num}",
        )

        return entry

    async def run_session(self, num_rounds: Optional[int] = None) -> dict:
        """Run a self-play session of N rounds."""
        num_rounds = num_rounds or self.cfg.dpo.collect_rounds_per_session

        print(f"\n{'='*60}")
        print(f"Self-play session: {num_rounds} rounds")
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
                print("Starting game...")
                started = await start_game(self.browser.page)
                if not started:
                    print("[ERROR] Could not start game")
                    return {"error": "could_not_start"}
                await asyncio.sleep(2)

            if await self._ensure_game_started():
                await asyncio.sleep(2)
            else:
                print("[WARN] Could not start game — will retry each round")
                await self.browser.dump_page_html(
                    self.screenshot_dir / "debug_start_failed.html"
                )

            self._running = True
            session_entries = []

            for r in range(1, num_rounds + 1):
                try:
                    entry = await self._play_round(r)
                    if entry:
                        self.buffer.add(entry)
                        session_entries.append(entry)
                        self._rounds_played += 1
                        self._daily_rounds += 1

                    await go_to_next_round(self.browser.page)
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                    if self._daily_rounds >= self.cfg.game.max_daily_rounds:
                        print("  Daily round limit reached, pausing")
                        break

                except Exception as e:
                    print(f"  [ERROR] Round {r}: {e}")
                    continue

        finally:
            await self.browser.stop()

        distances = [e.distance_km for e in session_entries]
        stats = {
            "rounds_played": len(session_entries),
            "mean_distance_km": np.mean(distances) if distances else 0,
            "median_distance_km": np.median(distances) if distances else 0,
            "min_distance_km": np.min(distances) if distances else 0,
            "max_distance_km": np.max(distances) if distances else 0,
            "buffer_size": len(self.buffer),
        }

        print(f"\nSession complete: {stats}")
        return stats

    def get_buffer_stats(self) -> dict:
        return self.buffer.stats

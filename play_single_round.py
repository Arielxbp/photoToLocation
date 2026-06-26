#!/usr/bin/env python3
"""
Single-round OpenGuessr player with Grad-CAM heatmap visualization:
  1. Launches Playwright browser to openguessr.com
  2. Dismisses cookie popups / waits for loading
  3. Takes a screenshot of the Street View
  4. Runs the trained GeoLocator model — predicts country + coordinates
  5. Generates a Grad-CAM heatmap overlay showing what the model focused on
  6. Hovers on the minimap to expand it into the big world map
  7. Clicks on the predicted location on the big map
  8. Submits the guess

NOTE on coordinates:
  The region head was trained with a broken mapping (all region_idx = 0),
  so region-based coordinate prediction is stuck at S2 cell 0.
  This script uses country centroids + jitter for click positioning
  (same approach as SelfPlayLoop._infer_location).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
from pathlib import Path
import re
import httpx
import numpy as np
import torch
from PIL import Image

from geoguessr_agent.game.actions import (
    click_guess_on_map,
    dismiss_cookie_banner,
    hover_minimap_to_expand,
    submit_guess,
    place_on_map
)
from geoguessr_agent.game.browser import OpenGuessrBrowser
from geoguessr_agent.game.state_machine import GameState, GameStateMachine
from geoguessr_agent.inference.gradcam import generate_annotated_heatmap
from geoguessr_agent.model.geolocator import GeoLocator
from geoguessr_agent.geoutils import get_country_centroids

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path("checkpoints/best_model.pth")
INDICES_PATH = Path("data/indices.json")
SCREENSHOT_DIR = Path("data/screenshots")

RNG = random.Random()


# ===================================================================
# Preprocessing (mirrors InferencePipeline / SelfPlayLoop)
# ===================================================================

def preprocess_image(img: Image.Image, device: str) -> torch.Tensor:
    """Resize to 320x180 and normalise with ImageNet stats. Returns [1,3,180,320]."""
    img = img.convert("RGB").resize((320, 180), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = torch.from_numpy((arr - mean) / std).float()
    return tensor.unsqueeze(0).to(device), img


# ===================================================================
# Model loading & inference
# ===================================================================

def build_kaggle_country_points() -> dict[str, list[tuple[float, float]]]:
    """
    Parse Kaggle region data to build a lookup: country_name → list of (lat, lng)
    using real training-data region midpoints.  Guarantees on-land coordinates.

    Normalises country names with the same mapping used during training
    (CountryMapper.COUNTRY_NAME_NORMALIZE) so model country predictions match
    Kaggle region prefixes.
    """
    from collections import defaultdict
    from geoguessr_agent.data.mapper import COUNTRY_NAME_NORMALIZE

    def norm(c: str) -> str:
        return COUNTRY_NAME_NORMALIZE.get(c, c)

    kaggle_dir = Path("data/kaggle")
    with open(kaggle_dir / "region_to_index.json") as f:
        r2i = json.load(f)
    with open(kaggle_dir / "region_index_to_middle_point.json") as f:
        ri2mp = json.load(f)

    country_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for region_name, ridx in r2i.items():
        country = norm(region_name.split("_")[0])
        ridx_str = str(ridx)
        if ridx_str in ri2mp:
            latlng = ri2mp[ridx_str]
            country_points[country].append((float(latlng[0]), float(latlng[1])))

    return dict(country_points)


def load_model_and_indices(device: str):
    """Load the GeoLocator model, country index mappings, and Kaggle points."""
    with open(INDICES_PATH) as f:
        indices = json.load(f)

    idx_to_country = {int(k): v for k, v in indices["idx_to_country"].items()}
    idx_to_continent = {int(k): v for k, v in indices["idx_to_continent"].items()}
    country_index = indices["country_index"]

    model = GeoLocator.load(str(MODEL_PATH), device=device)
    model.eval()

    country_centroids = get_country_centroids(country_index)
    kaggle_points = build_kaggle_country_points()

    return model, idx_to_country, idx_to_continent, country_centroids, kaggle_points


@torch.no_grad()
def infer_location(
    model: GeoLocator,
    input_tensor: torch.Tensor,
    idx_to_country: dict[int, str],
    idx_to_continent: dict[int, str],
    country_centroids: torch.Tensor,
    kaggle_points: dict[str, list[tuple[float, float]]],
    device: str,
) -> dict:
    """
    Single-image inference.  Coordinates come from Kaggle training-data
    regional midpoints (so they always land on real land).  Falls back to
    country centroid with small jitter if the country is missing from Kaggle.
    """
    tensor = input_tensor.to(device)
    outputs = model(tensor)

    # --- country ---
    country_probs = torch.softmax(outputs["country_logits"], dim=-1)
    top5_conf, top5_idx = torch.topk(country_probs, k=5, dim=-1)
    top5_conf = top5_conf.squeeze(0).cpu().tolist()
    top5_idx = top5_idx.squeeze(0).cpu().tolist()

    country_idx = top5_idx[0]
    country_conf = top5_conf[0]
    top_country = idx_to_country.get(country_idx, f"unknown_{country_idx}")

    top5_countries = [
        {"country": idx_to_country.get(i, f"unknown_{i}"), "confidence": float(c)}
        for i, c in zip(top5_idx, top5_conf)
    ]

    # --- continent ---
    continent_probs = torch.softmax(outputs["continent_logits"], dim=-1)
    continent_idx = continent_probs.argmax(-1).item()
    continent = idx_to_continent.get(continent_idx, f"unknown_{continent_idx}")

    # --- coordinates via Kaggle regional midpoints ---
    lat, lng = _pick_coordinate(top_country, country_idx, country_centroids, kaggle_points)

    return {
        "latitude": lat,
        "longitude": lng,
        "country": top_country,
        "country_idx": country_idx,
        "country_confidence": country_conf,
        "continent": continent,
        "top5_countries": top5_countries,
    }


def _pick_coordinate(
    country_name: str,
    country_idx: int,
    country_centroids: torch.Tensor,
    kaggle_points: dict[str, list[tuple[float, float]]],
) -> tuple[float, float]:
    """
    Try Kaggle regional midpoints first (guaranteed on-land).
    Fall back to country centroid with ±1° jitter.
    """
    points = kaggle_points.get(country_name)
    if points:
        lat, lng = RNG.choice(points)
        print(f"  (picked Kaggle region mid-point: {lat:.4f}, {lng:.4f})")
        return (round(lat, 4), round(lng, 4))

    centroid = country_centroids[country_idx].cpu()
    lat = float(torch.rad2deg(centroid[0]).item())
    lng = float(torch.rad2deg(centroid[1]).item())
    if lat != 0.0 or lng != 0.0:
        lat += RNG.uniform(-1.0, 1.0)
        lng += RNG.uniform(-1.0, 1.0)
    lat = max(-85.0, min(85.0, lat))
    lng = max(-180.0, min(180.0, lng))
    print(f"  (Kaggle country not found, using centroid + jitter: {lat:.4f}, {lng:.4f})")
    return (round(lat, 4), round(lng, 4))


# ===================================================================
# Main round player
# ===================================================================

API_KEY=os.environ.get("MAPS_API", "")
IMAGE_SIZE="1280x1280"
PANO_REGEX = re.compile(r"panoid=([^&]+)")

async def access_with_api_key(browser, device, state_machine):
    intercept_pano_id = None
    # ---- launch ----
    print("Launching browser...")
    await browser.start()
    print("Browser launched. Waiting for page to settle...")
    await asyncio.sleep(2)

    def handle_request(request):
        nonlocal intercept_pano_id
        url = request.url
        #print(f"Analyzing request with url: {url}\n")
        if "streetviewpixels" in url and "panoid" in url:
            match = PANO_REGEX.search(url)
            if match and not intercept_pano_id:
                intercept_pano_id = match.group(1)
                print(f"[Network Intercept] Successfully captured Pano ID: {intercept_pano_id}")


    browser.page.context.on("request", handle_request)
    # ---- cookie ----
    print("Handling cookie consent...")
    dismissed = await dismiss_cookie_banner(browser.page)
    if dismissed:
        print("  Cookie banner dismissed")
    await asyncio.sleep(1.5)

    # ---- detect state / start game ----
    state = await state_machine.detect_state(browser.page)
    print(f"Detected state: {state}")

    if state == GameState.MAIN_MENU:
        print("On main menu — starting game...")
        from geoguessr_agent.game.actions import start_game
        started = await start_game(browser.page)
        if not started:
            print("[ERROR] Could not start the game.")
            return
        await asyncio.sleep(2)

    # ---- wait for Street View ----
    print("Waiting for Street View to appear...")
    try:
        await browser.page.wait_for_selector(
            '#panorama-iframe, .gm-style, [class*="panorama"], '
            '[class*="street-view"], .leaflet-container, #map',
            timeout=20_000,
        )
    except Exception:
        print("[WARN] Timed out waiting for Street View")
        await browser.page.screenshot(path=str(SCREENSHOT_DIR / "debug_no_sv.png"))
        return

    await asyncio.sleep(3)

    # ---- clear loading spinner ----
    print("Waiting for loading screen to clear...")
    for _ in range(10):
        loading = await browser.page.evaluate("""
        () => {
            const spinners = document.querySelectorAll(
                '[class*="loading"], [class*="spinner"], .compass_spinner, [role="progressbar"]'
            );
            for (const s of spinners) {
                if (s.offsetParent !== null) return true;
            }
            return false;
        }
        """)
        if not loading:
            break
        await asyncio.sleep(1.0)
    else:
        print("  (loading spinner may still be present, continuing)")

    print("Scanning browser network cache for Pano ID...")
    await asyncio.sleep(1.0) 

    # If the live listener missed it, extract it from the historical request log
    if not intercept_pano_id:
        all_requests = browser.page.context.background_pages + [browser.page]

        for frame in browser.page.frames:
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
                        print(f"[Cache Recovery] Successfully recovered Pano ID from frame logs: {intercept_pano_id}")
                        break
            except Exception:
                continue

    try:
        browser.page.context.remove_listener("request", handle_request)
    except Exception:
        pass

    if not intercept_pano_id:
        print("[ERROR] Critical Timeout: The Pano ID could not be located in network history.")
        await browser.page.screenshot(path=str(SCREENSHOT_DIR / "debug_timeout_state.png"))
        return

    url = f"https://maps.googleapis.com/maps/api/streetview?size={IMAGE_SIZE}&pano={intercept_pano_id}&key={API_KEY}"

    print(f"Fetching clean imagery for Pano: {intercept_pano_id}...")
    
    # 2. Fetch the image asynchronously over HTTP
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching API: {response.status_code}")
            return None

        img_bytes = response.content
        print(f"  Clean image downloaded → ({len(img_bytes)} bytes)")

    # 3. Pass the clean bytes directly to PIL (No UI noise)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_image.save(SCREENSHOT_DIR / "round_streetview.png", quality=95)
    
    # 4. Run your preprocessing and inference pipeline
    input_tensor, resized_image = preprocess_image(pil_image, device)
    return input_tensor, pil_image

async def access_without_api(browser, device, state_machine):
    # ---- launch ----
    print("Launching browser...")
    await browser.start()
    print("Browser launched. Waiting for page to settle...")
    await asyncio.sleep(2)

    # ---- cookie ----
    print("Handling cookie consent...")
    dismissed = await dismiss_cookie_banner(browser.page)
    if dismissed:
        print("  Cookie banner dismissed")
    await asyncio.sleep(1.5)

    # ---- detect state / start game ----
    state = await state_machine.detect_state(browser.page)
    print(f"Detected state: {state}")

    if state == GameState.MAIN_MENU:
        print("On main menu — starting game...")
        from geoguessr_agent.game.actions import start_game
        started = await start_game(browser.page)
        if not started:
            print("[ERROR] Could not start the game.")
            return
        await asyncio.sleep(2)

    # ---- wait for Street View ----
    print("Waiting for Street View to appear...")
    try:
        await browser.page.wait_for_selector(
            '#panorama-iframe, .gm-style, [class*="panorama"], '
            '[class*="street-view"], .leaflet-container, #map',
            timeout=20_000,
        )
    except Exception:
        print("[WARN] Timed out waiting for Street View")
        await browser.page.screenshot(path=str(SCREENSHOT_DIR / "debug_no_sv.png"))
        return

    await asyncio.sleep(3)

    # ---- clear loading spinner ----
    print("Waiting for loading screen to clear...")
    for _ in range(10):
        loading = await browser.page.evaluate("""
        () => {
            const spinners = document.querySelectorAll(
                '[class*="loading"], [class*="spinner"], .compass_spinner, [role="progressbar"]'
            );
            for (const s of spinners) {
                if (s.offsetParent !== null) return true;
            }
            return false;
        }
        """)
        if not loading:
            break
        await asyncio.sleep(1.0)
    else:
        print("  (loading spinner may still be present, continuing)")

    # ---- screenshot ----
    print("Capturing Street View screenshot...")
    screenshot_path = SCREENSHOT_DIR / "round_streetview.png"
    img_bytes = await browser.page.screenshot(path=str(screenshot_path), full_page=False)
    print(f"  Screenshot saved → {screenshot_path}  ({len(img_bytes)} bytes)")

    # ---- preprocess for model ----
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor, resized_image = preprocess_image(pil_image, device)
    return input_tensor, pil_image

async def play_one_round():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model, idx_to_country, idx_to_continent, country_centroids, kaggle_points = load_model_and_indices(device)
    country_centroids = country_centroids.to(device)
    n_kaggle = len(kaggle_points)
    total_regions = sum(len(v) for v in kaggle_points.values())
    print(f"Model loaded.  Kaggle: {n_kaggle} countries, {total_regions} regions.")

    browser = OpenGuessrBrowser(
        url="https://openguessr.com/",
        headless=False,
        stealth=True,
        viewport_width=1280,
        viewport_height=720,
    )
    state_machine = GameStateMachine()
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        input_tensor, pil_image = await access_with_api_key(browser, device, state_machine) if API_KEY != "" else await access_without_api(browser, device, state_machine)

        print("Running model inference...")
        result = infer_location(
            model, input_tensor,
            idx_to_country, idx_to_continent, country_centroids, kaggle_points, device,
        )
    

        pred_lat = result["latitude"]
        pred_lng = result["longitude"]
        pred_country = result["country"]
        pred_conf = result["country_confidence"]
        pred_country_idx = result["country_idx"]
        continent = result["continent"]

        print(f"  Location:  ({pred_lat:.4f}, {pred_lng:.4f})")
        print(f"  Country:   {pred_country}  ({pred_conf:.1%})")
        print(f"  Continent: {continent}")
        print(f"  Top-5:")
        for i, e in enumerate(result["top5_countries"]):
            marker = " <--" if i == 0 else ""
            print(f"    {i + 1}. {e['country']:30s} {e['confidence']:.2%}{marker}")

        # ---- Grad-CAM heatmap ----
        print("Computing Grad-CAM heatmap...")
        annotated = generate_annotated_heatmap(
            model=model,
            input_tensor=input_tensor,
            original_image=pil_image,
            target_class=pred_country_idx,
            country_name=pred_country,
            confidence=pred_conf,
            continent=continent,
            lat=pred_lat,
            lng=pred_lng,
            device=device,
        )
        if annotated:
            heatmap_path = SCREENSHOT_DIR / "round_heatmap.jpg"
            annotated.save(str(heatmap_path), quality=95)
            print(f"  Heatmap saved → {heatmap_path}")
        else:
            print("  [WARN] Heatmap generation failed (scipy missing?)")

        print("Placing guess on the big map...")
        await place_on_map(browser.page, pred_lat, pred_lng)

        await asyncio.sleep(5.0)  # wait for map to update

        # ---- submit ----
        print("Submitting guess...")
        submitted = await submit_guess(browser.page)
        if submitted:
            print("  ✅ Guess submitted!")
        else:
            print("  [WARN] Could not find submit button, pressing Enter...")
            await browser.page.keyboard.press("Enter")

        await asyncio.sleep(1.0)

        # Screenshot results
        results_path = SCREENSHOT_DIR / "round_results.png"
        await browser.page.screenshot(path=str(results_path))
        print(f"  Results screenshot → {results_path}")

        distance_text = await browser.page.evaluate("""
        () => {
            const text = document.body.innerText || '';
            const m = text.match(/([\\d,.]+)\\s*km/i);
            return m ? m[0] : null;
        }
        """)
        if distance_text:
            print(f"  Score: {distance_text}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nClosing browser...")
        await browser.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(play_one_round())

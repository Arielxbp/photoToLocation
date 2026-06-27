from __future__ import annotations

import asyncio
import base64
import random
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import io
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from ..data.panorama import (
    PanoramaInfo,
    TileCollector,
    stitch_panorama,
    generate_view_crops,
    panorama_to_inference_image,
)


def _image_has_content(data: bytes, min_mean: float = 15.0) -> bool:
    """Check whether an image has real content (not blank/black)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        return arr.mean() > min_mean
    except Exception:
        return False


class OpenGuessrBrowser:
    """Playwright-based browser automation for OpenGuessr."""

    def __init__(
        self,
        url: str = "https://openguessr.com/",
        headless: bool = False,
        stealth: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        self.url = url
        self.headless = headless
        self.stealth = stealth
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._tile_collector: Optional[TileCollector] = None
        self._tile_interception_active = False

    async def start(self) -> Page:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )

        context_options = {
            "viewport": self.viewport,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "locale": "en-US",
        }

        if self.stealth:
            context_options["bypass_csp"] = True
            context_options["extra_http_headers"] = {
                "Accept-Language": "en-US,en;q=0.9",
            }

        self._context = await self._browser.new_context(**context_options)

        if self.stealth:
            await self._context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                delete navigator.__proto__.webdriver;
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)

        # Capture Leaflet map instances as they are created so the guess map can
        # be reset to the world view and projected with the map's own API. Works
        # when Leaflet is exposed as window.L; harmless otherwise.
        await self._context.add_init_script("""
            (() => {
                if (window.__ogLeafletHookInstalled) return;
                window.__ogLeafletHookInstalled = true;
                window.__ogMaps__ = [];
                const hook = (L) => {
                    try {
                        if (L && L.Map && L.Map.prototype && !L.Map.__ogHooked) {
                            const orig = L.Map.prototype.initialize;
                            L.Map.prototype.initialize = function() {
                                try { window.__ogMaps__.push(this); } catch (e) {}
                                return orig.apply(this, arguments);
                            };
                            L.Map.__ogHooked = true;
                        }
                    } catch (e) {}
                };
                let _L = window.L;
                if (_L) hook(_L);
                try {
                    Object.defineProperty(window, 'L', {
                        configurable: true,
                        get() { return _L; },
                        set(v) { _L = v; hook(v); },
                    });
                } catch (e) {}
            })();
        """)

        self._page = await self._context.new_page()
        await self._page.goto(self.url, wait_until="networkidle", timeout=30_000)
        return self._page

    async def setup_tile_interception(self) -> TileCollector:
        """
        Install route interception for Street View tile requests.

        Intercepts calls to Google's Street View tile servers and collects
        the raw tile bytes keyed by (panoid, zoom, x, y) via a TileCollector.
        """
        if self._tile_interception_active:
            return self._tile_collector

        self._tile_collector = TileCollector()

        async def handle_route(route):
            url = route.request.url
            if self._tile_collector.is_tile_request(url):
                entry = self._tile_collector.parse_tile_request(url)
                try:
                    response = await route.fetch()
                    body = await response.body()
                    if entry and body:
                        await self._tile_collector.add_tile(
                            entry.panoid, entry.zoom, entry.x, entry.y, body
                        )
                    await route.fulfill(response=response)
                except Exception:
                    await route.continue_()
            else:
                await route.continue_()

        await self.page.route("**/*", handle_route)
        self._tile_interception_active = True
        return self._tile_collector

    async def capture_panorama_tiles(
        self,
        wait_timeout: float = 6.0,
    ) -> Optional[PanoramaInfo]:
        """
        Wait for tile data to accumulate and return a PanoramaInfo ready for
        stitching. Returns None if no tiles were collected within the timeout.
        """
        if not self._tile_collector:
            return None

        await asyncio.sleep(0.5)

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            active = self._tile_collector.active_panoids
            if active:
                panoid = active[-1]
                pano = await self._tile_collector.get_panorama(panoid, timeout=1.0)
                if pano and pano.tiles:
                    return pano
            await asyncio.sleep(0.3)

        active = self._tile_collector.active_panoids
        if active:
            return await self._tile_collector.get_panorama(active[-1], timeout=0.5)
        return None

    async def clear_tiles(self) -> None:
        if self._tile_collector:
            await self._tile_collector.clear()

    async def capture_panorama_image(
        self, debug_dir: Optional[Path] = None
    ) -> Optional[tuple[list[Image.Image], str]]:
        """
        Capture the Street View as perspective crops via tile interception.

        Strategy:
        1. Wait for tile interception to accumulate pano tiles
        2. Stitch into equirectangular
        3. Generate perspective crops at 5 horizontal headings
        4. Fall back to single-image capture if tiles unavailable

        Returns (list_of_PIL_crops, debug_log) or None.
        """
        debug_lines = []

        iframe_src = await self._read_iframe_src()
        params = await self._read_iframe_params()

        base_heading = 0.0
        if params:
            _, _, base_heading, _ = params

        pano = await self.capture_panorama_tiles(wait_timeout=3.0)
        if pano and pano.tiles:
            debug_lines.append(
                f"Tile capture: panoid={pano.panoid} tiles={len(pano.tiles)} "
                f"zoom={pano.zoom} grid={pano.cols}x{pano.rows}"
            )

            equi = stitch_panorama(pano)
            if equi:
                debug_lines.append(f"Stitched: {equi.size[0]}x{equi.size[1]}")
                if debug_dir:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    equi.save(debug_dir / "panorama_equi.jpg", quality=92)

                crops = generate_view_crops(
                    equi, base_heading=base_heading, n_horizontal=5,
                    fov_deg=90.0, out_size=(640, 640),
                )
                crop_images = [c[0] for c in crops]
                debug_lines.append(f"Generated {len(crop_images)} perspective crops")
                if debug_dir:
                    for i, (crop, h, p) in enumerate(crops):
                        crop.save(
                            debug_dir / f"crop_{i}_h{h:.0f}_p{p:.0f}.jpg",
                            quality=92,
                        )

                log = "\n".join(debug_lines)
                return crop_images, log

            debug_lines.append("Stitch failed — falling back")

        debug_lines.append("Tile capture unavailable — falling back to API")
        result = await self.capture_streetview_image(debug_dir=debug_dir)
        if result is None:
            return None
        img_bytes, sub_log = result
        debug_lines.append(f"API fallback: {sub_log}")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        log = "\n".join(debug_lines)
        return [img], log

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    async def screenshot(self, path: str | Path, full_page: bool = False) -> bytes:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = await self.page.screenshot(path=str(path), full_page=full_page)
        return data

    async def capture_streetview_image(
        self, debug_dir: Optional[Path] = None
    ) -> Optional[tuple[bytes, str]]:
        """
        Capture the Street View image for model inference.
        Returns (image_bytes, debug_log).

        Strategy:
        1. Extract iframe URL from page, parse location/heading/key
        2. Download clean image via Google Street View Image API
           (through the browser's fetch to preserve referrer)
        3. Fall back to element screenshot
        """
        debug_lines = []

        iframe_src = await self._read_iframe_src()
        if iframe_src:
            debug_lines.append(f"iframe src: {iframe_src}")

        params = await self._read_iframe_params()
        if params:
            lat, lng, heading, api_key = params
            debug_lines.append(
                f"Parsed: lat={lat:.6f} lng={lng:.6f} heading={heading:.2f} key={api_key[:20]}..."
            )

            api_url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"?location={lat},{lng}"
                f"&size=640x360"
                f"&heading={heading}"
                f"&fov=90"
                f"&key={api_key}"
            )
            debug_lines.append(f"API URL: {api_url}")

            img_bytes = await self._fetch_via_browser(api_url)
            if img_bytes:
                debug_lines.append(f"Browser fetch OK: {len(img_bytes)} bytes")
                log = "\n".join(debug_lines)
                if debug_dir:
                    self._save_debug(debug_dir, log, img_bytes)
                return img_bytes, log

            img_bytes = await self._fetch_via_python(api_url)
            if img_bytes:
                debug_lines.append(f"Python fetch OK: {len(img_bytes)} bytes")
                log = "\n".join(debug_lines)
                if debug_dir:
                    self._save_debug(debug_dir, log, img_bytes)
                return img_bytes, log

            debug_lines.append("API fetch failed (both browser and python)")

        else:
            debug_lines.append("No iframe params found")

        debug_lines.append("Falling back to element screenshot")
        img_bytes = await self._screenshot_panorama_element()
        if img_bytes:
            debug_lines.append(f"Screenshot OK: {len(img_bytes)} bytes")
        else:
            debug_lines.append("Screenshot FAILED")

        log = "\n".join(debug_lines)
        if debug_dir:
            self._save_debug(debug_dir, log, img_bytes)
        return (img_bytes, log) if img_bytes else (None, log)

    def _save_debug(self, debug_dir: Path, log: str, img_bytes: Optional[bytes]) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "capture_debug.txt").write_text(log)
        if img_bytes:
            (debug_dir / "captured_image.png").write_bytes(img_bytes)

    async def _read_iframe_src(self) -> Optional[str]:
        try:
            return await self.page.evaluate(
                "() => document.querySelector('#panorama-iframe')?.src || null"
            )
        except Exception:
            return None

    async def read_pano_location(self) -> Optional[tuple[float, float]]:
        """
        Return the true (lat, lng) of the currently displayed Street View
        panorama, parsed from the embed iframe's ``location`` parameter.

        This is OpenGuessr's ground-truth location for the active round. It is
        only available while the panorama is on screen (i.e. before the guess
        is submitted), so it must be read during the round, not from the
        results screen. Unlike ``_read_iframe_params`` it does not require the
        API key to be present.
        """
        try:
            src = await self._read_iframe_src()
            if not src:
                return None
            location = parse_qs(urlparse(src).query).get("location", [None])[0]
            if not location:
                return None
            parts = location.split(",")
            if len(parts) < 2:
                return None
            return (float(parts[0].strip()), float(parts[1].strip()))
        except Exception:
            return None

    async def _read_iframe_params(self) -> Optional[tuple[float, float, float, str]]:
        try:
            src = await self._read_iframe_src()
            if not src:
                return None

            parsed = urlparse(src)
            params = parse_qs(parsed.query)

            location = params.get("location", [None])[0]
            if not location:
                return None

            parts = location.split(",")
            if len(parts) < 2:
                return None

            lat = float(parts[0].strip())
            lng = float(parts[1].strip())

            heading_val = params.get("heading", [None])[0]
            heading = float(heading_val) if heading_val else 0.0

            key = params.get("key", [None])[0]
            if not key:
                key_match = re.search(r"key=([^&\s]+)", src)
                key = key_match.group(1) if key_match else None

            if key:
                return (lat, lng, heading, key)

        except Exception:
            pass

        return None

    async def _fetch_via_browser(self, url: str) -> Optional[bytes]:
        """Download image through the browser's fetch (preserves referrer)."""
        try:
            b64 = await self.page.evaluate("""
            async (url) => {
                try {
                    const resp = await fetch(url, { cache: 'no-cache' });
                    if (!resp.ok) return null;
                    const blob = await resp.blob();
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result);
                        reader.readAsDataURL(blob);
                    });
                } catch(e) { return null; }
            }
            """, url)
            if b64 and isinstance(b64, str) and "," in b64:
                return base64.b64decode(b64.split(",", 1)[1])
        except Exception:
            pass
        return None

    @staticmethod
    def _fetch_url(url: str) -> Optional[bytes]:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 512:
                    return data
        except Exception:
            pass
        return None

    async def _fetch_via_python(self, url: str) -> Optional[bytes]:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._fetch_url, url)
        except Exception:
            return None

    async def _screenshot_panorama_element(self, crop_ui: bool = True) -> Optional[bytes]:
        selectors = [
            "#panorama-iframe",
            ".gm-style",
            '[class*="panorama"]',
            '[class*="street-view"]',
            '[class*="StreetView"]',
        ]
        screenshot_bytes = None
        for sel in selectors:
            try:
                el = await self.page.wait_for_selector(sel, timeout=2000)
                if el:
                    screenshot_bytes = await el.screenshot()
                    if screenshot_bytes and len(screenshot_bytes) > 512:
                        break
            except Exception:
                continue

        if screenshot_bytes is None:
            screenshot_bytes = await self.page.screenshot()

        if not screenshot_bytes or len(screenshot_bytes) <= 512:
            return None

        if crop_ui:
            try:
                img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
                w, h = img.size
                crop_margin = 0.10
                left = int(w * crop_margin)
                top = int(h * crop_margin)
                right = int(w * (1 - crop_margin))
                bottom = int(h * (1 - crop_margin))
                if right > left and bottom > top:
                    cropped = img.crop((left, top, right, bottom))
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    screenshot_bytes = buf.getvalue()
            except Exception:
                pass

        if not _image_has_content(screenshot_bytes):
            screenshot_bytes = await self.page.screenshot()
            if crop_ui:
                try:
                    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
                    w, h = img.size
                    left = int(w * 0.10)
                    top = int(h * 0.10)
                    right = int(w * 0.90)
                    bottom = int(h * 0.90)
                    cropped = img.crop((left, top, right, bottom))
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    screenshot_bytes = buf.getvalue()
                except Exception:
                    pass

        return screenshot_bytes if (screenshot_bytes and len(screenshot_bytes) > 512) else None

    async def click(self, x: int, y: int, delay_ms: Optional[int] = None) -> None:
        if delay_ms is None:
            delay_ms = random.randint(200, 500)
        await self.page.mouse.click(x, y, delay=delay_ms)
        await asyncio.sleep(random.uniform(0.1, 0.3))

    async def click_with_human_timing(self, x: int, y: int) -> None:
        start_x = random.randint(100, self.viewport["width"] - 100)
        start_y = random.randint(100, self.viewport["height"] - 100)
        await self.page.mouse.move(start_x, start_y)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        steps = random.randint(5, 15)
        for i in range(steps):
            t = (i + 1) / steps
            ix = int(start_x + (x - start_x) * t + random.uniform(-5, 5))
            iy = int(start_y + (y - start_y) * t + random.uniform(-5, 5))
            await self.page.mouse.move(ix, iy)
            await asyncio.sleep(random.uniform(0.01, 0.04))

        await asyncio.sleep(random.uniform(0.05, 0.2))
        await self.page.mouse.click(x, y)
        await asyncio.sleep(random.uniform(0.1, 0.3))

    async def type_text(self, text: str, delay_ms: int = 100) -> None:
        for char in text:
            await self.page.keyboard.press(char)
            await asyncio.sleep(random.uniform(0.05, delay_ms / 1000))

    async def wait_for_selector_safe(
        self, selectors: list[str], timeout: int = 10000
    ) -> Optional[str]:
        for selector in selectors:
            try:
                await self.page.wait_for_selector(selector, timeout=timeout)
                return selector
            except Exception:
                continue
        return None

    async def dump_page_html(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(await self.page.content())

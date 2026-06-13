from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class GameState(Enum):
    LOADING = auto()
    MAIN_MENU = auto()
    ROUND_ACTIVE = auto()
    GUESS_SUBMITTED = auto()
    RESULTS = auto()
    GAME_OVER = auto()
    ERROR = auto()


@dataclass
class RoundInfo:
    game_id: str
    round_number: int
    true_lat: Optional[float] = None
    true_lng: Optional[float] = None
    true_country: Optional[str] = None
    score: Optional[float] = None
    distance_km: Optional[float] = None


class GameStateMachine:
    """Detects and tracks the game state in OpenGuessr."""

    STATE_SELECTORS: dict[GameState, list[str]] = {
        GameState.LOADING: [
            '[class*="loading"]',
            '[class*="spinner"]',
            ".compass_spinner",
            '[role="progressbar"]',
            "img[alt*='Loading']",
        ],
        GameState.MAIN_MENU: [
            '[class*="start"]',
            "button:has-text('Play')",
            "button:has-text('Start')",
            "a:has-text('Play')",
            '[class*="menu"]',
        ],
        GameState.ROUND_ACTIVE: [
            '[class*="guess-map"]',
            ".leaflet-container",
            "#map",
            '[data-testid="guess-map"]',
            "canvas[class*='map']",
        ],
        GameState.RESULTS: [
            '[class*="result"]',
            '[class*="score"]',
            "text=/\\d+\\.?\\d*\\s*km/i",
            "text=/\\d+,\\d+\\s*points/i",
        ],
    }

    def __init__(self):
        self.current_state = GameState.LOADING
        self.round_number = 0
        self._intercepted_data: list = []
        self._current_panoid: Optional[str] = None
        self._current_pano_zoom: int = 3

    async def detect_state(self, page) -> GameState:
        """Detect the current game state by checking DOM selectors."""
        query = """
        () => {
            const stateMap = {
                'LOADING': ['.loading', '.spinner', '.compass_spinner', '[role="progressbar"]'],
                'MAIN_MENU': ['button', 'a'],
                'RESULTS': ['.result', '.score', '.results'],
            };

            let bestState = 'ROUND_ACTIVE';
            let bestPriority = 0;

            // Check MAIN_MENU: look for Play/Start text on buttons/links
            const menuTexts = ['play', 'start', 'singleplayer', 'single player'];
            const allButtons = document.querySelectorAll('button, a');
            let menuMatch = false;
            for (const btn of allButtons) {
                const text = (btn.textContent || '').toLowerCase().trim();
                if (menuTexts.some(t => text.includes(t))) {
                    menuMatch = true;
                    break;
                }
            }
            if (menuMatch) {
                // Only set MAIN_MENU if we're NOT in a round (no map or pano visible)
                const mapEl = document.querySelector('.leaflet-container, [class*="guess-map"], #map, canvas[class*="map"]');
                const panoEl = document.querySelector('canvas, [class*="street-view"], .gm-style');
                if (!mapEl && !panoEl) {
                    bestState = 'MAIN_MENU';
                    bestPriority = 1;
                }
            }

            // Check RESULTS: look for result/score elements
            for (const sel of stateMap['RESULTS']) {
                try {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) {
                        bestState = 'RESULTS';
                        bestPriority = 3;
                        break;
                    }
                } catch(e) {}
            }

            // Also check body text for distance/score patterns
            if (bestState !== 'RESULTS') {
                const text = document.body.innerText || '';
                if (/\\d+[.,]?\\d*\\s*km/i.test(text) || /\\d+[.,]?\\d*\\s*points/i.test(text)) {
                    bestState = 'RESULTS';
                    bestPriority = 3;
                }
            }

            const mapEl = document.querySelector('.leaflet-container, [class*="guess-map"], #map');
            const panoEl = document.querySelector('canvas, [class*="street-view"], .gm-style');

            return {
                state: bestState,
                hasMap: !!mapEl,
                hasPano: !!panoEl,
            };
        }
        """

        result = await page.evaluate(query)
        state_name = result.get("state", "ROUND_ACTIVE")

        state_map = {
            "LOADING": GameState.LOADING,
            "MAIN_MENU": GameState.MAIN_MENU,
            "RESULTS": GameState.RESULTS,
            "ROUND_ACTIVE": GameState.ROUND_ACTIVE,
        }
        detected = state_map.get(state_name, GameState.ROUND_ACTIVE)

        if detected != self.current_state:
            if detected == GameState.ROUND_ACTIVE and self.current_state != GameState.ROUND_ACTIVE:
                self.round_number += 1
            self.current_state = detected

        return detected

    async def wait_for_state(
        self,
        page,
        target: GameState,
        timeout_ms: int = 15000,
        poll_ms: int = 500,
    ) -> bool:
        """Wait until the game reaches the target state."""
        elapsed = 0
        while elapsed < timeout_ms:
            state = await self.detect_state(page)
            if state == target:
                return True
            await asyncio.sleep(poll_ms / 1000)
            elapsed += poll_ms
        return False

    async def poll_results_content(
        self, page, timeout_ms: int = 8000, poll_ms: int = 150
    ) -> Optional[dict]:
        """
        Rapidly poll the page for results content (distance, score, coordinates).
        Returns a dict with distance, points, lat, lng, or None if not found.
        """
        elapsed = 0
        while elapsed < timeout_ms:
            result = await page.evaluate("""
            () => {
                const text = document.body.innerText || '';
                // Try multiple km patterns
                let kmMatch = text.match(/([\\d,.]+)\\s*km/i);
                if (!kmMatch) kmMatch = text.match(/([\\d,.]+)\\s*kilometers/i);
                if (!kmMatch) kmMatch = text.match(/distance[\\s:]*([\\d,.]+)/i);
                const ptsMatch = text.match(/([\\d,.]+)\\s*points/i);
                const coordMatch = text.match(
                    /(-?\\d+\\.\\d+)\\s*[°,;\\s]+\\s*(-?\\d+\\.\\d+)/
                );
                if (kmMatch || coordMatch) {
                    return {
                        distance: kmMatch ? parseFloat(kmMatch[1].replace(/,/g, '')) : null,
                        points: ptsMatch ? parseFloat(ptsMatch[1].replace(/,/g, '')) : null,
                        lat: coordMatch ? parseFloat(coordMatch[1]) : null,
                        lng: coordMatch ? parseFloat(coordMatch[2]) : null,
                    };
                }
                return null;
            }
            """)
            if result and (result.get("distance") is not None or result.get("lat") is not None):
                return result
            await asyncio.sleep(poll_ms / 1000)
            elapsed += poll_ms
        return None

    async def setup_response_interception(self, page) -> None:
        """Intercept JSON API responses to capture possible round data."""

        async def on_response(response):
            if not response.ok:
                return
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return
            try:
                data = await response.json()
                self._intercepted_data.append(data)
            except Exception:
                pass

        page.on("response", on_response)

    def clear_intercepted_data(self) -> None:
        """Clear intercepted network data between rounds."""
        self._intercepted_data.clear()
        self._current_panoid = None
        self._current_pano_zoom = 3

    async def extract_panoid_from_network(self) -> Optional[str]:
        """
        Scan intercepted API responses for a Street View panoid.
        Google's Street View API often embeds the panoid in metadata
        responses (gRPC-web JSON or protobuf).
        """
        for data in self._intercepted_data:
            panoid = self._dig_panoid(data)
            if panoid:
                self._current_panoid = panoid
                return panoid
        return self._current_panoid

    @staticmethod
    def _dig_panoid(data, depth: int = 0) -> Optional[str]:
        if depth > 12:
            return None

        if isinstance(data, dict):
            for key in ("panoid", "pano_id", "pano", "panorama_id"):
                if key in data and isinstance(data[key], str) and len(data[key]) >= 10:
                    return data[key]

            if "image" in data and isinstance(data["image"], dict):
                for key in ("panoid", "pano_id", "panorama_id"):
                    if key in data["image"] and isinstance(data["image"][key], str):
                        if len(data["image"][key]) >= 10:
                            return data["image"][key]

            if "panorama" in data and isinstance(data["panorama"], dict):
                pid = data["panorama"].get("panoid") or data["panorama"].get("id")
                if isinstance(pid, str) and len(pid) >= 10:
                    return pid

            for v in data.values():
                result = GameStateMachine._dig_panoid(v, depth + 1)
                if result:
                    return result

        elif isinstance(data, (list, tuple)):
            for item in data:
                result = GameStateMachine._dig_panoid(item, depth + 1)
                if result:
                    return result

        return None

    @property
    def current_panoid(self) -> Optional[str]:
        return self._current_panoid

    async def read_score(self, page) -> Optional[dict]:
        """Read the round score/result from the DOM."""
        result = await page.evaluate("""
        () => {
            const body = document.body.innerText;
            const kmRegex = /([\\d,.]+)\\s*km/i;
            const ptsRegex = /([\\d,.]+)\\s*points/i;
            const kmMatch = body.match(kmRegex);
            const ptsMatch = body.match(ptsRegex);

            const latRegex = /(-?\\d+\\.?\\d*)\\s*[°,]\\s*(-?\\d+\\.?\\d*)/;
            const coordMatch = body.match(latRegex);

            return {
                distance: kmMatch ? parseFloat(kmMatch[1].replace(',', '')) : null,
                points: ptsMatch ? parseFloat(ptsMatch[1].replace(',', '')) : null,
            };
        }
        """)
        return result

    async def read_true_coordinates(self, page) -> Optional[tuple[float, float]]:
        """Extract true coordinates from page state, network data, or DOM."""

        coords = await self._read_coords_from_network()
        if coords:
            return coords

        coords = await self._read_coords_from_globals(page)
        if coords:
            return coords

        coords = await self._read_coords_from_dom(page)
        if coords:
            return coords

        return None

    async def _read_coords_from_network(self) -> Optional[tuple[float, float]]:
        for data in self._intercepted_data:
            coords = self._extract_coords_from_state(data)
            if coords:
                return coords
        return None

    async def _read_coords_from_globals(self, page) -> Optional[tuple[float, float]]:
        try:
            data = await page.evaluate("""
            () => {
                const globals = [
                    '__NEXT_DATA__', '__NUXT__', '__INITIAL_STATE__',
                    '__DATA__', '__APOLLO_STATE__', '__REDUX_STATE__',
                    '__STORE__', '__GATSBY_STATE__', '__SVELTE_STORE__',
                ];
                for (const key of globals) {
                    if (window[key]) return JSON.stringify(window[key]);
                }
                return null;
            }
            """)
            if data:
                parsed = json.loads(data)
                coords = self._extract_coords_from_state(parsed)
                if coords:
                    return coords
        except Exception:
            pass
        return None

    async def _read_coords_from_dom(self, page) -> Optional[tuple[float, float]]:
        try:
            result = await page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script');
                for (const script of scripts) {
                    const text = script.textContent || '';
                    const m = text.match(/"lat"\\s*:\\s*(-?\\d+\\.?\\d*)/);
                    const m2 = text.match(/"lng"\\s*:\\s*(-?\\d+\\.?\\d*)/);
                    if (m && m2) {
                        return JSON.stringify({lat: parseFloat(m[1]), lng: parseFloat(m2[1])});
                    }
                }
                const body = document.body.innerText;
                const cm = body.match(
                    /(-?\\d+\\.?\\d*)\\s*[°,;\\s]+\\s*(-?\\d+\\.?\\d*)/
                );
                if (cm) {
                    return JSON.stringify({lat: parseFloat(cm[1]), lng: parseFloat(cm[2])});
                }
                return null;
            }
            """)
            if result:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "lat" in parsed:
                    return (float(parsed["lat"]), float(parsed["lng"]))
        except Exception:
            pass
        return None

    def _extract_coords_from_state(self, data, depth: int = 0) -> Optional[tuple[float, float]]:
        if depth > 10:
            return None

        if isinstance(data, dict):
            if "lat" in data and "lng" in data:
                lat = data.get("lat") or data.get("latitude")
                lng = data.get("lng") or data.get("longitude")
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    return (float(lat), float(lng))
            if "location" in data and isinstance(data["location"], dict):
                loc = data["location"]
                lat = loc.get("lat") or loc.get("latitude")
                lng = loc.get("lng") or loc.get("longitude")
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    return (float(lat), float(lng))
            for v in data.values():
                result = self._extract_coords_from_state(v, depth + 1)
                if result:
                    return result

        elif isinstance(data, (list, tuple)):
            for item in data:
                result = self._extract_coords_from_state(item, depth + 1)
                if result:
                    return result

            floats = [x for x in data if isinstance(x, (int, float))
                      and not isinstance(x, bool)]
            for i in range(len(floats) - 1):
                a, b = floats[i], floats[i + 1]
                if -90 <= a <= 90 and -180 <= b <= 180:
                    return (float(a), float(b))

        return None

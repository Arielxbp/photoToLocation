from __future__ import annotations

import asyncio
import math
import random
from typing import Optional

from playwright.async_api import Page


_MAP_BOUNDS = {"min_x": 100, "min_y": 100, "max_x": 1180, "max_y": 620}

async def place_on_map(page: Page, latitudine: float, longitudine: float):
    """
    Inietta ed esegue lo script JavaScript in Playwright per calcolare le 
    proporzioni di Mercatore sulla mappa di OpenGuessr e simulare il click.
    """
    
    # Script JavaScript da iniettare nella pagina
    js_script = """
    (async (targetLat, targetLng) => {
        // Trova il contenitore della mappa di Google (il box interattivo)
        const mapWrapper = document.querySelector('.leaflet-container') || document.querySelector('[id*="map"]');
        
        if (!mapWrapper) {
            return "Contenitore della mappa non trovato";
        }
        
        // Simula il passaggio del mouse per attivare l'animazione di ingrandimento
        mapWrapper.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
        mapWrapper.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
        
        // Attende che l'animazione CSS di ingrandimento si completi del tutto
        await new Promise(resolve => setTimeout(resolve, 400));
        
        const rect = mapWrapper.getBoundingClientRect();
        
        // Calcolo della proiezione di Mercatore globale adattata alle dimensioni correnti
        const percentX = (targetLng + 180) / 360;
        
        const latRad = (targetLat * Math.PI) / 180;
        const mercatorY = Math.log(Math.tan(Math.PI / 4 + latRad / 2));
        const maxMercatorY = Math.log(Math.tan(Math.PI / 4 + (85.0511 * Math.PI / 180) / 2));
        const percentY = 0.5 - (mercatorY / (2 * maxMercatorY));
        
        const clickX = rect.left + (rect.width * percentX);
        const clickY = rect.top + (rect.height * percentY);
        
        // Simula il click del mouse sul punto esatto calcolato
        const clickEvent = new MouseEvent('click', {
            clientX: clickX,
            clientY: clickY,
            bubbles: true,
            cancelable: true,
            view: window
        });
        
        mapWrapper.dispatchEvent(clickEvent);
        return `Click simulato a pixel: (${clickX.toFixed(2)}, ${clickY.toFixed(2)})`;
    })
    """
    
    try:
        # Passa lo script e gli argomenti (latitudine e longitudine) a Playwright
        risultato = await page.evaluate(f"({js_script})({latitudine}, {longitudine})")
        print(f"[Playwright Log]: {risultato}")
    except Exception as e:
        print(f"[Errore]: Impossibile eseguire lo script sulla pagina. {e}")

def _latlng_to_mercator(lat: float, lng: float) -> tuple[float, float]:
    x = (lng + 180) / 360
    lat_rad = math.radians(lat)
    y = 0.5 - math.log(math.tan(math.pi / 4 + lat_rad / 2)) / (2 * math.pi)
    return (x, y)


def coordinates_to_map_click(
    lat: float,
    lng: float,
    map_bounds: Optional[dict[str, int]] = None,
) -> tuple[int, int]:
    bounds = map_bounds or _MAP_BOUNDS
    x_merc, y_merc = _latlng_to_mercator(lat, lng)
    map_width = bounds["max_x"] - bounds["min_x"]
    map_height = bounds["max_y"] - bounds["min_y"]
    x = bounds["min_x"] + int(x_merc * map_width)
    y = bounds["min_y"] + int(y_merc * map_height)
    return (x, y)


async def reset_minimap_to_world(page) -> bool:
    """
    Reset the guess minimap to the default whole-world view.

    OpenGuessr keeps the same Leaflet map across rounds, so a round can start
    zoomed/panned to the previous result. When that happens the predicted
    coordinate is off-screen, so the guess click misses entirely or lands on
    the wrong place after being clamped to the map edge. Resetting to the world
    view guarantees every coordinate is visible before we click.

    Strategies (best-effort, in order):
      1. Hooked Leaflet instance -> setView to (20, 0) at min zoom.
      2. Click the Leaflet zoom-out control several times.
    """
    did = await page.evaluate("""
    () => {
        const mapEl = document.querySelector(
            '.leaflet-container, [class*="guess-map"], #map'
        );
        const maps = window.__ogMaps__ || [];
        let map = null;
        for (const m of maps) {
            try { if (m && m._container === mapEl) { map = m; break; } } catch (e) {}
        }
        if (!map && maps.length === 1) map = maps[0];
        if (!map && mapEl && mapEl._leaflet_map) map = mapEl._leaflet_map;
        if (map && typeof map.setView === 'function') {
            try {
                const z = (typeof map.getMinZoom === 'function') ? (map.getMinZoom() || 0) : 0;
                map.setView([20, 0], z, { animate: false });
                if (typeof map.invalidateSize === 'function') map.invalidateSize();
                return true;
            } catch (e) { return false; }
        }
        return false;
    }
    """)
    if did:
        await asyncio.sleep(0.35)
        return True

    # Fallback: click the Leaflet zoom-out control until it bottoms out at the
    # world view (disabled controls simply no-op).
    try:
        btn = await page.query_selector(
            ".leaflet-control-zoom-out, a.leaflet-control-zoom-out"
        )
        if btn:
            for _ in range(8):
                try:
                    await btn.click(timeout=500)
                    await asyncio.sleep(0.12)
                except Exception:
                    break
            await asyncio.sleep(0.3)
            return True
    except Exception:
        pass
    return False


async def click_guess_on_map(
    page,
    lat: float,
    lng: float,
    map_bounds: Optional[dict[str, int]] = None,
) -> bool:
    """
    Click the minimap at (lat, lng).

    Projection strategy, most-accurate first:
      1. Leaflet ``latLngToContainerPoint`` — used when the live map instance
         is reachable from the DOM.
      2. Tile-derived Web Mercator — reads a visible raster tile's
         ``/{z}/{x}/{y}`` (or ``?x=&y=&z=``) coordinates together with its
         on-screen rectangle to reconstruct the exact pixel↔world mapping.
         This accounts for the map's actual centre, zoom and pan.
      3. Whole-world Mercator fitted to the element rect (last resort only).

    The old behaviour stretched the whole world across the element rect with
    independent x/y scaling, which ignored centre/zoom and distorted latitude —
    making the click land far from the intended coordinates.

    Primary strategy: if the live Leaflet instance is available, recenter the
    map on the guess and click the centre — this guarantees the target is on
    screen even on a tiny / zoomed / panned minimap. Otherwise fall back to the
    tile/Mercator projection below.
    """
    # Strategy 0 (most robust): recenter the Leaflet map on the guess, then
    # click the centre. A small minimap often cannot show the whole world at
    # once, so a coordinate like Japan can sit off the right edge and get
    # clamped to the wrong place. Moving the target to the centre avoids that.
    centered = await page.evaluate(
        """
    ([lat, lng]) => {
        const mapEl = document.querySelector(
            '.leaflet-container, [class*="guess-map"], #map, [class*="map"]'
        );
        if (!mapEl) return null;
        const maps = window.__ogMaps__ || [];
        let map = null;
        for (const m of maps) {
            try { if (m && m._container === mapEl) { map = m; break; } } catch (e) {}
        }
        if (!map && maps.length === 1) map = maps[0];
        if (!map && mapEl._leaflet_map) map = mapEl._leaflet_map;
        if (!map || typeof map.setView !== 'function') return null;
        try {
            if (typeof map.invalidateSize === 'function') map.invalidateSize();
            const minZ = (typeof map.getMinZoom === 'function') ? (map.getMinZoom() || 0) : 0;
            map.setView([lat, lng], minZ, { animate: false });
            const rect = mapEl.getBoundingClientRect();
            const pt = map.latLngToContainerPoint([lat, lng]);
            const x = rect.left + pt.x;
            const y = rect.top + pt.y;
            const onScreen = (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom);
            return { x: x, y: y, onScreen: onScreen,
                     rect: { left: rect.left, top: rect.top, w: rect.width, h: rect.height } };
        } catch (e) { return null; }
    }
    """,
        [lat, lng],
    )

    if centered and centered.get("x") is not None and centered.get("onScreen"):
        await asyncio.sleep(0.25)  # let the pan settle
        r = centered["rect"]
        x = max(r["left"] + 4, min(centered["x"], r["left"] + r["w"] - 4))
        y = max(r["top"] + 4, min(centered["y"], r["top"] + r["h"] - 4))
        print(f"    map-click [leaflet-center] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")
        await page.mouse.move(x, y)
        await asyncio.sleep(0.1)
        await page.mouse.click(x, y)
        await asyncio.sleep(random.uniform(0.3, 0.6))
        return True

    # Fallback: no usable Leaflet instance — try to reset to a world view so
    # the projection below has the whole map visible, then project.
    await reset_minimap_to_world(page)

    info = await page.evaluate(
        """
    ([lat, lng]) => {
        const mapEl = document.querySelector(
            '.leaflet-container, [class*="guess-map"], #map, [class*="map"]'
        );
        if (!mapEl) return { method: 'none' };
        const rect = mapEl.getBoundingClientRect();
        const rectInfo = { left: rect.left, top: rect.top, w: rect.width, h: rect.height };
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;

        // --- Strategy 1: live Leaflet map instance (hooked or discoverable) ---
        let map = null;
        const hooked = window.__ogMaps__ || [];
        for (const m of hooked) {
            try { if (m && m._container === mapEl) { map = m; break; } } catch (e) {}
        }
        if (!map && hooked.length === 1) map = hooked[0];
        if (!map && mapEl._leaflet_map) map = mapEl._leaflet_map;
        if (map && typeof map.latLngToContainerPoint === 'function') {
            const pt = map.latLngToContainerPoint([lat, lng]);
            const x = rect.left + pt.x;
            const y = rect.top + pt.y;
            const onScreen = (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom);
            return { method: 'leaflet', x: x, y: y, onScreen: onScreen, rect: rectInfo };
        }

        // --- Strategy 2: derive the projection from a visible raster tile ---
        const imgs = Array.from(mapEl.querySelectorAll('img'));
        let tile = null;
        let bestD = Infinity;
        for (const img of imgs) {
            const src = img.currentSrc || img.src || '';
            let z, tx, ty;
            let m = src.match(/\\/(\\d{1,2})\\/(\\d{1,7})\\/(\\d{1,7})(?:[.?&\\/]|$)/);
            if (m) { z = +m[1]; tx = +m[2]; ty = +m[3]; }
            else {
                const mz = src.match(/[?&](?:z|zoom)=(\\d{1,2})/);
                const mx = src.match(/[?&]x=(\\d{1,7})/);
                const my = src.match(/[?&]y=(\\d{1,7})/);
                if (mz && mx && my) { z = +mz[1]; tx = +mx[1]; ty = +my[1]; }
            }
            if (z === undefined || isNaN(z)) continue;
            const r = img.getBoundingClientRect();
            if (r.width < 64 || r.height < 64) continue;
            // must overlap the map viewport (skip stale off-screen tiles)
            if (r.right < rect.left || r.left > rect.right ||
                r.bottom < rect.top || r.top > rect.bottom) continue;
            const tcx = r.left + r.width / 2;
            const tcy = r.top + r.height / 2;
            const d = (tcx - cx) * (tcx - cx) + (tcy - cy) * (tcy - cy);
            if (d < bestD) {
                bestD = d;
                tile = { z: z, tx: tx, ty: ty, left: r.left, top: r.top, w: r.width, h: r.height };
            }
        }
        if (tile) {
            const span = tile.w * Math.pow(2, tile.z);   // full-world width (screen px)
            let x = tile.left + ((lng + 180) / 360) * span - tile.tx * tile.w;
            const s = Math.max(-0.9999, Math.min(0.9999, Math.sin(lat * Math.PI / 180)));
            const yNorm = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
            const y = tile.top + yNorm * (tile.h * Math.pow(2, tile.z)) - tile.ty * tile.h;
            // Longitude wraps: pick the world copy whose x is nearest the centre.
            x = x - span * Math.round((x - cx) / span);
            const onScreen = (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom);
            return { method: 'tiles', x: x, y: y, z: tile.z, onScreen: onScreen, rect: rectInfo };
        }

        return { method: 'rect_only', rect: rectInfo };
    }
    """,
        [lat, lng],
    )

    method = info.get("method", "none") if info else "none"
    on_screen = bool(info.get("onScreen")) if info else False
    x = y = None

    if method in ("leaflet", "tiles") and info.get("x") is not None:
        x, y = info["x"], info["y"]
    elif info and info.get("rect"):
        # Last resort: fit the whole-world Mercator to the element rect.
        r = info["rect"]
        x_merc, y_merc = _latlng_to_mercator(lat, lng)
        x = r["left"] + x_merc * r["w"]
        y = r["top"] + y_merc * r["h"]
        method = "mercator_elem"
    else:
        x, y = coordinates_to_map_click(lat, lng, map_bounds)
        method = "mercator_static"

    if method in ("leaflet", "tiles") and not on_screen:
        print(
            f"    [WARN] guess ({lat:.4f},{lng:.4f}) is off the visible minimap; "
            f"clamping to edge (map may still be zoomed in)"
        )

    # Keep the click inside the map element so the guess at least registers.
    if info and info.get("rect"):
        r = info["rect"]
        inset = 4
        x = max(r["left"] + inset, min(x, r["left"] + r["w"] - inset))
        y = max(r["top"] + inset, min(y, r["top"] + r["h"] - inset))
    x = max(0, min(x, 4000))
    y = max(0, min(y, 4000))

    print(f"    map-click [{method}] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")

    await page.mouse.move(x, y)
    await asyncio.sleep(0.1)
    await page.mouse.click(x, y)
    await asyncio.sleep(random.uniform(0.3, 0.6))
    return True


async def submit_guess(page) -> bool:
    submit_selectors = [
        "#confirm-button",
        "[class*='confirm-button']",
        "button:has-text('Guess')",
        "button:has-text('Submit')",
        "button:has-text('Confirm')",
        "button:has-text('OK')",
        '[data-testid="guess-button"]',
        '[class*="guess-button"]',
        "div:has-text('Guess')",
        "div:has-text('Confirm')",
    ]
    for selector in submit_selectors:
        try:
            btn = await page.wait_for_selector(selector, timeout=2000)
            if btn:
                await btn.click()
                await asyncio.sleep(random.uniform(0.3, 0.6))
                return True
        except Exception:
            continue
    return False


async def go_to_next_round(page) -> bool:
    next_selectors = [
        "#next-round",
        "button#next-round",
        "button:has-text('Continue')",
        "button:has-text('Next Round')",
        "button:has-text('Next')",
        "button:has-text('Play Again')",
        "button:has-text('Return')",
        '[class*="next-round"]',
        "div:has-text('Continue')",
        "div:has-text('Next Round')",
        "div:has-text('Play Again')",
        "div:has-text('Return')",
        "a:has-text('Next Round')",
        "a:has-text('Continue')",
    ]
    for selector in next_selectors:
        try:
            btn = await page.wait_for_selector(selector, timeout=2000)
            if btn:
                await btn.click()
                await asyncio.sleep(random.uniform(0.3, 0.6))
                return True
        except Exception:
            continue
    return False


async def start_game(page) -> bool:
    start_selectors = [
        "div:has-text('Singleplayer')",
        "div:has-text('Single Player')",
        "div:has-text('singleplayer')",
        "button:has-text('Play')",
        "button:has-text('Start')",
        "a:has-text('Play')",
        '[class*="start-button"]',
        '[data-testid="play-button"]',
    ]
    for selector in start_selectors:
        try:
            btn = await page.wait_for_selector(selector, timeout=2000)
            if btn:
                await btn.click()
                await asyncio.sleep(random.uniform(0.5, 1.0))
                return True
        except Exception:
            continue

    try:
        await page.keyboard.press("Enter")
        await asyncio.sleep(0.5)
        return True
    except Exception:
        pass

    return False


async def dismiss_cookie_banner(page) -> bool:
    for attempt in range(5):
        clicked = await page.evaluate("""
        () => {
            const selectors = [
                '.fc-primary-button', '.fc-consent-root .fc-primary-button',
                '[class*="fc-button"]', '.fc-button',
                'button:has-text("Accept")', 'button:has-text("Consent")',
                'div:has-text("Accept")', 'div:has-text("Consent")',
            ];
            for (const sel of selectors) {
                try {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) {
                        el.click();
                        return true;
                    }
                } catch(e) {}
            }
            const texts = ['accept all', 'accept', 'consent', 'agree', 'ok'];
            const all = document.querySelectorAll('button, a, div, span');
            for (const el of all) {
                if (el.onclick || el.getAttribute('role') === 'button'
                    || el.classList.toString().includes('button')
                    || el.classList.toString().includes('btn')) {
                    const t = (el.textContent || '').toLowerCase().trim();
                    if (texts.some(x => t === x || t.startsWith(x))) {
                        if (el.offsetParent !== null) {
                            el.click();
                            return true;
                        }
                    }
                }
            }
            return false;
        }
        """)
        if clicked:
            await asyncio.sleep(random.uniform(0.4, 0.7))

        await page.evaluate("""
        () => {
            const buttons = document.querySelectorAll(
                'button, div[role="button"], span[role="button"], '
                + '[class*="close"], [class*="dismiss"], '
                + '.fc-dialog-header-back-button, [aria-label*="close" i]'
            );
            for (const b of buttons) {
                const t = (b.textContent || '').toLowerCase().trim();
                if (['close', 'x', '\u00d7', 'dismiss', 'no thanks',
                     'not now', 'continue', 'ok'].includes(t)) {
                    if (b.offsetParent !== null || getComputedStyle(b).display !== 'none') {
                        b.click();
                    }
                }
            }
            document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
        }
        """)
        await asyncio.sleep(0.3)

        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.3)
        except Exception:
            pass

        overlay_gone = await page.evaluate("""
        () => {
            const roots = document.querySelectorAll(
                '.fc-consent-root, [class*="fc-dialog-overlay"]'
            );
            if (roots.length === 0) return true;
            for (const el of roots) {
                const s = getComputedStyle(el);
                if (s.display !== 'none' && s.visibility !== 'hidden'
                    && parseFloat(s.opacity) > 0.1) {
                    return false;
                }
            }
            return true;
        }
        """)
        if overlay_gone:
            return True

        await asyncio.sleep(0.5)

    await page.evaluate("""
    () => {
        document.querySelectorAll(
            '.fc-consent-root, [class*="fc-dialog-overlay"], '
            + '.fc-consent-root *'
        ).forEach(el => {
            el.style.display = 'none';
            el.style.visibility = 'hidden';
            el.style.opacity = '0';
            el.style.pointerEvents = 'none';
            el.style.zIndex = '-1';
        });
        document.body.style.overflow = '';
        document.body.style.position = '';
    }
    """)
    return True


async def hover_minimap_to_expand(page, map_selector: str | None = None) -> bool:
    """
    Move the mouse over the minimap so the UI expands it to the big world map.

    OpenGuessr's minimap is a small Leaflet container that grows on hover.
    We move the mouse in a human-like curve to its centre and wait for
    the CSS transition to finish.

    Returns True if a map element was found and hovered.
    """
    selectors = [map_selector] if map_selector else [
        ".leaflet-container",
        '[class*="guess-map"]',
        "#map",
        '[class*="minimap"]',
        '[class*="mini-map"]',
    ]

    found_sel = None
    box = None
    for sel in selectors:
        if not sel:
            continue
        try:
            el = await page.query_selector(sel)
            if el:
                found_sel = sel
                break
        except Exception:
            continue

    if not found_sel:
        return False

    box = await page.evaluate(f"""
    () => {{
        const el = document.querySelector('{found_sel}');
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return {{ left: r.left, top: r.top, w: r.width, h: r.height }};
    }}
    """)

    if not box:
        return False

    center_x = box["left"] + box["w"] / 2
    center_y = box["top"] + box["h"] / 2
    print(f"    Hovering minimap at ({center_x:.0f}, {center_y:.0f}) — {box['w']:.0f}x{box['h']:.0f}")

    start_x = random.randint(100, 800)
    start_y = random.randint(100, 600)
    await page.mouse.move(start_x, start_y)
    await asyncio.sleep(0.1)

    steps = random.randint(6, 12)
    for i in range(steps):
        t = (i + 1) / steps
        ix = int(start_x + (center_x - start_x) * t + random.uniform(-3, 3))
        iy = int(start_y + (center_y - start_y) * t + random.uniform(-3, 3))
        await page.mouse.move(ix, iy)
        await asyncio.sleep(random.uniform(0.01, 0.03))

    await asyncio.sleep(1.0)  # expansion animation
    return True


async def add_jitter_to_click(
    page, x: int, y: int, jitter: int = 8
) -> None:
    jx = x + random.randint(-jitter, jitter)
    jy = y + random.randint(-jitter, jitter)
    await page.mouse.click(jx, jy)
    await asyncio.sleep(random.uniform(0.1, 0.3))

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


async def click_guess_on_map(
    page,
    lat: float,
    lng: float,
    map_bounds: Optional[dict[str, int]] = None,
) -> bool:
    """
    Click the minimap at (lat, lng). Returns debug info as second value.
    Tries multiple strategies in order:
      1. Leaflet latLngToContainerPoint (correct for any zoom)
      2. Compute pixel from map's pixel bounds
      3. Mercator fallback
    """
    x_merc, y_merc = _latlng_to_mercator(lat, lng)
    debug = []

    # Strategy 1: Leaflet API
    info = await page.evaluate("""
    ([lat, lng]) => {
        const mapEl = document.querySelector(
            '.leaflet-container, [class*="guess-map"], #map, [class*="map"]'
        );
        if (!mapEl) return { err: 'no_map_element' };
        const rect = mapEl.getBoundingClientRect();

        let map = null;
        if (mapEl._leaflet_map) {
            map = mapEl._leaflet_map;
        } else if (typeof L !== 'undefined' && L.Map && L.Map._instances) {
            for (const m of Object.values(L.Map._instances)) {
                if (m._container === mapEl || m.getContainer() === mapEl) {
                    map = m;
                    break;
                }
            }
        }

        if (map && map.latLngToContainerPoint) {
            const pt = map.latLngToContainerPoint([lat, lng]);
            return {
                method: 'leaflet',
                x: rect.left + pt.x,
                y: rect.top + pt.y,
                rect: { left: rect.left, top: rect.top, w: rect.width, h: rect.height },
                pt: { x: pt.x, y: pt.y },
            };
        }

        return {
            method: 'no_leaflet',
            rect: { left: rect.left, top: rect.top, w: rect.width, h: rect.height },
        };
    }
    """, [lat, lng])

    x = y = None
    method = "unknown"

    if info and info.get("method") == "leaflet":
        x, y = info["x"], info["y"]
        method = "leaflet"
        r = info["rect"]
        debug.append(
            f"leaflet: rect({r['left']:.0f},{r['top']:.0f} {r['w']:.0f}x{r['h']:.0f}) "
            f"pt({info['pt']['x']:.1f},{info['pt']['y']:.1f}) "
            f"-> page({x:.0f},{y:.0f})"
        )
    elif info and "rect" in info:
        r = info["rect"]
        # Strategy 2: use Mercator within the actual map element rect
        x = r["left"] + x_merc * r["w"]
        y = r["top"] + y_merc * r["h"]
        method = "mercator_elem"
        debug.append(
            f"mercator_elem: rect({r['left']:.0f},{r['top']:.0f} {r['w']:.0f}x{r['h']:.0f}) "
            f"merc({x_merc:.4f},{y_merc:.4f}) -> page({x:.0f},{y:.0f})"
        )
    else:
        # Strategy 3: hardcoded bounds
        x, y = coordinates_to_map_click(lat, lng, map_bounds)
        method = "mercator_static"
        debug.append(f"mercator_static: ({x:.0f},{y:.0f})")

    x = max(0, min(x, 2000))
    y = max(0, min(y, 2000))

    debug.append(f"click: ({lat:.4f},{lng:.4f}) -> page({x:.0f},{y:.0f}) [{method}]")
    print(f"    {' | '.join(debug[-2:])}")

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

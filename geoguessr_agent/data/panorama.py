from __future__ import annotations

import asyncio
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


@dataclass
class TileEntry:
    panoid: str
    zoom: int
    x: int
    y: int
    data: bytes


@dataclass
class PanoramaInfo:
    panoid: str
    zoom: int
    tile_size: int
    cols: int
    rows: int
    tiles: dict[tuple[int, int], bytes] = field(default_factory=dict)
    lat: Optional[float] = None
    lng: Optional[float] = None
    heading: float = 0.0


class TileCollector:
    """Collects Street View tiles intercepted from the browser's network layer."""

    TILE_HOSTS = {
        "streetviewpixels-pa.googleapis.com",
        "geo0.ggpht.com",
        "geo1.ggpht.com",
        "geo2.ggpht.com",
        "geo3.ggpht.com",
        "lh3.googleusercontent.com",
        "lh4.googleusercontent.com",
        "lh5.googleusercontent.com",
        "lh6.googleusercontent.com",
        "maps.googleapis.com",
        "tile.googleapis.com",
        "streetviewpixels.googleapis.com",
    }

    TILE_PATH_TOKENS = {
        "streetviewpixels",
        "tile",
        "pano",
        "cbk",
    }

    def __init__(self, timeout: float = 3.0):
        self._tiles: dict[str, PanoramaInfo] = {}
        self._lock = asyncio.Lock()

    def is_tile_request(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        path = parsed.path or ""

        if any(h in host for h in self.TILE_HOSTS):
            return True

        for token in self.TILE_PATH_TOKENS:
            if token in path:
                return True

        return False

    def parse_tile_request(self, url: str) -> Optional[TileEntry]:
        """Parse a tile URL and extract panoid, x, y, zoom."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        panoid = params.get("panoid", [None])[0]
        if not panoid:
            path_parts = parsed.path.strip("/").split("/")
            for part in path_parts:
                if len(part) >= 20:
                    panoid = part
                    break

        if not panoid:
            return None

        x = int(params.get("x", ["0"])[0])
        y = int(params.get("y", ["0"])[0])
        zoom = int(params.get("zoom", ["3"])[0])

        return TileEntry(panoid=panoid, zoom=zoom, x=x, y=y, data=b"")

    async def add_tile(self, panoid: str, zoom: int, x: int, y: int, data: bytes) -> None:
        async with self._lock:
            if panoid not in self._tiles:
                self._tiles[panoid] = PanoramaInfo(
                    panoid=panoid,
                    zoom=zoom,
                    tile_size=512,
                    cols=2 ** (zoom + 1),
                    rows=2 ** zoom,
                )
            pano = self._tiles[panoid]
            if zoom > pano.zoom:
                pano.zoom = zoom
                pano.cols = 2 ** (zoom + 1)
                pano.rows = 2 ** zoom
            pano.tiles[(x, y)] = data

    async def get_panorama(
        self, panoid: str, timeout: float = 3.0
    ) -> Optional[PanoramaInfo]:
        """Wait for enough tiles to stitch a panorama, then return it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._lock:
                pano = self._tiles.get(panoid)
                if pano and len(pano.tiles) > 0:
                    expected = pano.cols * pano.rows
                    actual = len(pano.tiles)
                    if actual >= max(4, expected * 0.25):
                        return pano
            await asyncio.sleep(0.15)
        async with self._lock:
            pano = self._tiles.get(panoid)
            return pano if pano and len(pano.tiles) >= 4 else None

    async def clear(self) -> None:
        async with self._lock:
            self._tiles.clear()

    @property
    def active_panoids(self) -> list[str]:
        return list(self._tiles.keys())


def stitch_panorama(pano: PanoramaInfo) -> Optional[Image.Image]:
    """Reconstruct an equirectangular panorama from collected tiles."""
    if not pano.tiles:
        return None

    first_data = next(iter(pano.tiles.values()))
    try:
        tile_img = Image.open(io.BytesIO(first_data))
        tile_w, tile_h = tile_img.size
    except Exception:
        return None

    pano_w = pano.cols * tile_w
    pano_h = pano.rows * tile_h

    canvas = Image.new("RGB", (pano_w, pano_h), (0, 0, 0))

    for (x, y), data in pano.tiles.items():
        try:
            tile = Image.open(io.BytesIO(data)).convert("RGB")
            left = x * tile_w
            top = y * tile_h
            canvas.paste(tile, (left, top))
        except Exception:
            continue

    return canvas


def equirect_to_perspective(
    equi: Image.Image,
    heading_deg: float = 0.0,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (640, 640),
) -> Image.Image:
    """
    Extract a perspective (gnomonic) crop from an equirectangular panorama.

    Converts each output pixel's ray direction back to equirectangular UV
    coordinates using gnomonic projection. Handles the equirectangular
    horizontal wrap-around.

    Args:
        equi: Equirectangular panorama (PIL Image, 2:1 aspect ratio).
        heading_deg: Yaw in degrees (0 = centre of panorama, 90 = right).
        pitch_deg: Pitch in degrees (0 = horizon, positive = up).
        fov_deg: Field of view in degrees.
        out_size: (width, height) of the output perspective image.

    Returns:
        A perspective-corrected PIL Image crop.
    """
    import numpy as np

    equi_w, equi_h = equi.size
    equi_arr = np.array(equi, dtype=np.uint8)

    out_w, out_h = out_size
    fov = math.radians(fov_deg)
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    focal = out_w / (2.0 * math.tan(fov / 2.0))

    xs = np.arange(out_w) - (out_w - 1) / 2.0
    ys = np.arange(out_h) - (out_h - 1) / 2.0
    xs, ys = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))

    cam_x = xs
    cam_y = -ys
    cam_z = np.full_like(xs, focal, dtype=np.float64)

    inv_norm = 1.0 / np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
    cam_x *= inv_norm
    cam_y *= inv_norm
    cam_z *= inv_norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    ch, sh = math.cos(heading), math.sin(heading)

    wx = ch * cam_x - sp * sh * cam_y + cp * sh * cam_z
    wy = cp * cam_y + sp * cam_z
    wz = -sh * cam_x - sp * ch * cam_y + cp * ch * cam_z

    lon = np.arctan2(wx, wz)
    lat = np.arcsin(np.clip(wy, -1.0, 1.0))

    u = (lon / (2.0 * math.pi) + 0.5) * equi_w
    v = (0.5 - lat / math.pi) * equi_h

    u = np.mod(u, equi_w)
    v = np.clip(v, 0, equi_h - 1)

    u_int = u.astype(np.int32)
    v_int = v.astype(np.int32)

    return Image.fromarray(equi_arr[v_int, u_int], mode="RGB")


def generate_view_crops(
    equi: Image.Image,
    base_heading: float = 0.0,
    n_horizontal: int = 5,
    n_vertical: int = 0,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (640, 640),
) -> list[tuple[Image.Image, float, float]]:
    """
    Generate multiple perspective crops from an equirectangular panorama.

    Returns a list of (crop_image, heading, pitch) tuples.
    """
    crops = []

    for i in range(n_horizontal):
        heading = base_heading + (i - n_horizontal // 2) * fov_deg * 0.6
        crop = equirect_to_perspective(
            equi, heading_deg=heading, pitch_deg=0.0,
            fov_deg=fov_deg, out_size=out_size,
        )
        crops.append((crop, heading, 0.0))

    for i in range(n_vertical):
        pitch = 30.0 if i == 0 else -30.0
        crop = equirect_to_perspective(
            equi, heading_deg=base_heading, pitch_deg=pitch,
            fov_deg=fov_deg, out_size=out_size,
        )
        crops.append((crop, base_heading, pitch))

    return crops


def panorama_to_inference_image(
    pano: PanoramaInfo,
    base_heading: float = 0.0,
    max_dimension: int = 2048,
) -> Optional[Image.Image]:
    """
    Stitch and resize a panorama for inference.

    Returns a single equirectangular image suitable for direct model input,
    or None if stitching fails.
    """
    equi = stitch_panorama(pano)
    if equi is None:
        return None

    w, h = equi.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        equi = equi.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    return equi

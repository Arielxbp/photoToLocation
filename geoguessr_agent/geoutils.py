from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import s2sphere


def generate_s2_region_centroids(level: int = 6) -> torch.Tensor:
    """
    Generate region centroids using S2 cell hierarchy.

    Level 6 → ~24,500 cells (65km × 65km)
    Level 8 → ~393,000 cells (12km × 12km)
    Level 10 → ~6,290,000 cells (3km × 3km)

    Returns:
        centroids: (num_cells, 2) tensor with (lat_rad, lng_rad) for each S2 cell center
    """
    centroids = []

    start = s2sphere.CellId.begin(level)
    end = s2sphere.CellId.end(level)
    cell = start
    while cell != end:
        latlng = cell.to_lat_lng()
        lat_rad = math.radians(latlng.lat().degrees)
        lng_rad = math.radians(latlng.lng().degrees)
        centroids.append([lat_rad, lng_rad])
        cell = cell.next()

    return torch.tensor(centroids, dtype=torch.float32)


def generate_geodesic_centroids(num_points: int = 3000) -> torch.Tensor:
    """
    Generate approximately uniform points on a sphere using Fibonacci lattice.

    Returns:
        centroids: (num_points, 2) tensor with (lat_rad, lng_rad)
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        lng_rad = theta % (2 * math.pi)
        lat_rad = math.asin(y)
        points.append([lat_rad, lng_rad])

    return torch.tensor(points, dtype=torch.float32)


def load_region_centroids(path: str | Path) -> torch.Tensor:
    """Load pre-saved region centroids."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data["centroids"]
    return data


def save_region_centroids(centroids: torch.Tensor, path: str | Path) -> None:
    """Save region centroids for reuse."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": centroids}, str(path))


def get_country_centroids(country_index: dict[str, int]) -> torch.Tensor:
    """
    Return approximate centroids for each country in the index.
    These are coarse — used for initial click positioning.

    Places each centroid at the tensor index matching the country's
    class index (from country_index values) so that the model's
    predicted class i maps to the correct country centroid.
    """
    COUNTRY_LATLNG = {
        "United States": (39.83, -98.58),
        "Canada": (56.13, -106.35),
        "Mexico": (23.63, -102.53),
        "Brazil": (-14.24, -51.93),
        "Argentina": (-38.42, -63.62),
        "Chile": (-35.68, -71.54),
        "Peru": (-9.19, -75.02),
        "Colombia": (4.57, -74.30),
        "Ecuador": (-1.83, -78.18),
        "France": (46.23, 2.21),
        "Germany": (51.17, 10.45),
        "Italy": (41.87, 12.57),
        "Spain": (40.46, -3.75),
        "United Kingdom": (55.38, -3.44),
        "Poland": (51.92, 19.15),
        "Netherlands": (52.13, 5.29),
        "Belgium": (50.50, 4.47),
        "Sweden": (60.13, 18.64),
        "Norway": (60.47, 8.47),
        "Finland": (61.92, 25.75),
        "Denmark": (56.26, 9.50),
        "Russia": (61.52, 105.32),
        "Turkey": (38.96, 35.24),
        "Greece": (39.07, 21.82),
        "Romania": (45.94, 24.97),
        "Bulgaria": (42.73, 25.49),
        "Hungary": (47.16, 19.50),
        "Czechia": (49.82, 15.47),
        "Slovakia": (48.67, 19.70),
        "Austria": (47.70, 13.35),
        "Switzerland": (46.82, 8.23),
        "Portugal": (39.40, -8.22),
        "Ireland": (53.14, -7.69),
        "Ukraine": (48.38, 31.17),
        "Belarus": (53.71, 27.95),
        "Lithuania": (55.17, 23.88),
        "Latvia": (56.88, 24.60),
        "Estonia": (58.60, 25.01),
        "Croatia": (45.10, 15.20),
        "Serbia": (44.02, 21.01),
        "Slovenia": (46.15, 15.00),
        "Albania": (41.15, 20.17),
        "China": (35.86, 104.20),
        "India": (20.59, 78.96),
        "Japan": (36.20, 138.25),
        "South Korea": (35.91, 127.77),
        "Taiwan": (23.70, 120.96),
        "Thailand": (15.87, 100.99),
        "Indonesia": (-0.79, 113.92),
        "Philippines": (12.88, 121.77),
        "Malaysia": (4.21, 101.98),
        "Singapore": (1.35, 103.82),
        "Vietnam": (14.06, 108.28),
        "Cambodia": (12.57, 104.99),
        "Laos": (19.86, 102.50),
        "Sri Lanka": (7.87, 80.77),
        "Bangladesh": (23.68, 90.36),
        "Pakistan": (30.38, 69.35),
        "Mongolia": (46.86, 103.85),
        "Kazakhstan": (48.02, 66.92),
        "Kyrgyzstan": (41.20, 74.77),
        "Bhutan": (27.51, 90.43),
        "Nepal": (28.39, 84.12),
        "United Arab Emirates": (23.42, 53.85),
        "Qatar": (25.35, 51.18),
        "Oman": (21.47, 55.92),
        "Jordan": (30.59, 36.24),
        "Israel": (31.05, 34.85),
        "Lebanon": (33.85, 35.86),
        "Iraq": (33.22, 43.68),
        "Australia": (-25.27, 133.78),
        "New Zealand": (-40.90, 174.89),
        "South Africa": (-30.56, 22.94),
        "Kenya": (-0.02, 37.91),
        "Nigeria": (9.08, 8.68),
        "Ghana": (7.95, -1.02),
        "Senegal": (14.50, -14.45),
        "Tunisia": (33.89, 9.54),
        "Egypt": (26.82, 30.80),
        "Botswana": (-22.33, 24.68),
        "Lesotho": (-29.61, 28.23),
        "Eswatini": (-26.52, 31.47),
        "Madagascar": (-18.77, 46.87),
        "Uganda": (1.37, 32.29),
        "Tanzania": (-6.37, 34.89),
        "Namibia": (-22.96, 18.49),
        "Mali": (17.57, -4.00),
        "Rwanda": (-1.94, 29.87),
        "Morocco": (31.79, -7.09),
        "Reunion": (-21.12, 55.54),
        "Panama": (8.54, -80.78),
        "Costa Rica": (9.75, -83.75),
        "Guatemala": (15.78, -90.23),
        "Dominican Republic": (18.74, -70.16),
        "Puerto Rico": (18.22, -66.59),
        "Uruguay": (-32.52, -55.77),
        "Bolivia": (-16.29, -63.59),
        "Greenland": (71.71, -42.60),
        "Iceland": (64.96, -19.02),
        "Faroe Islands": (61.89, -6.91),
        "Svalbard": (77.88, 17.30),
    }

    n = len(country_index)
    centroids = [(0.0, 0.0)] * n
    for country, idx in country_index.items():
        if country in COUNTRY_LATLNG:
            lat, lng = COUNTRY_LATLNG[country]
        else:
            lat, lng = (0.0, 0.0)
        centroids[idx] = (math.radians(lat), math.radians(lng))

    return torch.tensor(centroids, dtype=torch.float32)

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional


COUNTRY_NAME_NORMALIZE: dict[str, str] = {
    "Russian Federation": "Russia",
    "Korea, Republic of": "South Korea",
    "Türkiye": "Turkey",
    "Taiwan, Province of China": "Taiwan",
    "Viet Nam": "Vietnam",
    "Lao People's Democratic Republic": "Laos",
    "Bolivia, Plurinational State of": "Bolivia",
    "Tanzania, United Republic of": "Tanzania",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Syrian Arab Republic": "Syria",
    "Moldova, Republic of": "Moldova",
    "Congo, The Democratic Republic of the": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Brunei Darussalam": "Brunei",
    "Holy See (Vatican City State)": "Vatican City",
    "Iran, Islamic Republic of": "Iran",
    "Micronesia, Federated States of": "Micronesia",
    "Palestine, State of": "Palestine",
    "Réunion": "Reunion",
    "Virgin Islands, U.S.": "US Virgin Islands",
    "Åland Islands": "Aland Islands",
}


def _normalize_country(name: str) -> str:
    return COUNTRY_NAME_NORMALIZE.get(name, name)


class CountryMapper:
    """Maps country names to indices and handles distribution balancing."""

    def __init__(self):
        self.country_to_idx: dict[str, int] = {}
        self.idx_to_country: dict[int, str] = {}
        self.country_to_continent: dict[str, int] = {}
        self.continent_to_idx: dict[str, int] = {}
        self.idx_to_continent: dict[int, str] = {}

    def build_from_files(
        self,
        data_dir: str | Path,
        min_samples: int = 50,
    ) -> dict[str, int]:
        """Scan metadata JSON files and build country index."""
        data_dir = Path(data_dir)
        country_counts: Counter[str] = Counter()
        country_coords: dict[str, list[tuple[float, float]]] = {}

        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(root, f)) as fh:
                        meta = json.load(fh)
                    country = _normalize_country(meta.get("country_name") or meta.get("country", "Unknown"))
                    coords = meta.get("coordinates", [None, None])
                    lat = coords[0] if coords else meta.get("lat")
                    lng = coords[1] if len(coords) > 1 else meta.get("lng")
                    if country:
                        country_counts[country] += 1
                        if lat is not None and lng is not None:
                            country_coords.setdefault(country, []).append((lat, lng))
                except Exception:
                    continue

        valid_countries = [
            c for c, count in country_counts.items()
            if count >= min_samples and c != "Unknown"
        ]
        valid_countries.sort()

        self.country_to_idx = {c: i for i, c in enumerate(valid_countries)}
        self.idx_to_country = {i: c for c, i in self.country_to_idx.items()}

        self._assign_continents(valid_countries, country_coords)

        return self.country_to_idx

    def _assign_continents(
        self,
        countries: list[str],
        coords: dict[str, list[tuple[float, float]]],
    ) -> None:
        continent_map = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "SA": "South America",
            "OC": "Oceania",
            "AN": "Antarctica",
        }

        country_continent_override = {
            "United States": "North America",
            "Canada": "North America",
            "Mexico": "North America",
            "Brazil": "South America",
            "Argentina": "South America",
            "Chile": "South America",
            "Peru": "South America",
            "Colombia": "South America",
            "France": "Europe",
            "Germany": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
            "United Kingdom": "Europe",
            "Poland": "Europe",
            "Netherlands": "Europe",
            "Belgium": "Europe",
            "Russia": "Europe",
            "Turkey": "Europe",
            "China": "Asia",
            "India": "Asia",
            "Japan": "Asia",
            "South Korea": "Asia",
            "Thailand": "Asia",
            "Indonesia": "Asia",
            "Philippines": "Asia",
            "Malaysia": "Asia",
            "Australia": "Oceania",
            "New Zealand": "Oceania",
            "South Africa": "Africa",
            "Kenya": "Africa",
            "Nigeria": "Africa",
            "Ghana": "Africa",
            "Senegal": "Africa",
            "Egypt": "Africa",
            "Tunisia": "Africa",
            "Botswana": "Africa",
            "Lesotho": "Africa",
            "Eswatini": "Africa",
        }

        continent_names = sorted(set(continent_map.values()))
        self.continent_to_idx = {c: i for i, c in enumerate(continent_names)}
        self.idx_to_continent = {i: c for c, i in self.continent_to_idx.items()}

        for country in countries:
            if country in country_continent_override:
                self.country_to_continent[country] = self.continent_to_idx[
                    country_continent_override[country]
                ]
            elif country in coords:
                avg_lng = sum(c[1] for c in coords[country]) / len(coords[country])
                if avg_lng < -30:
                    cont = "South America" if avg_lng > -80 else "North America"
                elif avg_lng > 60:
                    cont = "Oceania" if avg_lng > 110 else "Asia"
                else:
                    cont = "Europe" if avg_lng > -10 else "Africa"
                self.country_to_continent[country] = self.continent_to_idx[cont]
            else:
                self.country_to_continent[country] = 0

def load_kaggle_metadata(data_dir: str | Path) -> list[dict]:
    """Load all metadata from a Kaggle-style dataset directory."""
    data_dir = Path(data_dir)
    records = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            fp = os.path.join(root, f)
            try:
                with open(fp) as fh:
                    record = json.load(fh)
                record["_json_path"] = fp
                img_name = Path(f).stem
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = Path(root) / f"{img_name}{ext}"
                    if img_path.exists():
                        record["_img_path"] = str(img_path)
                        break
                records.append(record)
            except Exception:
                continue
    return records


def build_balanced_split(
    data_dir: str | Path,
    min_per_country: int = 50,
    max_per_country: int = 2000,
    seed: int = 42,
    target_countries: Optional[list[str]] = None,
) -> list[str]:
    """
    Build a balanced file list using the HSLU report's approach:
    - Filter to target countries (if provided)
    - Cap per-country counts
    - Shuffle within each country
    """
    import random
    random.seed(seed)

    data_dir = Path(data_dir)
    country_files: dict[str, list[str]] = {}

    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, f)) as fh:
                    meta = json.load(fh)
                country = _normalize_country(meta.get("country_name") or meta.get("country", "Unknown"))
                img_name = Path(f).stem
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = Path(root) / f"{img_name}{ext}"
                    if img_path.exists():
                        country_files.setdefault(country, []).append(str(img_path))
                        break
            except Exception:
                continue

    balanced = []
    for country, files in country_files.items():
        if target_countries and country not in target_countries:
            continue
        if len(files) < min_per_country:
            continue
        random.shuffle(files)
        selected = files[:max_per_country]
        balanced.extend(selected)

    random.shuffle(balanced)
    return balanced

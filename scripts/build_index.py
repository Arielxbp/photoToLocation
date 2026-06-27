#!/usr/bin/env python3
"""Build country/region/continent indices from the Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoguessr_agent.data.mapper import CountryMapper
from geoguessr_agent.geoutils import (
    build_cell_to_index,
    generate_s2_region_centroids,
    save_cell_to_index,
    save_region_centroids,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build country/region/continent indices from Kaggle dataset"
    )
    parser.add_argument(
        "--data-dir", default="data/kaggle",
        help="Directory containing Kaggle metadata files"
    )
    parser.add_argument(
        "--output", default="data/indices.json",
        help="Output path for the indices JSON file"
    )
    parser.add_argument(
        "--min-samples", type=int, default=50,
        help="Minimum number of samples per country"
    )
    parser.add_argument(
        "--s2-level", type=int, default=6,
        help="S2 cell level for region centroids"
    )
    args = parser.parse_args()

    print(f"Building indices from {args.data_dir}...")
    mapper = CountryMapper()
    country_index = mapper.build_from_files(args.data_dir, min_samples=args.min_samples)

    print(f"Found {len(country_index)} countries with >= {args.min_samples} samples")

    region_centroids = generate_s2_region_centroids(level=args.s2_level)
    print(f"Generated {region_centroids.shape[0]} S2 L{args.s2_level} region centroids")

    cell_to_idx = build_cell_to_index(level=args.s2_level)
    print(f"Built cell-id -> index mapping ({len(cell_to_idx)} entries)")

    continent_index = mapper.continent_to_idx

    indices = {
        "country_index": country_index,
        "idx_to_country": {str(k): v for k, v in mapper.idx_to_country.items()},
        "continent_index": continent_index,
        "idx_to_continent": {str(k): v for k, v in mapper.idx_to_continent.items()},
        "country_to_continent": {
            str(k): v for k, v in mapper.country_to_continent.items()
        },
        "num_countries": len(country_index),
        "num_regions": region_centroids.shape[0],
        "num_continents": len(continent_index),
        "s2_level": args.s2_level,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(indices, f, indent=2)

    centroids_path = Path(args.output).with_suffix(".centroids.pt")
    save_region_centroids(region_centroids, centroids_path)

    cell_idx_path = Path(args.output).with_suffix(".cell_to_idx.pt")
    save_cell_to_index(cell_to_idx, cell_idx_path)

    print(f"Saved indices -> {args.output}")
    print(f"Saved centroids -> {centroids_path}")
    print(f"Saved cell mapping -> {cell_idx_path}")


if __name__ == "__main__":
    main()

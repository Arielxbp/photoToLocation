#!/usr/bin/env python3
"""Build the Plonkit clue knowledge base from plonkit.net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoguessr_agent.plonkit.scraper import build_clue_kb


def main():
    parser = argparse.ArgumentParser(description="Build the Plonkit clue knowledge base")
    parser.add_argument(
        "--cache-dir", default="data/plonkit_cache",
        help="Directory to cache scraped country data"
    )
    args = parser.parse_args()

    print(f"Building Plonkit KB -> {args.cache_dir}")
    kb = build_clue_kb(cache_dir=args.cache_dir)
    print(f"Collected {len(kb)} countries")
    for code in sorted(kb)[:5]:
        name = kb[code].get("name", code)
        clues = [k for k in kb[code] if k not in ("name", "slug")]
        print(f"  {code} ({name}): {len(clues)} clue categories")


if __name__ == "__main__":
    main()

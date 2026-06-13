from pathlib import Path
from typing import Optional

from .scraper import build_clue_kb, load_clue_kb


class ClueKnowledgeBase:
    """In-memory clue knowledge base for model verification and curriculum learning."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        if not any(self.cache_dir.glob("*.json")):
            print(f"[KB] Cache empty, building from Plonkit...")
            self.data = build_clue_kb(cache_dir=self.cache_dir)
        else:
            self.data = load_clue_kb(self.cache_dir)

    def __getitem__(self, code: str) -> dict:
        return self.data.get(code, {})

    def __contains__(self, code: str) -> bool:
        return code in self.data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def country_codes(self) -> list[str]:
        return sorted(self.data.keys())

    def shared_clues(self, code_a: str, code_b: str) -> set[str]:
        """Return clue categories shared between two countries."""
        a = self.data.get(code_a, {})
        b = self.data.get(code_b, {})
        shared = set()
        for key in set(a.keys()) & set(b.keys()):
            if key in ("name", "slug"):
                continue
            if a[key] == b[key]:
                shared.add(key)
        return shared

    def hard_negatives(self, code: str, top_k: int = 5) -> list[tuple[str, int]]:
        """
        Find countries that share the most clues with `code`.
        These are "hard negatives" — visually confusable countries.
        """
        scores = []
        for other in self.country_codes:
            if other == code:
                continue
            shared = self.shared_clues(code, other)
            scores.append((other, len(shared)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def unique_clues(self, code: str) -> dict[str, list[str] | str]:
        """Return clues that are unique or rare for this country."""
        entry = self.data.get(code, {})
        unique: dict[str, list[str] | str] = {}
        for key, val in entry.items():
            if key in ("name", "slug"):
                continue
            shared_count = 0
            for other in self.country_codes:
                if other == code:
                    continue
                other_val = self.data.get(other, {}).get(key)
                if other_val == val:
                    shared_count += 1
            if shared_count <= 2:
                unique[key] = val
        return unique

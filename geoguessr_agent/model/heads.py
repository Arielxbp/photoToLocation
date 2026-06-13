from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountryHead(nn.Module):
    """Classification head for country prediction."""

    def __init__(self, in_dim: int, num_countries: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_countries),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RegionHead(nn.Module):
    """Classification head for region prediction (S2 cells or admin regions)."""

    def __init__(self, in_dim: int, num_regions: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, num_regions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ContinentHead(nn.Module):
    """Classification head for continent prediction."""

    def __init__(self, in_dim: int, num_continents: int = 7, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_continents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CoordinateHead(nn.Module):
    """Direct coordinate regression head. Use with caution — empirically poor."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = self.fc(x)
        return F.normalize(xyz, p=2, dim=-1)

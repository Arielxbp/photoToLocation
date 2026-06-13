from __future__ import annotations

import math


class RewardCalculator:
    """
    Calculates shaped rewards for geolocation guesses.

    Uses a combination of:
        - Haversine distance (primary signal)
        - Hierarchical bonuses for correct country/region/continent
        - Threshold bonuses for close guesses
    """

    def __init__(
        self,
        country_bonus: float = 5.0,
        region_bonus: float = 3.0,
        continent_bonus: float = 1.0,
        close_threshold_km: float = 100.0,
        close_bonus: float = 2.0,
        excellent_threshold_km: float = 10.0,
        excellent_bonus: float = 10.0,
    ):
        self.country_bonus = country_bonus
        self.region_bonus = region_bonus
        self.continent_bonus = continent_bonus
        self.close_threshold_km = close_threshold_km
        self.close_bonus = close_bonus
        self.excellent_threshold_km = excellent_threshold_km
        self.excellent_bonus = excellent_bonus

    def shaped_reward(
        self,
        pred_lat: float,
        pred_lng: float,
        true_lat: float,
        true_lng: float,
        same_country: bool = False,
        same_region: bool = False,
        same_continent: bool = False,
    ) -> float:
        """
        Compute shaped reward for a single guess.

        Base reward is negative distance in km (normalized).
        Bonuses are added for hierarchical correctness.
        """
        d_km = self._haversine(pred_lat, pred_lng, true_lat, true_lng)

        reward = -d_km / 1000.0

        if same_continent:
            reward += self.continent_bonus
        if same_country:
            reward += self.country_bonus
        if same_region:
            reward += self.region_bonus
        if d_km < self.excellent_threshold_km:
            reward += self.excellent_bonus
        elif d_km < self.close_threshold_km:
            reward += self.close_bonus

        return reward

    def raw_reward(self, pred_lat: float, pred_lng: float, true_lat: float, true_lng: float) -> float:
        """Simple negative haversine distance reward."""
        d_km = self._haversine(pred_lat, pred_lng, true_lat, true_lng)
        return -d_km

    @staticmethod
    def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Haversine distance in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

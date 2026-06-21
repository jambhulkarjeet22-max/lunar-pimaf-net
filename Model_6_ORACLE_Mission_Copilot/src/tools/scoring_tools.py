"""Multi-criteria scoring tools for lunar site selection."""

from __future__ import annotations

import numpy as np


def find_best_coordinate(score_map: np.ndarray) -> tuple[int, int, float]:
    """Find the coordinate with the maximum score in a 2D array."""
    idx = np.unravel_index(np.argmax(score_map), score_map.shape)
    return int(idx[0]), int(idx[1]), float(score_map[idx])


def score_landing_sites(
    suitability: np.ndarray,
    radiation_risk: np.ndarray,
    traversability: np.ndarray,
) -> np.ndarray:
    """Calculate combined score for landing sites."""
    return suitability * 0.5 + (1.0 - radiation_risk) * 0.3 + traversability * 0.2


def score_drilling_sites(
    ice_concentration: np.ndarray,
    ice_depth: np.ndarray,
    traversability: np.ndarray,
) -> np.ndarray:
    """Calculate combined score for drilling sites."""
    # Depth is in [0, 1] range, where lower is better (closer to surface)
    return ice_concentration * 0.6 + (1.0 - ice_depth) * 0.2 + traversability * 0.2


def score_habitat_sites(
    radiation_risk: np.ndarray,
    shielding: np.ndarray,
    suitability: np.ndarray,
) -> np.ndarray:
    """Calculate combined score for habitats."""
    return (1.0 - radiation_risk) * 0.4 + shielding * 0.4 + suitability * 0.2

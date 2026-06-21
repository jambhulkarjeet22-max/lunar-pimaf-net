"""Multi-task prediction heads for rover hazard and navigation intelligence."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "traversability_score",
    "crater_hazard_probability",
    "boulder_hazard_probability",
    "slope_hazard_score",
    "navigation_cost_map",
    "rover_safety_score",
)


class _ScoreHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NavigationHeads(nn.Module):
    """Predict traversability, crater/boulder/slope hazards, cost map, and safety score maps."""

    def __init__(self, in_channels: int = 64) -> None:
        super().__init__()
        self.trav_head = _ScoreHead(in_channels)
        self.crater_head = _ScoreHead(in_channels)
        self.boulder_head = _ScoreHead(in_channels)
        self.slope_head = _ScoreHead(in_channels)
        self.cost_head = _ScoreHead(in_channels)
        self.safety_head = _ScoreHead(in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "traversability_score": self.trav_head(x),
            "crater_hazard_probability": self.crater_head(x),
            "boulder_hazard_probability": self.boulder_head(x),
            "slope_hazard_score": self.slope_head(x),
            "navigation_cost_map": self.cost_head(x),
            "rover_safety_score": self.safety_head(x),
        }


__all__ = ["NavigationHeads", "OUTPUT_KEYS"]

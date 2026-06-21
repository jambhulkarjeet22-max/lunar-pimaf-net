"""Multi-task prediction heads for landing suitability scoring."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "landing_safety_score",
    "hazard_probability",
    "illumination_score",
    "resource_accessibility_score",
    "final_suitability_score",
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


class LandingHeads(nn.Module):
    """Predict safety, hazard, illumination, resource, and composite suitability maps."""

    def __init__(self, in_channels: int = 64) -> None:
        super().__init__()
        self.safety_score_head = _ScoreHead(in_channels)
        self.hazard_prob_head = _ScoreHead(in_channels)
        self.illumination_score_head = _ScoreHead(in_channels)
        self.resource_score_head = _ScoreHead(in_channels)
        self.suitability_score_head = _ScoreHead(in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "landing_safety_score": self.safety_score_head(x),
            "hazard_probability": self.hazard_prob_head(x),
            "illumination_score": self.illumination_score_head(x),
            "resource_accessibility_score": self.resource_score_head(x),
            "final_suitability_score": self.suitability_score_head(x),
        }


__all__ = ["LandingHeads", "OUTPUT_KEYS"]

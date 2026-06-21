"""Multi-task prediction heads for radiation risk prediction."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "radiation_dose_rate",
    "radiation_risk_score",
    "shielding_effectiveness_score",
    "habitat_safety_score",
    "final_radiation_hazard_map",
)


class _ScoreHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, activation: str = "sigmoid") -> None:
        super().__init__()
        act: nn.Module = nn.Sigmoid() if activation == "sigmoid" else nn.Softplus()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadiationHeads(nn.Module):
    """Predict dose rate, risk score, shielding effectiveness, habitat safety, and final hazard map."""

    def __init__(self, in_channels: int = 64) -> None:
        super().__init__()
        self.dose_head = _ScoreHead(in_channels, activation="softplus")
        self.risk_head = _ScoreHead(in_channels, activation="sigmoid")
        self.shielding_head = _ScoreHead(in_channels, activation="sigmoid")
        self.safety_head = _ScoreHead(in_channels, activation="sigmoid")
        self.hazard_head = _ScoreHead(in_channels, activation="sigmoid")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "radiation_dose_rate": self.dose_head(x),
            "radiation_risk_score": self.risk_head(x),
            "shielding_effectiveness_score": self.shielding_head(x),
            "habitat_safety_score": self.safety_head(x),
            "final_radiation_hazard_map": self.hazard_head(x),
        }


__all__ = ["RadiationHeads", "OUTPUT_KEYS"]

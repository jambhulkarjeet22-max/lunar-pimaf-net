"""Multi-modal terrain encoders for rover hazard and navigation intelligence."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

MODALITY_CHANNELS: Final[dict[str, int]] = {
    "lola": 1,
    "mini_rf": 3,
    "dem": 1,
    "slope": 1,
    "crater": 1,
    "boulder": 1,
    "illumination": 1,
}

DEFAULT_FEATURE_DIM: Final[int] = 32


class TerrainEncoder(nn.Module):
    """Convolutional encoder for a single orbital/terrain modality."""

    def __init__(self, in_channels: int, out_channels: int = DEFAULT_FEATURE_DIM) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiModalTerrainEncoder(nn.Module):
    """Independent encoders for LOLA, Mini-RF, DEM, slope, crater, boulder, and illumination inputs."""

    def __init__(
        self,
        channels_per_modality: dict[str, int] | None = None,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        channels = channels_per_modality or dict(MODALITY_CHANNELS)
        self.modalities: tuple[str, ...] = tuple(sorted(channels.keys()))
        self.encoders = nn.ModuleDict(
            {name: TerrainEncoder(in_ch, feature_dim) for name, in_ch in channels.items()}
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.modalities if name not in inputs]
        if missing:
            raise ValueError(f"Missing modality inputs: {', '.join(missing)}")

        return {name: self.encoders[name](inputs[name]) for name in self.modalities}


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "MODALITY_CHANNELS",
    "MultiModalTerrainEncoder",
    "TerrainEncoder",
]

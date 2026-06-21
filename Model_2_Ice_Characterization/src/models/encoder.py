from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Processes a single modality channel into a dense feature map."""
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder processing different lunar instruments."""
    def __init__(self, channels_per_modality: dict[str, int], feature_dim: int = 32):
        super().__init__()
        self.modalities = sorted(list(channels_per_modality.keys()))
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(ch, feature_dim)
            for mod, ch in channels_per_modality.items()
        })

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = {}
        for mod in self.modalities:
            if mod in inputs:
                features[mod] = self.encoders[mod](inputs[mod])
        return features

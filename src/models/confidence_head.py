"""Prediction confidence head for LUNAR-PIMAF-Net.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §9.4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.decoder import DECODER_OUTPUT_CHANNELS
from src.models.prediction_heads import _validate_decoder_features


class ConfidenceHead(nn.Module):
    """Predict per-pixel model confidence in ``[0, 1]``.

    Trained against weak-label confidence from the preprocessing pipeline and
  optionally combined at inference with evidential and physics residuals.
    """

    def __init__(self, in_channels: int = DECODER_OUTPUT_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return confidence map ``(B, 1, 128, 128)``."""
        _validate_decoder_features(features)
        return torch.sigmoid(self.conv(features))


__all__ = ["ConfidenceHead"]

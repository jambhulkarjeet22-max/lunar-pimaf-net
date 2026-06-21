"""Multi-task prediction heads for LUNAR-PIMAF-Net ice detection.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §9.1–§9.2
"""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn

from src.models.decoder import DECODER_OUTPUT_CHANNELS

NUM_SEGMENTATION_CLASSES: Final[int] = 3


class PredictionOutputs(TypedDict):
    """Bundled outputs from ``MultiTaskPredictionHead``."""

    segmentation_logits: torch.Tensor
    surface_ice_probability: torch.Tensor
    subsurface_ice_probability: torch.Tensor


def _validate_decoder_features(features: torch.Tensor) -> None:
    """Validate decoder embedding shape ``(B, 128, 128, 128)``."""
    if features.dim() != 4:
        raise ValueError(
            f"Decoder features must be 4-D (B, C, H, W), got {tuple(features.shape)}."
        )
    if features.shape[1] != DECODER_OUTPUT_CHANNELS:
        raise ValueError(
            f"Decoder features expected {DECODER_OUTPUT_CHANNELS} channels, "
            f"got {features.shape[1]}."
        )
    if tuple(features.shape[2:]) != (128, 128):
        raise ValueError(
            f"Decoder features expected spatial size (128, 128), "
            f"got ({features.shape[2]}, {features.shape[3]})."
        )


class SegmentationHead(nn.Module):
    """Three-class segmentation logits: no ice / surface / subsurface."""

    def __init__(
        self,
        in_channels: int = DECODER_OUTPUT_CHANNELS,
        num_classes: int = NUM_SEGMENTATION_CLASSES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(B, 3, 128, 128)``."""
        _validate_decoder_features(features)
        return self.conv(features)


class IceProbabilityHead(nn.Module):
    """Independent sigmoid probabilities for surface and subsurface ice."""

    def __init__(self, in_channels: int = DECODER_OUTPUT_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(P_surface, P_subsurface)`` each ``(B, 1, 128, 128)``."""
        _validate_decoder_features(features)
        probs = torch.sigmoid(self.conv(features))
        return probs[:, 0:1], probs[:, 1:2]


class MultiTaskPredictionHead(nn.Module):
    """Segmentation and ice-probability heads sharing decoder features."""

    def __init__(
        self,
        in_channels: int = DECODER_OUTPUT_CHANNELS,
        num_classes: int = NUM_SEGMENTATION_CLASSES,
    ) -> None:
        super().__init__()
        self.segmentation = SegmentationHead(in_channels, num_classes)
        self.ice_probability = IceProbabilityHead(in_channels)

    def forward(self, features: torch.Tensor) -> PredictionOutputs:
        """Run all prediction heads on decoder features."""
        _validate_decoder_features(features)
        logits = self.segmentation(features)
        p_surface, p_subsurface = self.ice_probability(features)
        return PredictionOutputs(
            segmentation_logits=logits,
            surface_ice_probability=p_surface,
            subsurface_ice_probability=p_subsurface,
        )


__all__ = [
    "NUM_SEGMENTATION_CLASSES",
    "IceProbabilityHead",
    "MultiTaskPredictionHead",
    "PredictionOutputs",
    "SegmentationHead",
]

"""Orchestrator for Lunar Rover Hazard & Navigation Intelligence network."""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AttentionFusion
from .heads import OUTPUT_KEYS, NavigationHeads
from .terrain_encoder import MODALITY_CHANNELS, MultiModalTerrainEncoder

DEFAULT_PATCH_SIZE: Final[int] = 128


class NavigationOutput(TypedDict):
    traversability_score: torch.Tensor
    crater_hazard_probability: torch.Tensor
    boulder_hazard_probability: torch.Tensor
    slope_hazard_score: torch.Tensor
    navigation_cost_map: torch.Tensor
    rover_safety_score: torch.Tensor


class RoverNavigationNet(nn.Module):
    """Multi-modal lunar rover traversability and hazard prediction model.

    Encodes seven orbital/topographic modalities, fuses them using learned attention,
    and predicts six traversability, cost, and hazard maps in ``[0, 1]``.
    """

    def __init__(
        self,
        channels_per_modality: dict[str, int] | None = None,
        feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.channels_per_modality = channels_per_modality or dict(MODALITY_CHANNELS)
        self.feature_dim = feature_dim
        num_modalities = len(self.channels_per_modality)

        self.encoder = MultiModalTerrainEncoder(self.channels_per_modality, feature_dim=feature_dim)
        self.fusion = AttentionFusion(feature_dim=feature_dim, num_modalities=num_modalities)
        self.heads = NavigationHeads(in_channels=feature_dim * 2)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted(self.channels_per_modality.keys()))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reference = inputs[self.modalities[0]]
        if reference.dim() != 4:
            raise ValueError(f"Expected 4-D modality tensors, got {reference.dim()}-D.")

        encoded = self.encoder(inputs)
        fused = self.fusion(encoded)
        predictions = self.heads(fused)

        target_size = reference.shape[-2:]
        upsampled: dict[str, torch.Tensor] = {}
        for key in OUTPUT_KEYS:
            tensor = predictions[key]
            if tensor.shape[-2:] != target_size:
                tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
            upsampled[key] = tensor
        return upsampled


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "RoverNavigationNet",
    "NavigationOutput",
    "OUTPUT_KEYS",
]

"""Landing Site Intelligence network orchestration."""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AttentionFusion
from .heads import OUTPUT_KEYS, LandingHeads
from .terrain_encoder import MODALITY_CHANNELS, MultiModalTerrainEncoder

DEFAULT_PATCH_SIZE: Final[int] = 128


class LandingSiteOutput(TypedDict):
    landing_safety_score: torch.Tensor
    hazard_probability: torch.Tensor
    illumination_score: torch.Tensor
    resource_accessibility_score: torch.Tensor
    final_suitability_score: torch.Tensor


class LandingSiteNet(nn.Module):
    """Multi-modal lunar landing site intelligence model.

    Encodes six orbital modalities, fuses them with learned attention weights,
    and predicts five geospatial suitability maps in ``[0, 1]``.
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
        self.heads = LandingHeads(in_channels=feature_dim * 2)

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
            upsampled[key] = tensor.clamp(0.0, 1.0)
        return upsampled


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "LandingSiteNet",
    "LandingSiteOutput",
    "OUTPUT_KEYS",
]

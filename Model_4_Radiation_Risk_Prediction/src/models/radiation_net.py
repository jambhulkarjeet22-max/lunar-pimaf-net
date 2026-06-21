"""Orchestrator for Lunar Radiation Risk Prediction network."""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AttentionFusion
from .heads import OUTPUT_KEYS, RadiationHeads
from .radiation_encoder import MODALITY_CHANNELS, MultiModalRadiationEncoder

DEFAULT_PATCH_SIZE: Final[int] = 128


class RadiationOutput(TypedDict):
    radiation_dose_rate: torch.Tensor
    radiation_risk_score: torch.Tensor
    shielding_effectiveness_score: torch.Tensor
    habitat_safety_score: torch.Tensor
    final_radiation_hazard_map: torch.Tensor


class RadiationNet(nn.Module):
    """Multi-modal lunar radiation risk prediction model.

    Encodes seven orbital/topographic modalities, fuses them with attention,
    and predicts five multi-task radiation and shielding maps.
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

        self.encoder = MultiModalRadiationEncoder(self.channels_per_modality, feature_dim=feature_dim)
        self.fusion = AttentionFusion(feature_dim=feature_dim, num_modalities=num_modalities)
        self.heads = RadiationHeads(in_channels=feature_dim * 2)

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
    "RadiationNet",
    "RadiationOutput",
    "OUTPUT_KEYS",
]

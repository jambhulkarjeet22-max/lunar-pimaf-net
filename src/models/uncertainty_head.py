"""Evidential uncertainty estimation for LUNAR-PIMAF-Net.

Implements Dirichlet-based uncertainty decomposition (Sensoy et al., 2018).

Reference: docs/ARCHITECTURE_SPECIFICATION.md §9.3, §11
"""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import DECODER_OUTPUT_CHANNELS
from src.models.prediction_heads import NUM_SEGMENTATION_CLASSES, _validate_decoder_features

NUM_EVIDENTIAL_CLASSES: Final[int] = NUM_SEGMENTATION_CLASSES


class UncertaintyOutputs(TypedDict):
    """Evidential uncertainty decomposition."""

    dirichlet_alpha: torch.Tensor
    expected_probability: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor


class EvidentialHead(nn.Module):
    """Maps decoder features to Dirichlet concentration parameters ``α``."""

    def __init__(
        self,
        in_channels: int = DECODER_OUTPUT_CHANNELS,
        num_classes: int = NUM_EVIDENTIAL_CLASSES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return Dirichlet parameters ``α`` of shape ``(B, K, 128, 128)``, ``α > 1``."""
        _validate_decoder_features(features)
        return F.softplus(self.conv(features)) + 1.0


class UncertaintyEstimator(nn.Module):
    """Derive epistemic and aleatoric uncertainty from Dirichlet evidence."""

    def __init__(self, num_classes: int = NUM_EVIDENTIAL_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, alpha: torch.Tensor) -> UncertaintyOutputs:
        """Compute uncertainty maps from Dirichlet concentrations.

        Args:
            alpha: Dirichlet parameters ``(B, K, H, W)`` with ``α_k > 1``.

        Returns:
            Dictionary containing ``α``, expected probabilities, and uncertainty maps.
        """
        if alpha.dim() != 4:
            raise ValueError(
                f"alpha must be 4-D (B, K, H, W), got shape {tuple(alpha.shape)}."
            )
        if alpha.shape[1] != self.num_classes:
            raise ValueError(
                f"alpha expected {self.num_classes} classes, got {alpha.shape[1]}."
            )
        if torch.any(alpha <= 1.0):
            raise ValueError("Dirichlet alpha parameters must be strictly greater than 1.")

        sum_alpha = alpha.sum(dim=1, keepdim=True)
        expected_prob = alpha / sum_alpha

        epistemic = self.num_classes / sum_alpha
        aleatoric = self._dirichlet_aleatoric(alpha, sum_alpha, expected_prob)
        total = epistemic + aleatoric

        return UncertaintyOutputs(
            dirichlet_alpha=alpha,
            expected_probability=expected_prob,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
        )

    def _dirichlet_aleatoric(
        self,
        alpha: torch.Tensor,
        sum_alpha: torch.Tensor,
        expected_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dirichlet discord (data ambiguity) per pixel."""
        safe_prob = expected_prob.clamp_min(1e-8)
        digamma_sum = torch.digamma(sum_alpha)
        digamma_alpha = torch.digamma(alpha + 1.0)
        entropy_term = -torch.sum(
            safe_prob * (torch.log(safe_prob) - digamma_alpha + digamma_sum),
            dim=1,
            keepdim=True,
        )
        return entropy_term.clamp(min=0.0)


class EvidentialUncertaintyModule(nn.Module):
    """Combined evidential head and uncertainty estimator."""

    def __init__(
        self,
        in_channels: int = DECODER_OUTPUT_CHANNELS,
        num_classes: int = NUM_EVIDENTIAL_CLASSES,
    ) -> None:
        super().__init__()
        self.head = EvidentialHead(in_channels, num_classes)
        self.estimator = UncertaintyEstimator(num_classes)

    def forward(self, features: torch.Tensor) -> UncertaintyOutputs:
        alpha = self.head(features)
        return self.estimator(alpha)


__all__ = [
    "NUM_EVIDENTIAL_CLASSES",
    "EvidentialHead",
    "EvidentialUncertaintyModule",
    "UncertaintyEstimator",
    "UncertaintyOutputs",
]

"""Loss functions for LUNAR-PIMAF-Net training.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.physics_constraint_module import PhysicsResiduals
from src.models.prediction_heads import NUM_SEGMENTATION_CLASSES

DEFAULT_CLASS_WEIGHTS: Final[tuple[float, ...]] = (1.0, 2.5, 4.0)
DEFAULT_FOCAL_GAMMA: Final[float] = 2.0
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.4


@dataclass
class LossWeights:
    """Scalar weights for ``LunarPIMAFLoss`` composition."""

    focal: float = 1.0
    evidential: float = 0.5
    physics: float = 0.3
    probability: float = 0.4
    confidence: float = 0.2
    smoothness: float = 0.05


@dataclass
class PhysicsLossWeights:
    """Sub-weights inside ``PhysicsLoss``."""

    stefan: float = 1.0
    stability: float = 2.0
    radar: float = 0.8
    neutron: float = 0.6
    anchor: float = 1.5


@dataclass
class LunarPIMAFLossOutput:
    """Detailed loss breakdown."""

    total: torch.Tensor
    focal: torch.Tensor
    evidential: torch.Tensor
    physics: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor
    smoothness: torch.Tensor


class SoftFocalLoss(nn.Module):
    """Focal loss for soft multi-class segmentation labels.

    Supports weak labels ``y_soft`` of shape ``(B, K, H, W)`` and per-pixel
    confidence weights.
    """

    def __init__(
        self,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        class_weights: tuple[float, ...] = DEFAULT_CLASS_WEIGHTS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        super().__init__()
        if len(class_weights) != NUM_SEGMENTATION_CLASSES:
            raise ValueError(
                f"class_weights must have {NUM_SEGMENTATION_CLASSES} entries."
            )
        self.gamma = gamma
        self.confidence_threshold = confidence_threshold
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        y_soft: torch.Tensor,
        pixel_confidence: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != y_soft.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} must match y_soft {tuple(y_soft.shape)}."
            )
        if pixel_confidence.dim() != 4 or pixel_confidence.shape[1] != 1:
            raise ValueError(
                "pixel_confidence must have shape (B, 1, H, W), "
                f"got {tuple(pixel_confidence.shape)}."
            )

        probs = F.softmax(logits, dim=1).clamp_min(1e-8)
        focal_weight = (1.0 - probs).pow(self.gamma)
        ce = -y_soft * torch.log(probs)

        class_w = self.class_weights.view(1, -1, 1, 1)
        loss_map = pixel_confidence * class_w * focal_weight * ce
        valid = (pixel_confidence >= self.confidence_threshold).float()
        denom = valid.sum().clamp_min(1.0)
        return (loss_map.sum(dim=1, keepdim=True) * valid).sum() / denom


class EvidentialLoss(nn.Module):
    """Type-II maximum likelihood loss for Dirichlet evidence (Sensoy et al., 2018)."""

    def __init__(
        self,
        num_classes: int = NUM_SEGMENTATION_CLASSES,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight

    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        kl_annealing: float = 1.0,
    ) -> torch.Tensor:
        if alpha.dim() != 4 or alpha.shape[1] != self.num_classes:
            raise ValueError(
                f"alpha must have shape (B, {self.num_classes}, H, W), "
                f"got {tuple(alpha.shape)}."
            )
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.shape[1] != 1:
            raise ValueError("target must be class indices with shape (B, 1, H, W).")

        target = target.long().clamp(0, self.num_classes - 1)
        sum_alpha = alpha.sum(dim=1, keepdim=True)

        target_one_hot = torch.zeros_like(alpha)
        target_one_hot.scatter_(1, target, 1.0)

        alpha_target = (alpha * target_one_hot).sum(dim=1, keepdim=True)
        mismatch = alpha * (1.0 - target_one_hot)

        loss_map = (
            torch.log(sum_alpha)
            - torch.log(alpha_target.clamp_min(1e-8))
            + mismatch * (torch.log(alpha_target.clamp_min(1e-8)) - torch.log(alpha.clamp_min(1e-8)))
        ).sum(dim=1, keepdim=True)

        kl = self._dirichlet_kl(alpha) * self.kl_weight * kl_annealing
        return (loss_map + kl).mean()

    @staticmethod
    def _dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
        """KL(Dir(α) || Dir(1)) summed over classes."""
        num_classes = alpha.shape[1]
        sum_alpha = alpha.sum(dim=1, keepdim=True)
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        term2 = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
        digamma_diff = torch.digamma(alpha) - torch.digamma(sum_alpha)
        term3 = torch.sum((alpha - 1.0) * digamma_diff, dim=1, keepdim=True)
        return term1 - term2 + term3


class PhysicsLoss(nn.Module):
    """Aggregate physics residual and latent-anchor penalties."""

    def __init__(self, weights: PhysicsLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or PhysicsLossWeights()

    def forward(
        self,
        residuals: PhysicsResiduals,
        latent_physics: torch.Tensor,
        physics_priors: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stefan = residuals["stefan_residual"].mean()
        stability = residuals["stability_residual"].mean()
        radar = residuals["radar_residual"].mean()
        neutron = residuals["neutron_residual"].mean()

        anchor_channels = min(latent_physics.shape[1], physics_priors.shape[1])
        latent = latent_physics[:, :anchor_channels]
        priors = physics_priors[:, :anchor_channels]
        anchor = F.mse_loss(latent, priors, reduction="none")
        if prior_mask is not None:
            anchor = (anchor * prior_mask).sum() / prior_mask.sum().clamp_min(1.0)
        else:
            anchor = anchor.mean()

        return (
            self.weights.stefan * stefan
            + self.weights.stability * stability
            + self.weights.radar * radar
            + self.weights.neutron * neutron
            + self.weights.anchor * anchor
        )


class IceProbabilityLoss(nn.Module):
    """Binary cross-entropy on surface and subsurface ice probabilities."""

    def __init__(self, surface_weight: float = 0.5) -> None:
        super().__init__()
        self.surface_weight = surface_weight

    def forward(
        self,
        p_surface: torch.Tensor,
        p_subsurface: torch.Tensor,
        y_soft: torch.Tensor,
        pixel_confidence: torch.Tensor,
    ) -> torch.Tensor:
        if y_soft.shape[1] != NUM_SEGMENTATION_CLASSES:
            raise ValueError(
                f"y_soft must have {NUM_SEGMENTATION_CLASSES} channels, "
                f"got {y_soft.shape[1]}."
            )
        valid = (pixel_confidence >= DEFAULT_CONFIDENCE_THRESHOLD).float()
        denom = valid.sum().clamp_min(1.0)

        loss_surface = F.binary_cross_entropy(p_surface, y_soft[:, 1:2], reduction="none")
        loss_subsurface = F.binary_cross_entropy(p_subsurface, y_soft[:, 2:3], reduction="none")

        weighted = (
            loss_subsurface + self.surface_weight * loss_surface
        ) * valid
        return weighted.sum() / denom


class ConfidenceLoss(nn.Module):
    """MSE and correctness BCE for confidence calibration."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        conf_pred: torch.Tensor,
        conf_target: torch.Tensor,
        predicted_class: torch.Tensor,
        target_class: torch.Tensor,
        pixel_confidence: torch.Tensor,
    ) -> torch.Tensor:
        valid = (pixel_confidence >= DEFAULT_CONFIDENCE_THRESHOLD).float()
        denom = valid.sum().clamp_min(1.0)

        mse = F.mse_loss(conf_pred, conf_target, reduction="none")
        correct = (predicted_class == target_class).float()
        bce = F.binary_cross_entropy(conf_pred, correct, reduction="none")
        loss = (mse + bce) * valid
        return loss.sum() / denom


class SmoothnessLoss(nn.Module):
    """Total-variation penalty on subsurface probability inside PSR interiors."""

    def forward(
        self,
        p_subsurface: torch.Tensor,
        psr_interior_mask: torch.Tensor,
    ) -> torch.Tensor:
        if p_subsurface.shape != psr_interior_mask.shape:
            raise ValueError(
                "p_subsurface and psr_interior_mask must have identical shapes, "
                f"got {tuple(p_subsurface.shape)} and {tuple(psr_interior_mask.shape)}."
            )

        dx = torch.abs(p_subsurface[:, :, :, 1:] - p_subsurface[:, :, :, :-1])
        dy = torch.abs(p_subsurface[:, :, 1:, :] - p_subsurface[:, :, :-1, :])
        mask_x = psr_interior_mask[:, :, :, 1:] * psr_interior_mask[:, :, :, :-1]
        mask_y = psr_interior_mask[:, :, 1:, :] * psr_interior_mask[:, :, :-1, :]

        tv = dx * mask_x
        tv_y = dy * mask_y
        denom = (mask_x.sum() + mask_y.sum()).clamp_min(1.0)
        return (tv.sum() + tv_y.sum()) / denom


class LunarPIMAFLoss(nn.Module):
    """Composite training objective for LUNAR-PIMAF-Net.

    ``L_total`` combines focal segmentation, evidential, physics, probability,
    confidence, and smoothness terms with architecture-default weights.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        physics_weights: PhysicsLossWeights | None = None,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.focal = SoftFocalLoss(gamma=focal_gamma)
        self.evidential = EvidentialLoss()
        self.physics = PhysicsLoss(physics_weights)
        self.probability = IceProbabilityLoss()
        self.confidence = ConfidenceLoss()
        self.smoothness = SmoothnessLoss()

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        dirichlet_alpha: torch.Tensor,
        p_surface: torch.Tensor,
        p_subsurface: torch.Tensor,
        conf_pred: torch.Tensor,
        residuals: PhysicsResiduals,
        latent_physics: torch.Tensor,
        physics_priors: torch.Tensor,
        y_soft: torch.Tensor,
        target_class: torch.Tensor,
        pixel_confidence: torch.Tensor,
        conf_target: torch.Tensor,
        psr_interior_mask: torch.Tensor,
        kl_annealing: float = 1.0,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> LunarPIMAFLossOutput:
        focal_loss = self.focal(segmentation_logits, y_soft, pixel_confidence)
        evidential_loss = self.evidential(dirichlet_alpha, target_class, kl_annealing)
        physics_loss = self.physics(residuals, latent_physics, physics_priors, prior_mask)
        probability_loss = self.probability(
            p_surface, p_subsurface, y_soft, pixel_confidence
        )

        predicted_class = torch.argmax(segmentation_logits, dim=1, keepdim=True)
        confidence_loss = self.confidence(
            conf_pred,
            conf_target,
            predicted_class,
            target_class,
            pixel_confidence,
        )
        smoothness_loss = self.smoothness(p_subsurface, psr_interior_mask)

        total = (
            self.weights.focal * focal_loss
            + self.weights.evidential * evidential_loss
            + self.weights.physics * physics_loss
            + self.weights.probability * probability_loss
            + self.weights.confidence * confidence_loss
            + self.weights.smoothness * smoothness_loss
        )

        return LunarPIMAFLossOutput(
            total=total,
            focal=focal_loss,
            evidential=evidential_loss,
            physics=physics_loss,
            probability=probability_loss,
            confidence=confidence_loss,
            smoothness=smoothness_loss,
        )


__all__ = [
    "ConfidenceLoss",
    "DEFAULT_CLASS_WEIGHTS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_FOCAL_GAMMA",
    "EvidentialLoss",
    "IceProbabilityLoss",
    "LossWeights",
    "LunarPIMAFLoss",
    "LunarPIMAFLossOutput",
    "PhysicsLoss",
    "PhysicsLossWeights",
    "SmoothnessLoss",
    "SoftFocalLoss",
]

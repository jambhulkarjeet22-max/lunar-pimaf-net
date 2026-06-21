"""Physics Constraint Module (PCM) for LUNAR-PIMAF-Net.

Differentiable mid-network physics residuals enforcing thermodynamic,
dielectric, and neutron-consistency constraints on ice predictions.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §7
"""

from __future__ import annotations

from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_pyramid import DEFAULT_FPN_OUT_CHANNELS

STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8
T_TRAP_K: Final[float] = 110.0
EPSILON_ICE: Final[float] = 3.15
EPSILON_REGOLITH: Final[float] = 2.5
SLOPE_ROUGHNESS_DEG: Final[float] = 10.0
NUM_PHYSICS_PRIORS: Final[int] = 7
NUM_LATENT_PHYSICS: Final[int] = 5


class PhysicsInputs(TypedDict, total=False):
    """Observations and predictions required for physics residuals."""

    fused_features: torch.Tensor
    div_tbol_max: torch.Tensor
    div_insolation: torch.Tensor
    albedo_proxy: torch.Tensor
    lola_slope_deg: torch.Tensor
    lend_h_wt_pct: torch.Tensor
    lend_valid: torch.Tensor
    physics_priors: torch.Tensor
    subsurface_ice_probability: torch.Tensor


class PhysicsResiduals(TypedDict):
    """Normalized physics violation maps."""

    stefan_residual: torch.Tensor
    stability_residual: torch.Tensor
    radar_residual: torch.Tensor
    neutron_residual: torch.Tensor


class PhysicsConstraintOutputs(TypedDict):
    """PCM forward outputs."""

    residuals: PhysicsResiduals
    latent_physics: torch.Tensor
    corrected_features: torch.Tensor


def _validate_feature_map(tensor: torch.Tensor, name: str, channels: int) -> None:
    if tensor.dim() != 4:
        raise ValueError(f"{name} must be 4-D (B, C, H, W), got {tuple(tensor.shape)}.")
    if tensor.shape[1] != channels:
        raise ValueError(f"{name} expected {channels} channels, got {tensor.shape[1]}.")


def _resize_to(
    tensor: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


class PhysicsConstraintModule(nn.Module):
    """Compute physics residuals and feature corrections from fused representations.

    Latent physics variables are projected from ``fused_features`` and compared
    against observational inputs and model ice predictions to produce four
    residual maps used by the physics loss and monitoring heads.
    """

    def __init__(
        self,
        feature_channels: int = DEFAULT_FPN_OUT_CHANNELS,
        latent_channels: int = NUM_LATENT_PHYSICS,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.latent_channels = latent_channels

        self.latent_proj = nn.Conv2d(feature_channels, latent_channels, kernel_size=1)
        self.correction = nn.Conv2d(
            feature_channels + 4,
            feature_channels,
            kernel_size=1,
        )

        self.stefan_scale = nn.Parameter(torch.tensor(1.0))
        self.stability_scale = nn.Parameter(torch.tensor(1.0))
        self.radar_scale = nn.Parameter(torch.tensor(1.0))
        self.neutron_scale = nn.Parameter(torch.tensor(1.0))

    def _project_latent(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Project fused features to interpretable latent physics fields."""
        latent = self.latent_proj(fused_features)

        t_max = F.relu(latent[:, 0:1])
        emissivity = torch.sigmoid(latent[:, 1:2]) * 0.1 + 0.9
        dielectric = torch.sigmoid(latent[:, 2:3]) * 5.0 + 1.0
        hydrogen = torch.sigmoid(latent[:, 3:4])
        stability = torch.sigmoid(latent[:, 4:5])

        return torch.cat([t_max, emissivity, dielectric, hydrogen, stability], dim=1)

    @staticmethod
    def _maxwell_garnett_dielectric(ice_fraction: torch.Tensor) -> torch.Tensor:
        """Effective dielectric constant via Maxwell-Garnett mixing."""
        eps_m = torch.full_like(ice_fraction, EPSILON_REGOLITH)
        eps_i = torch.full_like(ice_fraction, EPSILON_ICE)
        numerator = eps_m * (
            eps_i + 2.0 * eps_m + 2.0 * ice_fraction * (eps_i - eps_m)
        )
        denominator = eps_i + 2.0 * eps_m - ice_fraction * (eps_i - eps_m)
        return numerator / denominator.clamp_min(1e-6)

    @staticmethod
    def _normalize_residual(residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Map unbounded residuals to ``[0, 1]`` for stable loss weighting."""
        return torch.sigmoid(residual * scale)

    def compute_residuals(
        self,
        latent: torch.Tensor,
        div_tbol_max: torch.Tensor,
        div_insolation: torch.Tensor,
        albedo_proxy: torch.Tensor,
        lola_slope_deg: torch.Tensor,
        lend_h_wt_pct: torch.Tensor,
        lend_valid: torch.Tensor,
        subsurface_ice_probability: torch.Tensor,
    ) -> PhysicsResiduals:
        """Compute the four physics residual maps at decoder resolution."""
        height, width = subsurface_ice_probability.shape[-2:]
        latent = _resize_to(latent, height, width)

        t_max = latent[:, 0:1]
        emissivity = latent[:, 1:2]
        dielectric = latent[:, 2:3]
        hydrogen = latent[:, 3:4]

        insolation = _resize_to(div_insolation, height, width)
        albedo = _resize_to(albedo_proxy, height, width).clamp(0.0, 1.0)
        observed_t = _resize_to(div_tbol_max, height, width)
        slope = _resize_to(lola_slope_deg, height, width)
        lend_h = _resize_to(lend_h_wt_pct, height, width)
        lend_mask = _resize_to(lend_valid, height, width).clamp(0.0, 1.0)
        p_subsurface = subsurface_ice_probability.clamp(0.0, 1.0)

        predicted_radiance = t_max.pow(4)
        expected_radiance = (1.0 - albedo) * insolation / (
            emissivity * STEFAN_BOLTZMANN + 1e-8
        )
        stefan = torch.abs(predicted_radiance - expected_radiance)
        stefan = stefan / (expected_radiance.abs() + 1.0)
        stefan = self._normalize_residual(stefan, self.stefan_scale)

        stability = F.relu(t_max - T_TRAP_K) * p_subsurface
        stability = self._normalize_residual(stability, self.stability_scale)

        expected_dielectric = self._maxwell_garnett_dielectric(hydrogen)
        roughness_mask = (slope > SLOPE_ROUGHNESS_DEG).float()
        radar = torch.abs(dielectric - expected_dielectric) * (1.0 - roughness_mask)
        radar = self._normalize_residual(radar, self.radar_scale)

        neutron = (hydrogen - lend_h).pow(2) * lend_mask
        neutron = self._normalize_residual(neutron, self.neutron_scale)

        return PhysicsResiduals(
            stefan_residual=stefan,
            stability_residual=stability,
            radar_residual=radar,
            neutron_residual=neutron,
        )

    def forward(self, inputs: PhysicsInputs) -> PhysicsConstraintOutputs:
        """Run PCM on fused features and observational context.

        Args:
            inputs: Dictionary containing at minimum ``fused_features``,
                ``div_tbol_max``, ``div_insolation``, ``albedo_proxy``,
                ``lola_slope_deg``, ``lend_h_wt_pct``, ``lend_valid``, and
                ``subsurface_ice_probability``.

        Returns:
            Residual maps, latent physics fields (upsampled to prediction resolution),
            and physics-corrected fused features.
        """
        required = (
            "fused_features",
            "div_tbol_max",
            "div_insolation",
            "albedo_proxy",
            "lola_slope_deg",
            "lend_h_wt_pct",
            "lend_valid",
            "subsurface_ice_probability",
        )
        for key in required:
            if key not in inputs:
                raise ValueError(f"PhysicsConstraintModule missing required input '{key}'.")

        fused = inputs["fused_features"]
        _validate_feature_map(fused, "fused_features", self.feature_channels)

        latent = self._project_latent(fused)
        residuals = self.compute_residuals(
            latent=latent,
            div_tbol_max=inputs["div_tbol_max"],
            div_insolation=inputs["div_insolation"],
            albedo_proxy=inputs["albedo_proxy"],
            lola_slope_deg=inputs["lola_slope_deg"],
            lend_h_wt_pct=inputs["lend_h_wt_pct"],
            lend_valid=inputs["lend_valid"],
            subsurface_ice_probability=inputs["subsurface_ice_probability"],
        )

        target_h, target_w = inputs["subsurface_ice_probability"].shape[-2:]
        latent_full = _resize_to(latent, target_h, target_w)
        fused_full = _resize_to(fused, target_h, target_w)

        residual_stack = torch.cat(
            [
                residuals["stefan_residual"],
                residuals["stability_residual"],
                residuals["radar_residual"],
                residuals["neutron_residual"],
            ],
            dim=1,
        )
        corrected = fused_full + self.correction(torch.cat([fused_full, residual_stack], dim=1))

        return PhysicsConstraintOutputs(
            residuals=residuals,
            latent_physics=latent_full,
            corrected_features=corrected,
        )


__all__ = [
    "NUM_LATENT_PHYSICS",
    "NUM_PHYSICS_PRIORS",
    "STEFAN_BOLTZMANN",
    "T_TRAP_K",
    "PhysicsConstraintModule",
    "PhysicsConstraintOutputs",
    "PhysicsInputs",
    "PhysicsResiduals",
]

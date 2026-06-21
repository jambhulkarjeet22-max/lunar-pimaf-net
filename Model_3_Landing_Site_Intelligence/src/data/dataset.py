"""Synthetic and collated datasets for landing site intelligence."""

from __future__ import annotations

from typing import Any, Final

import torch
from torch.utils.data import Dataset

from Model_3_Landing_Site_Intelligence.src.models.landing_site_net import DEFAULT_PATCH_SIZE
from Model_3_Landing_Site_Intelligence.src.models.terrain_encoder import MODALITY_CHANNELS

DEFAULT_PATCH_SIZE_DATA: Final[int] = DEFAULT_PATCH_SIZE


def _finite_difference_slope(dem: torch.Tensor) -> torch.Tensor:
    """Approximate surface slope magnitude from a normalized DEM patch."""
    dy, dx = torch.gradient(dem.squeeze(0), spacing=1.0)
    return torch.sqrt(dx**2 + dy**2).unsqueeze(0)


def _physics_aware_targets(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive pseudo-labels correlated with terrain physics for synthetic training."""
    lola = inputs["lola"]
    illumination = inputs["illumination"]
    lend = inputs["lend"]
    psr = inputs["psr"]
    mini_rf = inputs["mini_rf"]

    slope = _finite_difference_slope(lola)
    slope_norm = slope / (slope.max().clamp(min=1e-6))

    radar_roughness = mini_rf[:1].abs()
    roughness_norm = radar_roughness / (radar_roughness.max().clamp(min=1e-6))

    hazard = (0.55 * slope_norm + 0.45 * roughness_norm).clamp(0.0, 1.0)
    safety = (1.0 - hazard).clamp(0.0, 1.0)

    illum_score = illumination.clamp(0.0, 1.0)
    resource = (0.65 * lend + 0.35 * (1.0 - psr)).clamp(0.0, 1.0)

    suitability = (
        0.35 * safety + 0.20 * (1.0 - hazard) + 0.20 * illum_score + 0.25 * resource
    ).clamp(0.0, 1.0)

    return {
        "landing_safety_score": safety,
        "hazard_probability": hazard,
        "illumination_score": illum_score,
        "resource_accessibility_score": resource,
        "final_suitability_score": suitability,
        "slope_magnitude": slope,
    }


class LandingSiteDataset(Dataset):
    """Synthetic multi-modal lunar patch dataset for smoke tests and development."""

    def __init__(
        self,
        num_samples: int = 64,
        patch_size: int = DEFAULT_PATCH_SIZE_DATA,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _random_modality(self, channels: int) -> torch.Tensor:
        return torch.randn(channels, self.patch_size, self.patch_size, generator=self._rng).clamp(-2.0, 2.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        del index
        inputs: dict[str, torch.Tensor] = {}
        for name, channels in MODALITY_CHANNELS.items():
            tensor = self._random_modality(channels)
            if name in {"lola", "illumination", "lend", "psr"}:
                tensor = torch.sigmoid(tensor)
            inputs[name] = tensor

        targets = _physics_aware_targets(inputs)
        return {"inputs": inputs, "targets": targets}


def collate_dict_batch(batch: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
    """Stack variable dict samples into batched tensors."""
    input_keys = batch[0]["inputs"].keys()
    target_keys = batch[0]["targets"].keys()

    inputs = {key: torch.stack([sample["inputs"][key] for sample in batch], dim=0) for key in input_keys}
    targets = {key: torch.stack([sample["targets"][key] for sample in batch], dim=0) for key in target_keys}
    return {"inputs": inputs, "targets": targets}


__all__ = [
    "DEFAULT_PATCH_SIZE_DATA",
    "LandingSiteDataset",
    "collate_dict_batch",
]

"""Synthetic and collated datasets for rover hazard and navigation prediction."""

from __future__ import annotations

from typing import Any, Final

import torch
from torch.utils.data import Dataset

from ..models.rover_navigation_net import DEFAULT_PATCH_SIZE
from ..models.terrain_encoder import MODALITY_CHANNELS

DEFAULT_PATCH_SIZE_DATA: Final[int] = DEFAULT_PATCH_SIZE


def _physics_aware_targets(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive pseudo-labels correlated with lunar terrain physics for synthetic training.

    Physics rules:
    - High slope => high hazard.
    - Large crater density => lower traversability.
    - High boulder density => lower rover safety.
    - Good illumination + low hazard => high traversability.
    """
    dem = inputs["dem"]
    slope = inputs["slope"]
    crater = inputs["crater"]
    boulder = inputs["boulder"]
    illumination = inputs["illumination"]

    # Hazards
    crater_hazard = crater
    boulder_hazard = boulder
    slope_hazard = slope

    # Rover safety is lower where crater, boulder, or slope hazards are high
    rover_safety = (1.0 - crater_hazard) * (1.0 - boulder_hazard) * (1.0 - slope_hazard)

    # Traversability is high when safety is high and illumination is good
    traversability = (illumination * 0.4 + 0.6) * rover_safety

    # Safe navigation cost map is higher in hazardous/unsafe areas
    navigation_cost = 1.0 - rover_safety

    return {
        "traversability_score": traversability,
        "crater_hazard_probability": crater_hazard,
        "boulder_hazard_probability": boulder_hazard,
        "slope_hazard_score": slope_hazard,
        "navigation_cost_map": navigation_cost,
        "rover_safety_score": rover_safety,
    }


class RoverDataset(Dataset):
    """Synthetic multi-modal lunar patch dataset for rover hazard navigation smoke tests."""

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

    def _random_smooth_map(self) -> torch.Tensor:
        """Create a smooth 2D spatial map by generating random noise and smoothing it."""
        raw = torch.randn(1, self.patch_size, self.patch_size, generator=self._rng)
        kernel_size = 5
        padding = kernel_size // 2
        blur_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)
        smooth = torch.nn.functional.conv2d(raw.unsqueeze(0), blur_kernel, padding=padding).squeeze(0)
        return smooth

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Generate correlated features
        # 1. High-resolution DEM
        dem = torch.sigmoid(self._random_smooth_map())
        lola = dem  # LOLA topography correlates with DEM

        # 2. Slopes (can be higher-frequency noise or gradient of DEM)
        slope = torch.sigmoid(self._random_smooth_map() * 1.5 + 0.2)

        # 3. Crater maps (depression areas, lower elevation)
        crater = torch.sigmoid(-6.0 * (dem - 0.4))

        # 4. Boulder maps (high frequency scatter, high roughness)
        boulder = torch.sigmoid(self._random_smooth_map() * 2.0 - 0.5)

        # 5. Solar illumination (higher on elevated terrain, blocked in crater floors/shadows)
        illumination = (dem * 0.7 + 0.3) * (1.0 - crater)

        # 6. Mini-RF radar (roughness correlates with boulders and slope changes)
        radar_base = 0.5 * boulder + 0.3 * slope + 0.2 * torch.randn(1, self.patch_size, self.patch_size, generator=self._rng)
        mini_rf = torch.cat([
            torch.sigmoid(radar_base),
            torch.sigmoid(radar_base * 0.8 + 0.1),
            torch.sigmoid(radar_base * 1.2 - 0.1)
        ], dim=0)

        inputs = {
            "lola": lola,
            "mini_rf": mini_rf,
            "dem": dem,
            "slope": slope,
            "crater": crater,
            "boulder": boulder,
            "illumination": illumination,
        }

        # Clamp all inputs to [0, 1] for stability
        inputs = {k: v.clamp(0.0, 1.0) for k, v in inputs.items()}

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
    "RoverDataset",
    "collate_dict_batch",
]

"""Synthetic and collated datasets for radiation risk prediction."""

from __future__ import annotations

from typing import Any, Final

import torch
from torch.utils.data import Dataset

from ..models.radiation_encoder import MODALITY_CHANNELS
from ..models.radiation_net import DEFAULT_PATCH_SIZE

DEFAULT_PATCH_SIZE_DATA: Final[int] = DEFAULT_PATCH_SIZE


def _physics_aware_targets(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive pseudo-labels correlated with terrain and radiation physics for synthetic training.

    Physics rules:
    - Regions with thicker regolith and deep PSRs should generally have lower radiation exposure.
    - High elevation and exposed terrain should have higher radiation exposure.
    """
    elevation = inputs["elevation"]
    psr = inputs["psr"]
    flux = inputs["flux"]
    regolith = inputs["regolith"]

    # Dose rate decreases with regolith thickness and inside PSRs (crater shadow blocking),
    # and increases with cosmic/solar particle flux.
    radiation_dose = (0.5 + 1.0 * flux - 0.4 * regolith) * (1.0 - 0.2 * psr)
    radiation_dose = radiation_dose.clamp(min=0.0)

    # Risk score maps dose rate to [0, 1]
    radiation_risk = torch.sigmoid((radiation_dose - 0.6) / 0.2)

    # Shielding effectiveness is determined by regolith thickness
    shielding_eff = regolith

    # Habitat safety depends on shielding effectiveness and the radiation risk
    habitat_safety = shielding_eff * (1.0 - radiation_risk)

    # Hazard map correlates directly with radiation risk
    hazard_map = radiation_risk

    return {
        "radiation_dose_rate": radiation_dose,
        "radiation_risk_score": radiation_risk,
        "shielding_effectiveness_score": shielding_eff,
        "habitat_safety_score": habitat_safety,
        "final_radiation_hazard_map": hazard_map,
    }


class RadiationDataset(Dataset):
    """Synthetic multi-modal lunar patch dataset for radiation risk prediction smoke tests."""

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
        # Start with small random noise
        raw = torch.randn(1, self.patch_size, self.patch_size, generator=self._rng)
        # Smooth with a simple box/blur filter to make it look like continuous terrain
        kernel_size = 5
        padding = kernel_size // 2
        blur_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)
        smooth = torch.nn.functional.conv2d(raw.unsqueeze(0), blur_kernel, padding=padding).squeeze(0)
        return smooth

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Generate correlated features
        # 1. Base topography
        elevation = torch.sigmoid(self._random_smooth_map())
        lola = elevation  # topography

        # 2. PSR is high where elevation is low (craters)
        psr = torch.sigmoid(-8.0 * (elevation - 0.35))

        # 3. Solar illumination is high at higher elevations, and 0 in PSR
        illumination = (elevation * 0.8 + 0.2) * (1.0 - psr)

        # 4. Temperature (Diviner) correlates with solar illumination and PSR
        diviner = illumination * 0.7 + 0.15 * (1.0 - psr)

        # 5. Cosmic ray/solar particle flux: higher elevation has more sky view, PSR crater walls block it slightly
        flux = torch.sigmoid(0.5 * elevation + 0.5 * (1.0 - psr) + 0.2 * torch.randn(1, self.patch_size, self.patch_size, generator=self._rng))

        # 6. Regolith thickness is a smooth random map
        regolith = torch.sigmoid(self._random_smooth_map())

        inputs = {
            "lola": lola,
            "elevation": elevation,
            "illumination": illumination,
            "psr": psr,
            "diviner": diviner,
            "flux": flux,
            "regolith": regolith,
        }

        # Clamp all inputs to [0, 1] for sanity
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
    "RadiationDataset",
    "collate_dict_batch",
]

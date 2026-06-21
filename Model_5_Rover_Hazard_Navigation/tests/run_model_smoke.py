"""Model smoke test for Model 5 Rover Hazard & Navigation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_5_Rover_Hazard_Navigation.src.models.heads import OUTPUT_KEYS
from Model_5_Rover_Hazard_Navigation.src.models.rover_navigation_net import RoverNavigationNet
from Model_5_Rover_Hazard_Navigation.src.models.terrain_encoder import MODALITY_CHANNELS


def main() -> None:
    model = RoverNavigationNet()
    model.eval()

    patch = 64
    inputs = {
        name: torch.randn(2, channels, patch, patch)
        for name, channels in MODALITY_CHANNELS.items()
    }

    with torch.no_grad():
        outputs = model(inputs)

    for key in OUTPUT_KEYS:
        assert key in outputs, f"Missing output: {key}"
        tensor = outputs[key]
        assert tensor.shape == (2, 1, patch, patch), f"Bad shape for {key}: {tensor.shape}"
        assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, f"{key} outside [0, 1]"

    print("model smoke tests passed")


if __name__ == "__main__":
    main()

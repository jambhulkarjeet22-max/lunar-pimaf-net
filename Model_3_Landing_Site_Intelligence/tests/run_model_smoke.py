"""Model smoke test for Model 3 Landing Site Intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_3_Landing_Site_Intelligence.src.models.heads import OUTPUT_KEYS
from Model_3_Landing_Site_Intelligence.src.models.landing_site_net import LandingSiteNet
from Model_3_Landing_Site_Intelligence.src.models.terrain_encoder import MODALITY_CHANNELS


def main() -> None:
    model = LandingSiteNet()
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

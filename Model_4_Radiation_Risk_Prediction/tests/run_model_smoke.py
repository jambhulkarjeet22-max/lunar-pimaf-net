"""Model smoke test for Model 4 Radiation Risk Prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_4_Radiation_Risk_Prediction.src.models.heads import OUTPUT_KEYS
from Model_4_Radiation_Risk_Prediction.src.models.radiation_encoder import MODALITY_CHANNELS
from Model_4_Radiation_Risk_Prediction.src.models.radiation_net import RadiationNet


def main() -> None:
    model = RadiationNet()
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
        if key != "radiation_dose_rate":
            assert tensor.min() >= 0.0 and tensor.max() <= 1.0, f"{key} outside [0, 1]"
        else:
            assert tensor.min() >= 0.0, f"Dose rate must be non-negative"

    print("model smoke tests passed")


if __name__ == "__main__":
    main()

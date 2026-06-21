"""Lightweight smoke runner for LUNAR-PIMAF-Net (no pytest required)."""

from __future__ import annotations

import torch

from src.models.lunar_pimaf_net import EXPECTED_INPUT_CHANNELS, LunarPIMAFNet, LunarPIMAFOutput

EXPECTED_OUTPUT_KEYS: tuple[str, ...] = (
    "segmentation_logits",
    "surface_ice_probability",
    "subsurface_ice_probability",
    "dirichlet_alpha",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "total_uncertainty",
    "confidence",
    "physics_residuals",
    "latent_physics",
    "decoder_features",
    "fused_pyramid",
)


def main() -> None:
    print("model imports ok")

    model = LunarPIMAFNet()
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, EXPECTED_INPUT_CHANNELS, 128, 128)
    with torch.no_grad():
        outputs: LunarPIMAFOutput = model(x)

    missing = [key for key in EXPECTED_OUTPUT_KEYS if key not in outputs]
    if missing:
        raise AssertionError(f"Missing output keys: {', '.join(missing)}")

    assert outputs["segmentation_logits"].shape == (batch_size, 3, 128, 128)
    assert outputs["surface_ice_probability"].shape == (batch_size, 1, 128, 128)
    assert outputs["subsurface_ice_probability"].shape == (batch_size, 1, 128, 128)
    assert outputs["confidence"].shape == (batch_size, 1, 128, 128)
    assert outputs["decoder_features"].shape[0] == batch_size

    print("model smoke tests passed")


if __name__ == "__main__":
    main()

"""Inference smoke test for Model 4 Radiation Risk Prediction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.optim as optim

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_4_Radiation_Risk_Prediction.src.models.radiation_net import RadiationNet
from Model_4_Radiation_Risk_Prediction.src.training.checkpoint import save_checkpoint
from Model_4_Radiation_Risk_Prediction.src.training.config import PredictConfig
from Model_4_Radiation_Risk_Prediction.src.training.inference import run_prediction


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = tmp_path / "best.pt"
        model = RadiationNet()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        save_checkpoint(model, optimizer, 1, ckpt, metrics={"radiation_mse": 0.1})

        out_dir = tmp_path / "predictions"
        config = PredictConfig(
            checkpoint_path=str(ckpt),
            output_dir=str(out_dir),
            batch_size=2,
            patch_size=32,
            num_samples=4,
            device="cpu",
        )
        summary = run_prediction(config)
        assert summary["num_samples"] == 4
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "radiation_dose_rate.npy").exists()
        assert (out_dir / "radiation_risk_score.npy").exists()
        assert (out_dir / "shielding_effectiveness_score.npy").exists()
        assert (out_dir / "habitat_safety_score.npy").exists()
        assert (out_dir / "final_radiation_hazard_map.npy").exists()

    print("predict smoke tests passed")


if __name__ == "__main__":
    main()

"""Training smoke test for Model 4 Radiation Risk Prediction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_4_Radiation_Risk_Prediction.src.models.radiation_net import RadiationNet
from Model_4_Radiation_Risk_Prediction.src.training.checkpoint import load_checkpoint, save_checkpoint
from Model_4_Radiation_Risk_Prediction.src.training.config import TrainingConfig
from Model_4_Radiation_Risk_Prediction.src.training.trainer import Trainer


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            num_samples=4,
            patch_size=32,
            val_fraction=0.5,
            device="cpu",
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            early_stopping_patience=1,
        )
        trainer = Trainer(config)
        metrics = trainer.train_one_batch()
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

        ckpt_path = Path(config.checkpoint_dir) / "smoke.pt"
        save_checkpoint(trainer.model, trainer.optimizer, 1, ckpt_path, metrics=metrics)

        restored = RadiationNet()
        load_checkpoint(ckpt_path, restored)
        restored.eval()
        sample = torch.randn(1, 1, config.patch_size, config.patch_size)
        with torch.no_grad():
            out = restored({
                "lola": sample,
                "elevation": sample,
                "illumination": sample,
                "psr": sample,
                "diviner": sample,
                "flux": sample,
                "regolith": sample,
            })
        assert "radiation_dose_rate" in out

        trainer.train()
        assert (Path(config.checkpoint_dir) / "best.pt").exists()

    print("train smoke tests passed")


if __name__ == "__main__":
    main()

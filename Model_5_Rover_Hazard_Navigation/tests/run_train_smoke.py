"""Training smoke test for Model 5 Rover Hazard & Navigation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_5_Rover_Hazard_Navigation.src.models.rover_navigation_net import RoverNavigationNet
from Model_5_Rover_Hazard_Navigation.src.training.checkpoint import load_checkpoint, save_checkpoint
from Model_5_Rover_Hazard_Navigation.src.training.config import TrainingConfig
from Model_5_Rover_Hazard_Navigation.src.training.trainer import Trainer


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

        restored = RoverNavigationNet()
        load_checkpoint(ckpt_path, restored)
        restored.eval()
        sample = torch.randn(1, 1, config.patch_size, config.patch_size)
        mini_rf_sample = torch.randn(1, 3, config.patch_size, config.patch_size)
        with torch.no_grad():
            out = restored({
                "lola": sample,
                "mini_rf": mini_rf_sample,
                "dem": sample,
                "slope": sample,
                "crater": sample,
                "boulder": sample,
                "illumination": sample,
            })
        assert "traversability_score" in out

        trainer.train()
        assert (Path(config.checkpoint_dir) / "best.pt").exists()

    print("train smoke tests passed")


if __name__ == "__main__":
    main()

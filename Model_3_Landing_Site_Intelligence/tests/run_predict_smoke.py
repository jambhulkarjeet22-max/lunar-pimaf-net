"""Inference smoke test for Model 3 Landing Site Intelligence."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_3_Landing_Site_Intelligence.src.models.landing_site_net import LandingSiteNet
from Model_3_Landing_Site_Intelligence.src.training.checkpoint import save_checkpoint
from Model_3_Landing_Site_Intelligence.src.training.config import PredictConfig
from Model_3_Landing_Site_Intelligence.src.training.inference import run_prediction


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = tmp_path / "best.pt"
        model = LandingSiteNet()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(model, optimizer, 1, ckpt, metrics={"safety_mse": 0.1})

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
        assert (out_dir / "landing_safety_score.npy").exists()

    print("predict smoke tests passed")


if __name__ == "__main__":
    main()

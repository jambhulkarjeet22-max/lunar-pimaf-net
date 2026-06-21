"""Lightweight smoke tests for inference and export (no full dataset run)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.models.lunar_pimaf_net import LunarPIMAFNet
from src.training.checkpoint import save_checkpoint
from src.training.config import PredictConfig, TrainingConfig
from src.training.dataset import LunarPatchDataset, collate_batch
from src.training.inference import (
    build_model_from_checkpoint,
    export_probability_geotiff,
    export_probability_png,
    predict_batch,
)
from src.training.trainer import resolve_device


def main() -> None:
    print("predict imports ok")

    device = resolve_device("cpu")
    model = LunarPIMAFNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "checkpoint.pt"
        train_config = TrainingConfig(synthetic_samples=2, device="cpu")
        save_checkpoint(
            checkpoint,
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            best_val_loss=1.0,
            config=train_config.to_dict(),
            is_best=True,
        )

        loaded_model, payload = build_model_from_checkpoint(checkpoint, device=device)
        assert payload["epoch"] == 0

        dataset = LunarPatchDataset(synthetic_samples=2, seed=1)
        batch = collate_batch([dataset[0]])
        outputs = predict_batch(loaded_model, batch, device=device)

        for key in (
            "segmentation_logits",
            "surface_ice_probability",
            "subsurface_ice_probability",
            "total_uncertainty",
            "confidence",
        ):
            assert key in outputs

        prob = outputs["subsurface_ice_probability"][0, 0].numpy()
        geotiff_path = export_probability_geotiff(prob, tmp_path / "subsurface.tif", pole="north")
        png_path = export_probability_png(prob, tmp_path / "subsurface.png")
        assert geotiff_path.is_file()
        assert png_path.is_file()

        config = PredictConfig(
            checkpoint=tmp_path / "best.pt",
            output_dir=tmp_path / "predictions",
            synthetic_samples=1,
            batch_size=1,
            device="cpu",
        )
        from src.training.inference import run_prediction

        exported = run_prediction(config)
        assert exported

    print("predict smoke tests passed")


if __name__ == "__main__":
    main()

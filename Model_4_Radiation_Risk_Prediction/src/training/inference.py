"""Inference pipeline for radiation risk prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import RadiationDataset, collate_dict_batch
from ..models.heads import OUTPUT_KEYS
from ..models.radiation_net import RadiationNet
from .checkpoint import load_checkpoint
from .config import PredictConfig


class InferencePipeline:
    """Load checkpoints and produce radiation dose, risk, and safety score maps."""

    def __init__(self, config: PredictConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = RadiationNet().to(self.device)
        self._loaded_epoch = 0

        if config.checkpoint.exists():
            checkpoint = load_checkpoint(config.checkpoint, self.model, map_location=config.device)
            self._loaded_epoch = int(checkpoint.get("epoch", 0))
        else:
            raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {key: value.to(self.device) for key, value in inputs.items()}
        return self.model(moved)

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        dataset = RadiationDataset(
            num_samples=self.config.num_samples,
            patch_size=self.config.patch_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_dict_batch,
        )

        all_predictions: dict[str, list[np.ndarray]] = {key: [] for key in OUTPUT_KEYS}
        for batch in loader:
            outputs = self.predict_batch(batch["inputs"])
            for key in OUTPUT_KEYS:
                all_predictions[key].append(outputs[key].cpu().numpy())

        merged = {key: np.concatenate(chunks, axis=0) for key, chunks in all_predictions.items()}
        self._export_predictions(merged)
        return {
            "epoch": self._loaded_epoch,
            "num_samples": merged[OUTPUT_KEYS[0]].shape[0],
            "output_shapes": {key: list(arr.shape) for key, arr in merged.items()},
        }

    def _export_predictions(self, predictions: dict[str, np.ndarray]) -> None:
        output_dir = self.config.output
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "checkpoint": str(self.config.checkpoint),
            "epoch": self._loaded_epoch,
            "keys": list(predictions.keys()),
            "shapes": {key: list(arr.shape) for key, arr in predictions.items()},
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        for key, array in predictions.items():
            np.save(output_dir / f"{key}.npy", array)


def run_prediction(config: PredictConfig) -> dict[str, Any]:
    return InferencePipeline(config).run()


__all__ = ["InferencePipeline", "run_prediction"]

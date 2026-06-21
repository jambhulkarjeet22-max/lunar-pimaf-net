"""Lightweight smoke tests for the training pipeline (no full training run)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.models.lunar_pimaf_net import LunarPIMAFNet
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.training.dataset import LunarPatchDataset, collate_batch
from src.training.losses import ProductionTrainingLoss
from src.training.metrics import MetricAccumulator, subsurface_binary_predictions, subsurface_binary_targets
from src.training.trainer import Trainer, resolve_device


def main() -> None:
    print("training imports ok")

    device = resolve_device("cpu")
    config = TrainingConfig(
        synthetic_samples=8,
        batch_size=2,
        val_fraction=0.25,
        output_dir=Path("saved_models/_smoke"),
        log_dir=Path("logs/_smoke"),
        use_amp=False,
        num_workers=0,
        device="cpu",
    )

    trainer = Trainer(config)
    batch = next(iter(trainer.train_loader))
    batch = trainer._move_batch(batch)

    trainer.model.train()
    outputs = trainer.model(batch["inputs"])
    loss_out = trainer.criterion(outputs, batch)
    assert torch.isfinite(loss_out.total)
    assert loss_out.bce.ndim == 0
    assert loss_out.dice.ndim == 0
    assert loss_out.physics.ndim == 0
    assert loss_out.uncertainty.ndim == 0

    pred = subsurface_binary_predictions(outputs["segmentation_logits"])
    target = subsurface_binary_targets(batch["y_soft"])
    metrics = MetricAccumulator()
    metrics.update(pred, target)
    result = metrics.compute()
    for key in ("iou", "dice", "precision", "recall", "f1"):
        assert key in result

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "checkpoint.pt"
        save_checkpoint(
            ckpt_path,
            epoch=0,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            best_val_loss=float(loss_out.total.detach()),
            config=config.to_dict(),
            is_best=True,
        )
        assert ckpt_path.is_file()
        assert (Path(tmp) / "best.pt").is_file()

        restored = LunarPIMAFNet(dropout=config.dropout, fpn_channels=config.fpn_channels).to(device)
        payload = load_checkpoint(ckpt_path, model=restored, map_location=device)
        assert payload["epoch"] == 0
        restored.eval()
        with torch.no_grad():
            restored_out = restored(batch["inputs"])
        assert restored_out["subsurface_ice_probability"].shape == outputs["subsurface_ice_probability"].shape

    dataset = LunarPatchDataset(synthetic_samples=4, seed=0)
    sample = dataset[0]
    assert sample["inputs"].shape[0] == 59
    stacked = collate_batch([sample, sample])
    assert stacked["inputs"].shape[0] == 2

    print("training smoke tests passed")


if __name__ == "__main__":
    main()

"""Training loop with AMP, checkpointing, and TensorBoard logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.models.lunar_pimaf_net import LunarPIMAFNet
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.training.dataset import LunarPatchDataset, collate_batch
from src.training.losses import ProductionTrainingLoss
from src.training.metrics import (
    MetricAccumulator,
    subsurface_binary_predictions,
    subsurface_binary_targets,
)

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    """End-to-end trainer for LUNAR-PIMAF-Net."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        dataset = LunarPatchDataset(
            data_path=config.data_path if config.synthetic_samples <= 0 else None,
            synthetic_samples=config.synthetic_samples,
            seed=config.seed,
        )
        val_size = max(1, int(len(dataset) * config.val_fraction))
        train_size = len(dataset) - val_size
        if train_size < 1:
            raise ValueError("Training split is empty; increase synthetic_samples or dataset size.")

        generator = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_batch,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_batch,
        )

        self.model = LunarPIMAFNet(
            dropout=config.dropout,
            fpn_channels=config.fpn_channels,
        ).to(self.device)

        self.criterion = ProductionTrainingLoss(
            bce_weight=config.loss_bce,
            dice_weight=config.loss_dice,
            physics_weight=config.loss_physics,
            uncertainty_weight=config.loss_uncertainty,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and self.device.type == "cuda")

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.writer: SummaryWriter | None = None

        if config.resume is not None:
            payload = load_checkpoint(
                config.resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=self.device,
            )
            self.start_epoch = int(payload.get("epoch", 0)) + 1
            self.best_val_loss = float(payload.get("best_val_loss", float("inf")))

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running = {"loss_total": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "loss_physics": 0.0, "loss_uncertainty": 0.0}
        metric_acc = MetricAccumulator()
        num_batches = 0
        kl_annealing = min(1.0, epoch / max(1, self.config.epochs // 2))

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.config.use_amp and self.device.type == "cuda",
            ):
                outputs = self.model(batch["inputs"])
                loss_out = self.criterion(outputs, batch, kl_annealing=kl_annealing)
                loss = loss_out.total

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running["loss_total"] += float(loss.detach())
            running["loss_bce"] += float(loss_out.bce.detach())
            running["loss_dice"] += float(loss_out.dice.detach())
            running["loss_physics"] += float(loss_out.physics.detach())
            running["loss_uncertainty"] += float(loss_out.uncertainty.detach())

            pred = subsurface_binary_predictions(outputs["segmentation_logits"])
            target = subsurface_binary_targets(batch["y_soft"])
            metric_acc.update(pred, target)
            num_batches += 1

            if self.writer and batch_idx % self.config.log_every == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/batch_loss", float(loss.detach()), global_step)

        denom = max(num_batches, 1)
        stats = {key: value / denom for key, value in running.items()}
        stats.update({f"train_{key}": value for key, value in metric_acc.compute().items()})
        return stats

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        running = {"loss_total": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "loss_physics": 0.0, "loss_uncertainty": 0.0}
        metric_acc = MetricAccumulator()
        num_batches = 0
        kl_annealing = min(1.0, epoch / max(1, self.config.epochs // 2))

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast(
                "cuda",
                enabled=self.config.use_amp and self.device.type == "cuda",
            ):
                outputs = self.model(batch["inputs"])
                loss_out = self.criterion(outputs, batch, kl_annealing=kl_annealing)

            running["loss_total"] += float(loss_out.total.detach())
            running["loss_bce"] += float(loss_out.bce.detach())
            running["loss_dice"] += float(loss_out.dice.detach())
            running["loss_physics"] += float(loss_out.physics.detach())
            running["loss_uncertainty"] += float(loss_out.uncertainty.detach())

            pred = subsurface_binary_predictions(outputs["segmentation_logits"])
            target = subsurface_binary_targets(batch["y_soft"])
            metric_acc.update(pred, target)
            num_batches += 1

        denom = max(num_batches, 1)
        stats = {key: value / denom for key, value in running.items()}
        stats.update({f"val_{key}": value for key, value in metric_acc.compute().items()})
        return stats

    def fit(self) -> dict[str, float]:
        """Run the full training schedule with early stopping and checkpointing."""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        history: dict[str, float] = {}

        config_path = self.output_dir / "training_config.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_start = time.time()
                train_stats = self.train_one_epoch(epoch)
                val_stats = self.validate(epoch)
                val_loss = val_stats["loss_total"]
                self.scheduler.step(val_loss)

                for key, value in train_stats.items():
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
                for key, value in val_stats.items():
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
                self.writer.add_scalar("epoch/lr", self.optimizer.param_groups[0]["lr"], epoch)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if epoch % self.config.checkpoint_every == 0 or is_best:
                    save_checkpoint(
                        self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        best_val_loss=self.best_val_loss,
                        config=self.config.to_dict(),
                        is_best=is_best,
                    )

                elapsed = time.time() - epoch_start
                logger.info(
                    "Epoch %d/%d | val_loss=%.4f | val_iou=%.4f | val_dice=%.4f | %.1fs",
                    epoch + 1,
                    self.config.epochs,
                    val_loss,
                    val_stats.get("val_iou", 0.0),
                    val_stats.get("val_dice", 0.0),
                    elapsed,
                )

                history = {**train_stats, **val_stats, "best_val_loss": self.best_val_loss}

                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered after %d epochs without improvement.", epoch + 1)
                    break
        finally:
            if self.writer is not None:
                self.writer.close()

        return history


def run_training(config: TrainingConfig) -> dict[str, float]:
    """Convenience wrapper used by ``train.py`` and smoke tests."""
    trainer = Trainer(config)
    return trainer.fit()


__all__ = ["Trainer", "resolve_device", "run_training"]

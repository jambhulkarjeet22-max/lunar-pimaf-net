"""Training loop for lunar radiation risk prediction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ..data.dataset import RadiationDataset, collate_dict_batch
from ..models.radiation_net import RadiationNet
from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .losses import RadiationLoss
from .metrics import MetricsCalculator, format_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """Production-style trainer with validation, physics-aware loss, checkpointing, and early stopping."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = torch.device(config.device)
        self.model = RadiationNet().to(self.device)
        self.criterion = RadiationLoss(physics_weight=config.physics_weight)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.train_metrics = MetricsCalculator(hazard_threshold=config.hazard_threshold)
        self.val_metrics = MetricsCalculator(hazard_threshold=config.hazard_threshold)
        amp_enabled = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        dataset = RadiationDataset(
            num_samples=self.config.num_samples,
            patch_size=self.config.patch_size,
            seed=self.config.seed,
        )
        if self.config.val_fraction <= 0 or len(dataset) < 2:
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_dict_batch,
            )
            return train_loader, None

        val_size = max(1, int(len(dataset) * self.config.val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_dict_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_dict_batch,
        )
        return train_loader, val_loader

    def _move_batch(self, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        inputs = {key: value.to(self.device) for key, value in batch["inputs"].items()}
        targets = {key: value.to(self.device) for key, value in batch["targets"].items()}
        # Inject inputs into targets for physics loss access
        targets["inputs"] = inputs
        return inputs, targets

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        for batch in loader:
            inputs, targets = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.train_metrics.update(outputs, targets)

        avg_loss = total_loss / max(len(loader), 1)
        metrics = self.train_metrics.compute()
        print(f"Epoch {epoch} train loss={avg_loss:.4f} {format_metrics(metrics)}")
        return avg_loss, metrics

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        for batch in loader:
            inputs, targets = self._move_batch(batch)
            outputs = self.model(inputs)
            loss_dict = self.criterion(outputs, targets)
            total_loss += loss_dict["total_loss"].item()
            self.val_metrics.update(outputs, targets)

        avg_loss = total_loss / max(len(loader), 1)
        metrics = self.val_metrics.compute()
        print(f"Epoch {epoch} val   loss={avg_loss:.4f} {format_metrics(metrics)}")
        return avg_loss, metrics

    def train(self) -> dict[str, float]:
        train_loader, val_loader = self._build_loaders()
        last_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            last_metrics = train_metrics

            if val_loader is None:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                    metrics=train_metrics,
                    config=self.config.to_dict(),
                )
                continue

            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            last_metrics = val_metrics
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                metrics=val_metrics,
                config=self.config.to_dict(),
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.checkpoint_dir / "best.pt",
                    metrics=val_metrics,
                    config=self.config.to_dict(),
                    is_best=True,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        return last_metrics

    def train_one_batch(self) -> dict[str, float]:
        """Lightweight single-batch step for smoke testing."""
        dataset = RadiationDataset(num_samples=self.config.batch_size, patch_size=self.config.patch_size)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_dict_batch)
        batch = next(iter(loader))
        inputs, targets = self._move_batch(batch)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss_dict = self.criterion(outputs, targets)
        loss_dict["total_loss"].backward()
        self.optimizer.step()

        self.train_metrics.reset()
        self.train_metrics.update(outputs, targets)
        return {
            "loss": loss_dict["total_loss"].item(),
            **self.train_metrics.compute(),
        }


def run_training(config: TrainingConfig) -> dict[str, float]:
    return Trainer(config).train()


__all__ = ["Trainer", "run_training", "set_seed"]

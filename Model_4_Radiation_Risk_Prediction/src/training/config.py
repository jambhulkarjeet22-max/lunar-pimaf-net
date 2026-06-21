"""Training and prediction configuration for radiation risk prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration settings for radiation model training."""

    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_samples: int = 64
    val_fraction: float = 0.25
    patch_size: int = 64
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    early_stopping_patience: int = 3
    physics_weight: float = 0.1
    hazard_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictConfig:
    """Configuration settings for inference."""

    checkpoint_path: str = "checkpoints/best.pt"
    output_dir: str = "predictions"
    batch_size: int = 2
    patch_size: int = 64
    device: str = "cpu"
    num_samples: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def checkpoint(self) -> Path:
        return Path(self.checkpoint_path)

    @property
    def output(self) -> Path:
        return Path(self.output_dir)


__all__ = ["PredictConfig", "TrainingConfig"]

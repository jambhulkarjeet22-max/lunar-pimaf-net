"""Configuration dataclasses for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime options for ``train.py``."""

    data_path: Path = Path("data/processed/patches.zarr")
    output_dir: Path = Path("saved_models/experiment")
    log_dir: Path = Path("logs/tensorboard")
    resume: Path | None = None

    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 0
    val_fraction: float = 0.2
    seed: int = 42

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10

    synthetic_samples: int = 0
    dropout: float = 0.1
    fpn_channels: int = 256

    loss_bce: float = 1.0
    loss_dice: float = 1.0
    loss_physics: float = 0.3
    loss_uncertainty: float = 0.5

    checkpoint_every: int = 1
    log_every: int = 10

    device: str = "auto"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        payload["log_dir"] = str(self.log_dir)
        payload["resume"] = str(self.resume) if self.resume else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> TrainingConfig:
        return cls(
            data_path=Path(payload.get("data_path", "data/processed/patches.zarr")),
            output_dir=Path(payload.get("output_dir", "saved_models/experiment")),
            log_dir=Path(payload.get("log_dir", "logs/tensorboard")),
            resume=Path(payload["resume"]) if payload.get("resume") else None,
            epochs=int(payload.get("epochs", 50)),
            batch_size=int(payload.get("batch_size", 4)),
            num_workers=int(payload.get("num_workers", 0)),
            val_fraction=float(payload.get("val_fraction", 0.2)),
            seed=int(payload.get("seed", 42)),
            learning_rate=float(payload.get("learning_rate", 1e-4)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            grad_clip_norm=float(payload.get("grad_clip_norm", 1.0)),
            use_amp=bool(payload.get("use_amp", True)),
            scheduler_patience=int(payload.get("scheduler_patience", 5)),
            scheduler_factor=float(payload.get("scheduler_factor", 0.5)),
            early_stopping_patience=int(payload.get("early_stopping_patience", 10)),
            synthetic_samples=int(payload.get("synthetic_samples", 0)),
            dropout=float(payload.get("dropout", 0.1)),
            fpn_channels=int(payload.get("fpn_channels", 256)),
            loss_bce=float(payload.get("loss_bce", 1.0)),
            loss_dice=float(payload.get("loss_dice", 1.0)),
            loss_physics=float(payload.get("loss_physics", 0.3)),
            loss_uncertainty=float(payload.get("loss_uncertainty", 0.5)),
            checkpoint_every=int(payload.get("checkpoint_every", 1)),
            log_every=int(payload.get("log_every", 10)),
            device=str(payload.get("device", "auto")),
        )


@dataclass
class PredictConfig:
    """Runtime options for ``predict.py``."""

    checkpoint: Path = Path("saved_models/experiment/best.pt")
    data_path: Path | None = None
    output_dir: Path = Path("outputs/predictions")
    batch_size: int = 1
    num_workers: int = 0
    synthetic_samples: int = 1
    pole: str = "north"
    device: str = "auto"
    export_geotiff: bool = True
    export_png: bool = True
    fpn_channels: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint"] = str(self.checkpoint)
        payload["data_path"] = str(self.data_path) if self.data_path else None
        payload["output_dir"] = str(self.output_dir)
        return payload

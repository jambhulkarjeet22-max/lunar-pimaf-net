"""Checkpoint persistence for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path | str,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    best_val_loss: float,
    config: dict[str, Any],
    is_best: bool = False,
) -> Path:
    """Persist model, optimizer, scheduler, and training metadata."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(payload, resolved)

    if is_best:
        best_path = resolved.parent / "best.pt"
        torch.save(payload, best_path)

    return resolved


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer/scheduler/scaler states."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    payload: dict[str, Any] = torch.load(resolved, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    return payload


__all__ = ["load_checkpoint", "save_checkpoint"]

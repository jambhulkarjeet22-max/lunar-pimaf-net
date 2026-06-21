"""
LUNAR OS — Training Entry Point
Physics-Informed Ice Detection Engine

Production PyTorch training pipeline for LUNAR-PIMAF-Net.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training.config import TrainingConfig
from src.training.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


_DEFAULT_TRAIN = TrainingConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LUNAR-PIMAF-Net")
    parser.add_argument("--data-path", type=Path, default=_DEFAULT_TRAIN.data_path)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_TRAIN.output_dir)
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_TRAIN.log_dir)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_TRAIN.batch_size)
    parser.add_argument("--num-workers", type=int, default=_DEFAULT_TRAIN.num_workers)
    parser.add_argument("--val-fraction", type=float, default=_DEFAULT_TRAIN.val_fraction)
    parser.add_argument("--seed", type=int, default=_DEFAULT_TRAIN.seed)
    parser.add_argument("--learning-rate", type=float, default=_DEFAULT_TRAIN.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=_DEFAULT_TRAIN.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=_DEFAULT_TRAIN.grad_clip_norm)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--scheduler-patience", type=int, default=_DEFAULT_TRAIN.scheduler_patience)
    parser.add_argument("--scheduler-factor", type=float, default=_DEFAULT_TRAIN.scheduler_factor)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=_DEFAULT_TRAIN.early_stopping_patience,
    )
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=_DEFAULT_TRAIN.dropout)
    parser.add_argument("--fpn-channels", type=int, default=_DEFAULT_TRAIN.fpn_channels)
    parser.add_argument("--loss-bce", type=float, default=_DEFAULT_TRAIN.loss_bce)
    parser.add_argument("--loss-dice", type=float, default=_DEFAULT_TRAIN.loss_dice)
    parser.add_argument("--loss-physics", type=float, default=_DEFAULT_TRAIN.loss_physics)
    parser.add_argument("--loss-uncertainty", type=float, default=_DEFAULT_TRAIN.loss_uncertainty)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=not args.no_amp,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        early_stopping_patience=args.early_stopping_patience,
        synthetic_samples=args.synthetic_samples,
        dropout=args.dropout,
        fpn_channels=args.fpn_channels,
        loss_bce=args.loss_bce,
        loss_dice=args.loss_dice,
        loss_physics=args.loss_physics,
        loss_uncertainty=args.loss_uncertainty,
        device=args.device,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    logger.info("Starting training with output directory: %s", config.output_dir)
    history = run_training(config)
    logger.info("Training finished. Best validation loss: %.6f", history.get("best_val_loss", float("nan")))


if __name__ == "__main__":
    main()

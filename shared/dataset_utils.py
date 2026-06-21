"""Dataset and path helpers for multi-model training pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def resolve_repo_paths(*extra_paths: Path | str) -> list[str]:
    """Return PYTHONPATH entries for repository root, shared, and model roots."""
    repo_root = Path(__file__).resolve().parents[1]
    paths = [str(repo_root), str(repo_root / "shared")]
    for model_dir in sorted(repo_root.glob("Model_*")):
        paths.append(str(model_dir))
    for entry in extra_paths:
        paths.append(str(Path(entry).resolve()))
    unique: list[str] = []
    for path in paths:
        if path not in unique and path not in sys.path:
            unique.append(path)
    return unique


def ensure_import_paths(*extra_paths: Path | str) -> None:
    """Insert repository and model paths at the front of ``sys.path``."""
    for path in reversed(resolve_repo_paths(*extra_paths)):
        if path not in sys.path:
            sys.path.insert(0, path)


def collate_tensor_dict(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack single-sample dictionaries into a batch."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}

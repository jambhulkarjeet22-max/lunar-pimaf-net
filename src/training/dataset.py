"""PyTorch datasets for LUNAR-PIMAF-Net patch training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import NUM_FUSION_CHANNELS, PATCH_SIZE
from src.models.lunar_pimaf_net import EXPECTED_INPUT_CHANNELS
from src.models.prediction_heads import NUM_SEGMENTATION_CLASSES

LOLA_PSR_CHANNEL: int = 7


@dataclass(frozen=True)
class BatchLabels:
    """Supervision tensors bundled with each patch."""

    y_soft: torch.Tensor
    target_class: torch.Tensor
    pixel_confidence: torch.Tensor
    conf_target: torch.Tensor
    psr_interior_mask: torch.Tensor


def _class_indices_to_soft(labels: np.ndarray) -> np.ndarray:
    """Convert integer class map ``(H, W)`` to one-hot soft labels ``(K, H, W)``."""
    one_hot = np.zeros((NUM_SEGMENTATION_CLASSES, *labels.shape), dtype=np.float32)
    for class_idx in range(NUM_SEGMENTATION_CLASSES):
        one_hot[class_idx] = (labels == class_idx).astype(np.float32)
    return one_hot


def _derive_labels_from_patch(patch: np.ndarray) -> BatchLabels:
    """Build weak supervision from LOLA PSR fraction when explicit labels are absent."""
    psr = patch[LOLA_PSR_CHANNEL]
    psr_mask = (psr >= 0.5).astype(np.float32)
    class_map = np.zeros_like(psr, dtype=np.int64)
    class_map[psr >= 0.5] = 2
    class_map[(psr >= 0.3) & (psr < 0.5)] = 1

    y_soft = _class_indices_to_soft(class_map)
    target_class = class_map.astype(np.int64)
    pixel_confidence = np.clip(0.5 + 0.5 * np.abs(psr - 0.5), 0.0, 1.0).astype(np.float32)
    conf_target = pixel_confidence.copy()

    return BatchLabels(
        y_soft=torch.from_numpy(y_soft),
        target_class=torch.from_numpy(target_class).unsqueeze(0),
        pixel_confidence=torch.from_numpy(pixel_confidence).unsqueeze(0),
        conf_target=torch.from_numpy(conf_target).unsqueeze(0),
        psr_interior_mask=torch.from_numpy(psr_mask).unsqueeze(0),
    )


class LunarPatchDataset(Dataset):
    """Load normalized fusion patches from Zarr or an in-memory synthetic buffer."""

    def __init__(
        self,
        data_path: Path | None = None,
        synthetic_samples: int = 0,
        seed: int = 42,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.synthetic_samples = synthetic_samples
        self.seed = seed
        self._x: np.ndarray | None = None
        self._y_soft: np.ndarray | None = None
        self._target_class: np.ndarray | None = None

        if synthetic_samples > 0:
            rng = np.random.default_rng(seed)
            self._x = rng.standard_normal(
                (synthetic_samples, NUM_FUSION_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
            )
            labels = rng.integers(0, NUM_SEGMENTATION_CLASSES, (synthetic_samples, PATCH_SIZE, PATCH_SIZE))
            self._target_class = labels.astype(np.int64)
            self._y_soft = np.stack(
                [_class_indices_to_soft(labels[idx]) for idx in range(synthetic_samples)],
                axis=0,
            )
            return

        if self.data_path is None or not self.data_path.exists():
            raise FileNotFoundError(
                "Dataset path not found. Provide a Zarr store or set synthetic_samples > 0."
            )

        try:
            import zarr
        except ImportError as exc:
            raise ImportError("zarr is required to load patch datasets.") from exc

        root = zarr.open_group(str(self.data_path), mode="r")
        if "X" not in root:
            raise ValueError(f"Zarr store '{self.data_path}' is missing dataset 'X'.")
        self._x = np.asarray(root["X"], dtype=np.float32)

        if "y_soft" in root:
            self._y_soft = np.asarray(root["y_soft"], dtype=np.float32)
        elif "Y" in root:
            y = np.asarray(root["Y"])
            if y.ndim == 3:
                self._target_class = y.astype(np.int64)
                self._y_soft = np.stack([_class_indices_to_soft(y[idx]) for idx in range(len(y))], axis=0)
            else:
                self._y_soft = y.astype(np.float32)
        else:
            self._y_soft = None
            self._target_class = None

    def __len__(self) -> int:
        assert self._x is not None
        return int(self._x.shape[0])

    def _labels_for_index(self, index: int, patch: np.ndarray) -> BatchLabels:
        if self._y_soft is not None:
            y_soft = torch.from_numpy(self._y_soft[index])
            target_class = torch.argmax(y_soft, dim=0, keepdim=True)
            pixel_confidence = torch.ones(1, PATCH_SIZE, PATCH_SIZE)
            psr = patch[LOLA_PSR_CHANNEL]
            psr_mask = torch.from_numpy((psr >= 0.5).astype(np.float32)).unsqueeze(0)
            return BatchLabels(
                y_soft=y_soft,
                target_class=target_class,
                pixel_confidence=pixel_confidence,
                conf_target=pixel_confidence.clone(),
                psr_interior_mask=psr_mask,
            )

        if self._target_class is not None:
            class_map = self._target_class[index]
            y_soft = torch.from_numpy(_class_indices_to_soft(class_map))
            psr = patch[LOLA_PSR_CHANNEL]
            psr_mask = torch.from_numpy((psr >= 0.5).astype(np.float32)).unsqueeze(0)
            pixel_confidence = torch.ones(1, PATCH_SIZE, PATCH_SIZE)
            return BatchLabels(
                y_soft=y_soft,
                target_class=torch.from_numpy(class_map.astype(np.int64)).unsqueeze(0),
                pixel_confidence=pixel_confidence,
                conf_target=pixel_confidence.clone(),
                psr_interior_mask=psr_mask,
            )

        return _derive_labels_from_patch(patch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert self._x is not None
        patch = self._x[index]
        if patch.shape != (EXPECTED_INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Patch {index} shape {patch.shape} != "
                f"({EXPECTED_INPUT_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE})."
            )

        labels = self._labels_for_index(index, patch)
        return {
            "inputs": torch.from_numpy(patch.copy()),
            "y_soft": labels.y_soft,
            "target_class": labels.target_class,
            "pixel_confidence": labels.pixel_confidence,
            "conf_target": labels.conf_target,
            "psr_interior_mask": labels.psr_interior_mask,
        }


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack single-patch dicts into a batch."""
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


__all__ = ["BatchLabels", "LunarPatchDataset", "collate_batch"]

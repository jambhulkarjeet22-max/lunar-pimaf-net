"""Visualization helpers shared across LUNAR OS models."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_probability_png(
    probability: np.ndarray,
    output_path: Path | str,
    *,
    title: str = "Probability Map",
    cmap: str = "viridis",
) -> Path:
    """Save a single-band probability raster as a PNG quick-look image."""
    import matplotlib.pyplot as plt

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(5, 5), dpi=120)
    image = axis.imshow(probability, cmap=cmap, vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(resolved, bbox_inches="tight")
    plt.close(fig)
    return resolved

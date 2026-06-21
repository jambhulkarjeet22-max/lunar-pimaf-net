"""Uncertainty helpers shared across LUNAR OS models."""

from __future__ import annotations

import torch


def dirichlet_entropy(alpha: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Compute differential entropy of a Dirichlet distribution parameterized by ``alpha``."""
    sum_alpha = alpha.sum(dim=dim, keepdim=True)
    term1 = torch.lgamma(sum_alpha + eps) - torch.lgamma(alpha + eps).sum(dim=dim, keepdim=True)
    digamma_diff = torch.digamma(alpha + eps) - torch.digamma(sum_alpha + eps)
    term2 = torch.sum((alpha - 1.0) * digamma_diff, dim=dim, keepdim=True)
    return term1 + term2


def normalize_uncertainty_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize an uncertainty map to ``[0, 1]`` for visualization."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.zeros_like(values)
    min_val = finite.min()
    max_val = finite.max()
    return (values - min_val) / (max_val - min_val + eps)

"""Lunar reference constants used across LUNAR OS models."""

from __future__ import annotations

from typing import Final, Literal

Pole = Literal["north", "south"]

POLAR_CRS: Final[dict[Pole, int]] = {
    "north": 104905,
    "south": 104906,
}

DEFAULT_PIXEL_SIZE_M: Final[float] = 240.0
DEFAULT_NODATA: Final[float] = -3.4028235e38
PATCH_SIZE: Final[int] = 128
NUM_FUSION_CHANNELS: Final[int] = 59

STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8
T_TRAP_K: Final[float] = 110.0

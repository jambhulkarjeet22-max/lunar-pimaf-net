"""Geospatial map loading and routing tools for lunar exploration."""

from __future__ import annotations

import heapq
from pathlib import Path
import numpy as np


def load_map(
    model_dir_name: str,
    filename: str,
    shape: tuple[int, int] = (64, 64),
    default_val: float = 0.5,
) -> np.ndarray:
    """Load map from a model's predictions folder, with a synthetic generator fallback."""
    repo_root = Path(__file__).resolve().parents[3]
    pred_path = repo_root / model_dir_name / "predictions" / filename
    if pred_path.exists():
        try:
            arr = np.load(pred_path)
            # Standardize shape to 2D
            if len(arr.shape) == 4:  # [B, C, H, W]
                arr = arr[0, 0]
            elif len(arr.shape) == 3:  # [C, H, W]
                arr = arr[0]
            if arr.shape == shape:
                return arr
        except Exception:
            pass

    # Generate synthetic correlated data as fallback
    # Create smooth structured map
    rng = np.random.default_rng(42 + hash(filename) % 1000)
    raw = rng.standard_normal(shape)
    # Smooth via simple rolling mean / blur
    smooth = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # average surrounding pixels
            x_min, x_max = max(0, i - 2), min(shape[0], i + 3)
            y_min, y_max = max(0, j - 2), min(shape[1], j + 3)
            smooth[i, j] = np.mean(raw[x_min:x_max, y_min:y_max])

    # Normalize to [0, 1]
    s_min, s_max = smooth.min(), smooth.max()
    if s_max > s_min:
        smooth = (smooth - s_min) / (s_max - s_min)
    else:
        smooth = np.full(shape, default_val)
    return smooth


def find_rover_route(
    traversability_map: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """A* 8-neighbor pathfinder minimizing traversal cost.
    
    Cost is defined as: 1.0 + 10.0 * (1.0 - traversability).
    """
    H, W = traversability_map.shape
    cost_map = 1.0 + 10.0 * (1.0 - traversability_map)

    # 8-neighbor steps
    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    h = lambda p: float(np.sqrt((p[0] - end[0])**2 + (p[1] - end[1])**2))
    
    heap = []
    # Format: (f_score, cost_so_far, current_node, path_history)
    heapq.heappush(heap, (h(start), 0.0, start, [start]))
    visited = {}

    while heap:
        f, g, current, path = heapq.heappop(heap)

        if current == end:
            return path

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        x, y = current
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                step_cost = float(cost_map[nx, ny])
                dist_factor = 1.414 if (dx != 0 and dy != 0) else 1.0
                weight = step_cost * dist_factor

                next_g = g + weight
                next_node = (nx, ny)
                if next_node not in visited or visited[next_node] > next_g:
                    next_f = next_g + h(next_node)
                    heapq.heappush(heap, (next_f, next_g, next_node, path + [next_node]))

    # Fallback to straight line if unreachable
    return [start, end]

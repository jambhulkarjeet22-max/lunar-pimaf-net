#!/usr/bin/env python3
"""Run offline mission planning query using the ORACLE coordinator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.dataset_utils import ensure_import_paths
ensure_import_paths(Path(__file__).resolve().parent)

from Model_6_ORACLE_Mission_Copilot.src.orchestration.coordinator import AgentCoordinator


def main() -> None:
    print("Executing ORACLE offline mission planner query...")
    coordinator = AgentCoordinator()
    params = {
        "landing_zone_search_area": "Shackleton Crater Rim East",
        "duration_days": 14,
        "safety_margin": 0.85,
        "rover_range_m": 6000.0,
    }
    response = coordinator.plan_mission(params)
    print("\nMission Plan Generated successfully:")
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()

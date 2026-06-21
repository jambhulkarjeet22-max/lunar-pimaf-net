"""Agent communication and coordination smoke test for ORACLE Copilot."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_6_ORACLE_Mission_Copilot.src.orchestration.coordinator import AgentCoordinator


def main() -> None:
    print("Running agent communication smoke test...")
    coordinator = AgentCoordinator()
    
    params = {
        "landing_zone_search_area": "Shackleton Crater",
        "duration_days": 10,
        "safety_margin": 0.9,
        "rover_range_m": 4000.0,
    }
    
    response = coordinator.plan_mission(params)
    
    # Assert logs from all expected agents are present
    expected_agents = [
        "LandingAgent",
        "IceAgent",
        "RadiationAgent",
        "NavigationAgent",
        "ScienceAgent",
        "MissionPlannerAgent",
    ]
    
    for agent in expected_agents:
        assert agent in response.agent_logs, f"Missing logs for {agent}"
        assert len(response.agent_logs[agent]) > 0, f"Empty logs for {agent}"
        
    # Assert correctness of coordinates
    assert 0 <= response.best_landing_site.coordinate.x < 64
    assert 0 <= response.best_landing_site.coordinate.y < 64
    assert 0 <= response.best_drilling_site.coordinate.x < 64
    assert 0 <= response.best_drilling_site.coordinate.y < 64
    
    # Assert route waypoint count matches start/end
    assert len(response.rover_route_recommendation.route_waypoints) >= 2
    
    print("agent smoke tests passed")


if __name__ == "__main__":
    main()

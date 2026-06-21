"""API routes and startup smoke test for ORACLE Copilot."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient
from Model_6_ORACLE_Mission_Copilot.src.api.app import app


def main() -> None:
    print("Running API route smoke test...")
    client = TestClient(app)
    
    # Test POST /mission/plan
    plan_payload = {
        "landing_zone_search_area": "Shackleton Crater West Rim",
        "duration_days": 10,
        "safety_margin": 0.75,
        "rover_range_m": 5000.0,
    }
    
    response = client.post("/mission/plan", json=plan_payload)
    assert response.status_code == 200, f"Plan endpoint failed: {response.text}"
    
    data = response.json()
    assert "best_landing_site" in data
    assert "best_drilling_site" in data
    assert "habitat_recommendation" in data
    assert "rover_route_recommendation" in data
    assert "mission_risk_summary" in data
    
    # Test POST /mission/analyze
    analyze_payload = {
        "map_type": "ice",
        "threshold": 0.6,
    }
    
    response = client.post("/mission/analyze", json=analyze_payload)
    assert response.status_code == 200, f"Analyze endpoint failed: {response.text}"
    
    analyze_data = response.json()
    assert analyze_data["map_type"] == "ice"
    assert "grid_size" in analyze_data
    assert "active_pixel_count" in analyze_data
    assert "max_score_coordinate" in analyze_data
    
    print("api smoke tests passed")


if __name__ == "__main__":
    main()

"""FastAPI application for ORACLE Mission Copilot."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from ..orchestration.coordinator import AgentCoordinator
from ..schemas.mission_schema import (
    AnalysisRequest,
    AnalysisResponse,
    Coordinate,
    MissionRequest,
    MissionResponse,
)
from ..tools.map_tools import load_map
from ..tools.scoring_tools import find_best_coordinate

app = FastAPI(
    title="ORACLE Mission Copilot API",
    description="LLM-powered multi-agent mission planning system for lunar exploration.",
    version="1.0.0",
)


@app.post("/mission/plan", response_model=MissionResponse)
def plan_mission_endpoint(request: MissionRequest) -> MissionResponse:
    """Orchestrate all agents to generate a synthesized exploration report."""
    try:
        coordinator = AgentCoordinator()
        response = coordinator.plan_mission(request.model_dump())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate mission plan: {str(e)}")


@app.post("/mission/analyze", response_model=AnalysisResponse)
def analyze_map_endpoint(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze geospatial layers from Models 1-5 and return statistical metrics."""
    try:
        # Map requested type to file path and default
        if request.map_type == "ice":
            model_dir = "Model_1_Physics_Informed_Ice_Detection"
            filename = "ice_detection.npy"
            default_val = 0.6
        elif request.map_type == "landing":
            model_dir = "Model_3_Landing_Site_Intelligence"
            filename = "final_suitability_score.npy"
            default_val = 0.7
        elif request.map_type == "radiation":
            model_dir = "Model_4_Radiation_Risk_Prediction"
            filename = "radiation_risk_score.npy"
            default_val = 0.3
        elif request.map_type == "traversability":
            model_dir = "Model_5_Rover_Hazard_Navigation"
            filename = "traversability_map.npy"
            default_val = 0.8
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown map_type: {request.map_type}. Expected 'ice', 'landing', 'radiation', or 'traversability'.",
            )

        grid = load_map(model_dir, filename, default_val=default_val)
        H, W = grid.shape

        # Calculate metrics
        active_pixels = int(np.sum(grid >= request.threshold))
        max_x, max_y, max_val = find_best_coordinate(grid)

        summary_text = (
            f"Geospatial analysis of {request.map_type} map complete. Grid resolution: {H}x{W}. "
            f"Found {active_pixels} pixels exceeding the threshold of {request.threshold}. "
            f"Maximum value of {max_val:.4f} is situated at coordinate ({max_x}, {max_y})."
        )

        return AnalysisResponse(
            map_type=request.map_type,
            grid_size=[H, W],
            active_pixel_count=active_pixels,
            max_score_coordinate=Coordinate(x=max_x, y=max_y),
            max_score_value=max_val,
            summary=summary_text,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze map: {str(e)}")

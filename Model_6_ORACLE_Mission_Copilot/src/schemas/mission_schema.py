"""Pydantic schemas for the ORACLE Mission Copilot API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MissionRequest(BaseModel):
    """Input parameters for mission planning."""

    landing_zone_search_area: str = Field(
        default="South Pole - Shackleton Crater Rim",
        description="Text description or coordinate bounds of the target landing area.",
    )
    duration_days: int = Field(
        default=14, ge=1, description="Target duration of the surface exploration mission."
    )
    safety_margin: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Risk tolerance threshold (0 = high risk, 1 = ultra safe)."
    )
    rover_range_m: float = Field(
        default=5000.0, ge=100.0, description="Maximum traverse distance capability of the rover."
    )


class Coordinate(BaseModel):
    x: int = Field(..., description="X coordinate on the geospatial grid index.")
    y: int = Field(..., description="Y coordinate on the geospatial grid index.")


class LandingSiteInfo(BaseModel):
    coordinate: Coordinate
    suitability_score: float = Field(..., ge=0.0, le=1.0)
    description: str


class DrillingSiteInfo(BaseModel):
    coordinate: Coordinate
    ice_concentration_estimate: float = Field(..., ge=0.0, le=1.0)
    depth_m: float = Field(..., ge=0.0)
    description: str


class HabitatInfo(BaseModel):
    coordinate: Coordinate
    radiation_exposure_mSv_day: float = Field(..., ge=0.0)
    shielding_effectiveness_score: float = Field(..., ge=0.0, le=1.0)
    safety_score: float = Field(..., ge=0.0, le=1.0)
    description: str


class RoverRouteInfo(BaseModel):
    start_coordinate: Coordinate
    end_coordinate: Coordinate
    route_waypoints: list[Coordinate] = Field(default_factory=list)
    total_distance_m: float = Field(..., ge=0.0)
    average_traversability_score: float = Field(..., ge=0.0, le=1.0)
    description: str


class RiskSummary(BaseModel):
    overall_risk_level: str = Field(..., description="Overall risk level: 'Low', 'Medium', 'High', 'Critical'.")
    hazard_factors: list[str] = Field(default_factory=list)
    recommending_mitigations: list[str] = Field(default_factory=list)


class MissionResponse(BaseModel):
    """Output schema for the synthesized mission plan."""

    best_landing_site: LandingSiteInfo
    best_drilling_site: DrillingSiteInfo
    habitat_recommendation: HabitatInfo
    rover_route_recommendation: RoverRouteInfo
    mission_risk_summary: RiskSummary
    agent_logs: dict[str, str] = Field(
        ..., description="Key reasoning summaries or logs from individual agents."
    )


class AnalysisRequest(BaseModel):
    """Input parameters for map analysis."""

    map_type: str = Field(
        ..., description="Type of map to analyze: 'ice', 'landing', 'radiation', 'traversability'."
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Score threshold for filtering regions.")


class AnalysisResponse(BaseModel):
    """Output metrics for map analysis."""

    map_type: str
    grid_size: list[int]
    active_pixel_count: int
    max_score_coordinate: Coordinate
    max_score_value: float
    summary: str

"""Orchestrator for multi-agent exploration planning."""

from __future__ import annotations

from typing import Any

from ..agents.ice_agent import IceAgent
from ..agents.landing_agent import LandingAgent
from ..agents.mission_planner_agent import MissionPlannerAgent
from ..agents.navigation_agent import NavigationAgent
from ..agents.radiation_agent import RadiationAgent
from ..agents.science_agent import ScienceAgent
from ..llm.oracle_llm import OracleLLM
from ..schemas.mission_schema import (
    Coordinate,
    DrillingSiteInfo,
    HabitatInfo,
    LandingSiteInfo,
    MissionResponse,
    RiskSummary,
    RoverRouteInfo,
)
from .memory import SharedMemory


class AgentCoordinator:
    """Orchestrates sequential execution of specialized agents and aggregates final reports."""

    def __init__(self, llm_key: str | None = None) -> None:
        self.llm = OracleLLM(api_key=llm_key)
        self.landing_agent = LandingAgent(self.llm)
        self.ice_agent = IceAgent(self.llm)
        self.radiation_agent = RadiationAgent(self.llm)
        self.navigation_agent = NavigationAgent(self.llm)
        self.science_agent = ScienceAgent(self.llm)
        self.planner_agent = MissionPlannerAgent(self.llm)

    def plan_mission(self, params: dict[str, Any]) -> MissionResponse:
        memory = SharedMemory()
        memory.set_parameters(params)

        # Execute agents sequentially
        self.landing_agent.run(memory)
        self.ice_agent.run(memory)
        self.radiation_agent.run(memory)
        self.navigation_agent.run(memory)
        self.science_agent.run(memory)
        self.planner_agent.run(memory)

        # Retrieve outputs
        landing_out = memory.read_agent_output("LandingAgent") or {}
        ice_out = memory.read_agent_output("IceAgent") or {}
        radiation_out = memory.read_agent_output("RadiationAgent") or {}
        navigation_out = memory.read_agent_output("NavigationAgent") or {}

        # Consolidate Landing Info
        landing_coord = landing_out.get("coordinate", (10, 10))
        best_landing = LandingSiteInfo(
            coordinate=Coordinate(x=landing_coord[0], y=landing_coord[1]),
            suitability_score=landing_out.get("suitability_score", 0.0),
            description=landing_out.get("description", "touchdown site selection"),
        )

        # Consolidate Drilling Info
        ice_coord = ice_out.get("coordinate", (40, 40))
        best_drilling = DrillingSiteInfo(
            coordinate=Coordinate(x=ice_coord[0], y=ice_coord[1]),
            ice_concentration_estimate=ice_out.get("ice_concentration_estimate", 0.0),
            depth_m=ice_out.get("depth_m", 0.0),
            description=ice_out.get("description", "drill target selection"),
        )

        # Consolidate Habitat Info
        habitat_coord = radiation_out.get("coordinate", (15, 15))
        best_habitat = HabitatInfo(
            coordinate=Coordinate(x=habitat_coord[0], y=habitat_coord[1]),
            radiation_exposure_mSv_day=radiation_out.get("radiation_exposure_mSv_day", 0.0),
            shielding_effectiveness_score=radiation_out.get("shielding_effectiveness_score", 0.0),
            safety_score=radiation_out.get("safety_score", 0.0),
            description=radiation_out.get("description", "shielded base selection"),
        )

        # Consolidate Rover Path Info
        nav_start = navigation_out.get("start_coordinate", (10, 10))
        nav_end = navigation_out.get("end_coordinate", (40, 40))
        nav_waypoints = navigation_out.get("route_waypoints", [nav_start, nav_end])
        waypoints_list = [Coordinate(x=p[0], y=p[1]) for p in nav_waypoints]
        best_route = RoverRouteInfo(
            start_coordinate=Coordinate(x=nav_start[0], y=nav_start[1]),
            end_coordinate=Coordinate(x=nav_end[0], y=nav_end[1]),
            route_waypoints=waypoints_list,
            total_distance_m=navigation_out.get("total_distance_m", 0.0),
            average_traversability_score=navigation_out.get("average_traversability_score", 0.0),
            description=navigation_out.get("description", "A* cost routing"),
        )

        # Consolidate Risks & Mitigations
        safety_margin = params.get("safety_margin", 0.8)
        risk_level = "Low" if safety_margin >= 0.8 else "Medium"
        risk_summary = RiskSummary(
            overall_risk_level=risk_level,
            hazard_factors=[
                "High elevation GCR exposure risk",
                "Crater slope slippage risk during rover transit",
            ],
            recommending_mitigations=[
                "Emplace habitat in regolith depression (>2m shielding)",
                "Follow spatial attention ridge cost-path avoiding steep bounds",
            ],
        )

        return MissionResponse(
            best_landing_site=best_landing,
            best_drilling_site=best_drilling,
            habitat_recommendation=best_habitat,
            rover_route_recommendation=best_route,
            mission_risk_summary=risk_summary,
            agent_logs=memory.get_all_logs(),
        )

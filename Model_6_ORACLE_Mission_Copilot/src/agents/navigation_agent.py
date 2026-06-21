"""Rover traversability and path routing agent."""

from __future__ import annotations

import numpy as np

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory
from ..tools.map_tools import find_rover_route, load_map


class NavigationAgent:
    """Analyzes rover traversability maps and plans safe, low-cost navigation paths between sites."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Load traversability map (with synthetic fallback)
        trav_map = load_map("Model_5_Rover_Hazard_Navigation", "traversability_map.npy", default_val=0.8)

        # Retrieve landing site and drilling site coordinates from memory
        landing_info = memory.read_agent_output("LandingAgent")
        ice_info = memory.read_agent_output("IceAgent")

        start_coord = landing_info["coordinate"] if landing_info else (10, 10)
        end_coord = ice_info["coordinate"] if ice_info else (40, 40)

        # Plan the path
        path = find_rover_route(trav_map, start_coord, end_coord)

        # Calculate metrics
        route_len_pixels = len(path)
        total_distance = float(route_len_pixels * 120.0)
        
        # Calculate average traversability along path
        trav_values = [trav_map[x, y] for x, y in path]
        avg_trav = float(np.mean(trav_values))

        # Prompt LLM to synthesize explanation
        prompt = (
            f"As the NavigationAgent, explain the planned route from start ({start_coord[0]}, {start_coord[1]}) "
            f"to end ({end_coord[0]}, {end_coord[1]}). The total distance is {total_distance:.2f} meters, and the "
            f"average traversability along the path is {avg_trav:.4f}. Waypoint count: {len(path)}."
        )
        explanation = self.llm.generate(prompt)

        # Write findings to shared memory
        memory.write_agent_output(
            agent_name="NavigationAgent",
            data={
                "start_coordinate": start_coord,
                "end_coordinate": end_coord,
                "route_waypoints": path,
                "total_distance_m": total_distance,
                "average_traversability_score": avg_trav,
                "description": explanation,
            },
            log_summary=explanation,
        )

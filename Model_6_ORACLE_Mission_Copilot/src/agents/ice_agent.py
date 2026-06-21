"""Ice volatile and subsurface resource analyzer agent."""

from __future__ import annotations

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory
from ..tools.map_tools import load_map
from ..tools.scoring_tools import find_best_coordinate, score_drilling_sites


class IceAgent:
    """Analyzes ice maps and resources to select optimal, high-yield drilling targets."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Load maps (with synthetic fallbacks)
        ice_det = load_map("Model_1_Physics_Informed_Ice_Detection", "ice_detection.npy", default_val=0.6)
        ice_depth = load_map("Model_2_Ice_Characterization", "ice_depth.npy", default_val=0.4)
        traversability = load_map("Model_5_Rover_Hazard_Navigation", "traversability_map.npy", default_val=0.8)

        # Score drilling sites
        scores = score_drilling_sites(ice_det, ice_depth, traversability)
        x, y, best_score = find_best_coordinate(scores)

        # Estimate parameters
        ice_concentration = float(ice_det[x, y])
        depth = float(ice_depth[x, y] * 2.0)  # assume scaled depth up to 2 meters

        # Prompt LLM to synthesize explanation
        prompt = (
            f"As the IceAgent, explain the selection of the drilling target at ({x}, {y}) "
            f"with ice concentration estimate {ice_concentration:.4f} at depth {depth:.2f}m."
        )
        explanation = self.llm.generate(prompt)

        # Write findings to shared memory
        memory.write_agent_output(
            agent_name="IceAgent",
            data={
                "coordinate": (x, y),
                "ice_concentration_estimate": ice_concentration,
                "depth_m": depth,
                "description": explanation,
            },
            log_summary=explanation,
        )

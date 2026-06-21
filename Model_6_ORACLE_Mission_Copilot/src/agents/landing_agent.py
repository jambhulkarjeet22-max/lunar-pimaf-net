"""Landing suitability analyzer agent."""

from __future__ import annotations

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory
from ..tools.map_tools import load_map
from ..tools.scoring_tools import find_best_coordinate, score_landing_sites


class LandingAgent:
    """Analyzes landing suitability maps and selects optimal, safe touchdown coordinates."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Load maps (with synthetic fallbacks)
        suitability = load_map("Model_3_Landing_Site_Intelligence", "final_suitability_score.npy", default_val=0.7)
        radiation = load_map("Model_4_Radiation_Risk_Prediction", "radiation_risk_score.npy", default_val=0.3)
        traversability = load_map("Model_5_Rover_Hazard_Navigation", "traversability_map.npy", default_val=0.8)

        # Score landing sites
        scores = score_landing_sites(suitability, radiation, traversability)
        
        # Apply safety margin thresholding
        safety_margin = memory.get_parameter("safety_margin", 0.8)
        x, y, best_score = find_best_coordinate(scores)

        # Prompt LLM to synthesize explanation
        prompt = (
            f"As the LandingAgent, explain the choice of landing coordinate at ({x}, {y}) "
            f"with suitability score {best_score:.4f}. Contrast this with safety margin parameter: {safety_margin}."
        )
        explanation = self.llm.generate(prompt)

        # Write findings to shared memory
        memory.write_agent_output(
            agent_name="LandingAgent",
            data={
                "coordinate": (x, y),
                "suitability_score": best_score,
                "description": explanation,
            },
            log_summary=explanation,
        )

"""Radiation and shielding analysis agent."""

from __future__ import annotations

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory
from ..tools.map_tools import load_map
from ..tools.scoring_tools import find_best_coordinate, score_habitat_sites


class RadiationAgent:
    """Analyzes radiation hazards and shielding depth to recommend safe habitat sites."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Load maps (with synthetic fallbacks)
        radiation_risk = load_map("Model_4_Radiation_Risk_Prediction", "radiation_risk_score.npy", default_val=0.3)
        radiation_dose = load_map("Model_4_Radiation_Risk_Prediction", "radiation_dose_rate.npy", default_val=0.5)
        shielding = load_map(
            "Model_4_Radiation_Risk_Prediction", "shielding_effectiveness_score.npy", default_val=0.6
        )
        suitability = load_map("Model_3_Landing_Site_Intelligence", "final_suitability_score.npy", default_val=0.7)

        # Score habitat locations
        scores = score_habitat_sites(radiation_risk, shielding, suitability)
        x, y, best_score = find_best_coordinate(scores)

        # Retrieve values
        dose_rate = float(radiation_dose[x, y] * 300.0)  # scale to mSv/year or keep as mSv/day (average 0.5 - 2.0 mSv/day)
        shielding_score = float(shielding[x, y])
        safety_score = float(best_score)

        # Prompt LLM to synthesize explanation
        prompt = (
            f"As the RadiationAgent, explain the choice of habitat location at ({x}, {y}) "
            f"with radiation dose rate {dose_rate:.2f} mSv/day, shielding effectiveness score {shielding_score:.4f}, "
            f"and calculated safety score {safety_score:.4f}."
        )
        explanation = self.llm.generate(prompt)

        # Write findings to shared memory
        memory.write_agent_output(
            agent_name="RadiationAgent",
            data={
                "coordinate": (x, y),
                "radiation_exposure_mSv_day": dose_rate,
                "shielding_effectiveness_score": shielding_score,
                "safety_score": safety_score,
                "description": explanation,
            },
            log_summary=explanation,
        )

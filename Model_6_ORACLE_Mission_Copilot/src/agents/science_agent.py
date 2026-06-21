"""Science analysis and hypothesis generator agent."""

from __future__ import annotations

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory


class ScienceAgent:
    """Generates scientific explanations, exploration hypotheses, and geologic value statements."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Retrieve target ice resource coordinates
        ice_info = memory.read_agent_output("IceAgent")
        coord = ice_info["coordinate"] if ice_info else (40, 40)
        ice_conc = ice_info["ice_concentration_estimate"] if ice_info else 0.6

        # Prompt LLM to synthesize explanation
        prompt = (
            f"As the ScienceAgent, write a short scientific hypothesis for the selected drilling site at "
            f"coordinate ({coord[0]}, {coord[1]}) which shows an ice concentration estimate of {ice_conc:.4f}. "
            f"Focus on geological history and volatile origin."
        )
        explanation = self.llm.generate(prompt)

        # Write findings to shared memory
        memory.write_agent_output(
            agent_name="ScienceAgent",
            data={
                "drilling_site": coord,
                "hypothesis_summary": explanation,
            },
            log_summary=explanation,
        )

"""Master mission planner agent."""

from __future__ import annotations

from ..llm.oracle_llm import OracleLLM
from ..orchestration.memory import SharedMemory


class MissionPlannerAgent:
    """Ingests findings from specialized agents to formulate the final exploration mission plan."""

    def __init__(self, llm: OracleLLM) -> None:
        self.llm = llm

    def run(self, memory: SharedMemory) -> None:
        # Read findings from all agents
        landing = memory.read_agent_output("LandingAgent")
        ice = memory.read_agent_output("IceAgent")
        radiation = memory.read_agent_output("RadiationAgent")
        navigation = memory.read_agent_output("NavigationAgent")
        science = memory.read_agent_output("ScienceAgent")

        # Compile summaries
        summary_findings = (
            f"Landing Site: {landing}\n"
            f"Drilling Site: {ice}\n"
            f"Radiation Safety: {radiation}\n"
            f"Navigation Corridor: {navigation}\n"
            f"Science Objectives: {science}\n"
        )

        # Prompt LLM to synthesize final plan
        prompt = (
            f"As the MissionPlannerAgent, review the specialized agent outputs and construct the "
            f"final synthesized lunar exploration recommendation report.\n\n"
            f"{summary_findings}"
        )
        final_recommendation = self.llm.generate(prompt)

        # Write overall recommendation to shared memory
        memory.write_agent_output(
            agent_name="MissionPlannerAgent",
            data={
                "recommendation_summary": final_recommendation,
            },
            log_summary=final_recommendation,
        )

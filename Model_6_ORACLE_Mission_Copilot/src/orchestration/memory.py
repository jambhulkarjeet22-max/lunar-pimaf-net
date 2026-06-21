"""Shared memory for multi-agent coordination."""

from __future__ import annotations

from typing import Any


class SharedMemory:
    """Collects parameters, intermediate agent findings, and conversation logs."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.agent_logs: dict[str, str] = {}
        self.parameters: dict[str, Any] = {}

    def set_parameters(self, params: dict[str, Any]) -> None:
        self.parameters.update(params)

    def get_parameter(self, key: str, default: Any = None) -> Any:
        return self.parameters.get(key, default)

    def write_agent_output(self, agent_name: str, data: Any, log_summary: str) -> None:
        self.state[agent_name] = data
        self.agent_logs[agent_name] = log_summary

    def read_agent_output(self, agent_name: str) -> Any | None:
        return self.state.get(agent_name)

    def get_all_outputs(self) -> dict[str, Any]:
        return self.state

    def get_all_logs(self) -> dict[str, str]:
        return self.agent_logs

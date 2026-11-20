"""Ground-truth leak detector (`leak_checker.py`).

Scans agent observations, error messages, info dicts, file paths, and recorded
trajectories (`trajectories.jsonl`) to verify that closed ground-truth markers
(`truth.json`, `provenance.json`, `ISS-`, seed values, or hidden explanations)
never leak to the agent.
"""

from __future__ import annotations

from typing import Any


def check_observation_for_leaks(
    observation: dict[str, Any],
    info: dict[str, Any],
    truth_data: dict[str, Any] | None = None,
) -> list[str]:
    """Inspect an observation/info pair and return a list of detected leak violations (empty if clean)."""
    raise NotImplementedError("check_observation_for_leaks is not implemented yet.")

"""Abstract base policy interface (`base.py`).

Defines the uniform contract for policies interacting with `FinAuditEnv`
or `IsolatedFinAuditClient`.
"""

from __future__ import annotations

from typing import Any


class BasePolicy:
    """Base interface for all financial audit RL and heuristic policies."""

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = seed

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset internal policy state at the start of an episode."""
        raise NotImplementedError("BasePolicy.reset is not implemented yet.")

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Select the next JSON-serializable action given the latest observation."""
        raise NotImplementedError("BasePolicy.act is not implemented yet.")

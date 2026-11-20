"""Core RL environment implementation (`core.py`).

Implements the Gymnasium-style JSON-compatible `reset(seed, config)` and
`step(action)` loop with a 100-step budget, bounded error handling, and
strict separation from closed ground-truth data.
"""

from __future__ import annotations

from typing import Any


class FinAuditEnv:
    """Reproducible financial audit RL environment (`reset` / `step` contract)."""

    MAX_STEPS: int = 100

    def __init__(self, mode: str = "train", **kwargs: Any) -> None:
        raise NotImplementedError("FinAuditEnv.__init__ is not implemented yet.")

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment state and return initial `(observation, info)`."""
        raise NotImplementedError("FinAuditEnv.reset is not implemented yet.")

    def step(
        self,
        action: dict[str, Any] | Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one agent action and return `(observation, reward, terminated, truncated, info)`."""
        raise NotImplementedError("FinAuditEnv.step is not implemented yet.")

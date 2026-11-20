"""Evaluator-only ground-truth oracle policy (`oracle.py`).

Uses evaluator ground truth (`TruthFile`) to inspect the required evidence
records through the standard tools and submit exact findings, verifying the
upper bound of terminal reward.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy
from finaudit_env.schemas import TruthFile


class OraclePolicy(BasePolicy):
    """Evaluator-only oracle policy that verifies maximum achievable reward."""

    def __init__(self, truth: TruthFile | None = None, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self.truth = truth

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset oracle action queue for a new episode."""
        raise NotImplementedError("OraclePolicy.reset is not implemented yet.")

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Inspect evidence records and submit ground-truth findings."""
        raise NotImplementedError("OraclePolicy.act is not implemented yet.")

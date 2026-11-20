"""Adversarial 'flag-all' specification-gaming attack policy (`flag_all_attack.py`).

Attempts to maximize recall by spamming maximum allowed findings (up to 30)
across seen record IDs and issue types; used to verify that false positives
penalize precision, F1, and terminal reward.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy


class FlagAllAttackPolicy(BasePolicy):
    """Adversarial policy that flags as many records/types as possible."""

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset adversarial state for a new episode."""
        raise NotImplementedError("FlagAllAttackPolicy.reset is not implemented yet.")

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Collect IDs quickly and submit maximum findings payload."""
        raise NotImplementedError("FlagAllAttackPolicy.act is not implemented yet.")

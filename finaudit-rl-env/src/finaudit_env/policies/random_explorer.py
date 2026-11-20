"""Fixed-seed random explorer baseline policy (`random_explorer.py`).

Samples random valid read actions for a random number of steps before
submitting empty or randomly sampled findings.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy


class RandomExplorerPolicy(BasePolicy):
    """Random explorer baseline policy with deterministic seed."""

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset random explorer state for a new episode."""
        raise NotImplementedError("RandomExplorerPolicy.reset is not implemented yet.")

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Choose a random valid tool call or submission."""
        raise NotImplementedError("RandomExplorerPolicy.act is not implemented yet.")

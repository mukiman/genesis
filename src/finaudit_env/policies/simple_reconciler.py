"""Simple heuristic reconciliation baseline policy (`simple_reconciler.py`).

Paginates through open records and performs basic rule-based checks
(e.g., exact invoice duplicates or unreferenced payments).
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy


class SimpleReconcilerPolicy(BasePolicy):
    """Heuristic reconciliation baseline policy."""

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset reconciliation buffers for a new episode."""
        raise NotImplementedError("SimpleReconcilerPolicy.reset is not implemented yet.")

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Execute next pagination/inspection step or submit reconciled findings."""
        raise NotImplementedError("SimpleReconcilerPolicy.act is not implemented yet.")

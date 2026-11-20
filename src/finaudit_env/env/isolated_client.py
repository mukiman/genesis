"""Isolated client wrapper for agent interaction (`isolated_client.py`).

Provides the standard `reset(seed, config)` and `step(action)` interface to an
agent while communicating with an out-of-process `EvaluatorServer`.
"""

from __future__ import annotations

from typing import Any


class IsolatedFinAuditClient:
    """Client proxy that forwards `reset` and `step` calls to `EvaluatorServer`."""

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError("IsolatedFinAuditClient.__init__ is not implemented yet.")

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward `reset` to the isolated evaluator server."""
        raise NotImplementedError("IsolatedFinAuditClient.reset is not implemented yet.")

    def step(
        self,
        action: dict[str, Any] | Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Forward `step` to the isolated evaluator server."""
        raise NotImplementedError("IsolatedFinAuditClient.step is not implemented yet.")

    def close(self) -> None:
        """Shut down the isolated evaluator connection."""
        raise NotImplementedError("IsolatedFinAuditClient.close is not implemented yet.")

"""Out-of-process evaluator server (`evaluator_server.py`).

Hosts the environment and closed ground-truth evaluator (`truth.json`,
`provenance.json`) in an isolated process so that the agent process cannot
access truth files, seeds, or internal evaluator memory.
"""

from __future__ import annotations

from typing import Any


class EvaluatorServer:
    """Isolated evaluator server handling `reset` and `step` requests over IPC."""

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError("EvaluatorServer.__init__ is not implemented yet.")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a single JSON IPC request (`reset` or `step`) without leaking truth."""
        raise NotImplementedError("EvaluatorServer.handle_request is not implemented yet.")

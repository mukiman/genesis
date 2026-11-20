"""Causal financial error injection engine (`errors.py`).

Injects the 4 canonical `IssueType` errors (`duplicate_invoice`,
`unallocated_or_partial_payment`, `fx_conversion_error`, `period_cutoff_error`)
as well as interacting multi-error scenarios, recording `TruthIssue` and
`ProvenanceEvent` entries.
"""

from __future__ import annotations

from typing import Any


def inject_errors(
    clean_episode: dict[str, Any],
    seed: int,
    error_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject configured causal errors into a clean company episode and build truth/provenance."""
    raise NotImplementedError("inject_errors is not implemented yet.")

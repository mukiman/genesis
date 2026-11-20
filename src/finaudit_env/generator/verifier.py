"""Independent episode and ground-truth verifier (`verifier.py`).

Verifies pre-mutation accounting identities, post-mutation discrepancy accuracy,
and independent sufficiency of `valid_evidence_sets` in `truth.json` using only
open records.
"""

from __future__ import annotations

from typing import Any


def verify_episode(episode_data: dict[str, Any]) -> dict[str, Any]:
    """Verify accounting invariants and evidence sufficiency for a generated episode."""
    raise NotImplementedError("verify_episode is not implemented yet.")

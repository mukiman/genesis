"""Isomorphic episode transformations (`transforms.py`).

Applies benign record reordering and canonical ID remapping while preserving
underlying financial semantics and updating `truth.json` / `provenance.json`
for paired generalization checks.
"""

from __future__ import annotations

from typing import Any


def permute_and_remap_episode(
    episode_data: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Permute record order and remap IDs, returning transformed episode and ID mapping."""
    raise NotImplementedError("permute_and_remap_episode is not implemented yet.")

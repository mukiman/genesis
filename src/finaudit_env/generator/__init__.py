"""Synthetic company & financial error generator subpackage (`finaudit_env.generator`).

Owned by Participant 1 (`feat/generator` branch).
"""

from __future__ import annotations

from typing import Any

from finaudit_env.generator.company import generate_clean_company
from finaudit_env.generator.errors import inject_errors
from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    extract_iso_date,
    generate_fx_rates,
    lookup_fx_rate,
    resolve_effective_fx_date,
)
from finaudit_env.generator.transforms import permute_and_remap_episode, remap_submission_ids
from finaudit_env.generator.verifier import verify_episode


def generate_episode(seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a full synthetic company episode (clean generation + configured error injection)."""
    cfg = dict(config or {})
    clean_ep = generate_clean_company(seed=seed, config=cfg)
    error_cfg = cfg.get("error_config", cfg)
    return inject_errors(clean_episode=clean_ep, seed=seed + 1000, error_config=error_cfg)


__all__ = [
    "convert_amount_minor",
    "extract_iso_date",
    "generate_clean_company",
    "generate_episode",
    "generate_fx_rates",
    "inject_errors",
    "lookup_fx_rate",
    "permute_and_remap_episode",
    "remap_submission_ids",
    "resolve_effective_fx_date",
    "verify_episode",
]

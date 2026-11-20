"""Foreign exchange rate generation and minor-unit currency conversion utilities (`fx_utils.py`).

Handles deterministic daily FX table generation (`FxRateRow`) and policy-compliant
minor-unit conversion with configurable rounding modes.
"""

from __future__ import annotations

from finaudit_env.schemas import FxRateRow, PolicyConfig


def generate_fx_rates(
    seed: int,
    base_currency: str,
    currencies: list[str],
    period_start: str,
    period_end: str,
) -> list[FxRateRow]:
    """Generate deterministic daily FX rate rows for the given reporting window."""
    raise NotImplementedError("generate_fx_rates is not implemented yet.")


def convert_amount_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate_str: str,
    rounding_mode: str = "HALF_EVEN",
) -> int:
    """Convert an integer minor-unit amount between currencies using a decimal rate string."""
    raise NotImplementedError("convert_amount_minor is not implemented yet.")


def lookup_fx_rate(
    fx_rows: list[FxRateRow],
    source_currency: str,
    target_currency: str,
    date: str,
    policy: PolicyConfig | None = None,
) -> FxRateRow:
    """Resolve the applicable FX rate row according to company accounting policy."""
    raise NotImplementedError("lookup_fx_rate is not implemented yet.")

"""Foreign exchange rate generation and minor-unit currency conversion utilities (`fx_utils.py`).

Handles deterministic daily FX table generation (`FxRateRow`) and policy-compliant
minor-unit conversion with configurable rounding modes.
"""

from __future__ import annotations

import datetime as dt
import random
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP

from finaudit_env.schemas import CURRENCY_EXPONENTS, FxRateRow, PolicyConfig

# Map schema RoundingMode literals to Python decimal rounding constants
_ROUNDING_MAP: dict[str, str] = {
    "HALF_EVEN": ROUND_HALF_EVEN,
    "HALF_UP": ROUND_HALF_UP,
    "DOWN": ROUND_DOWN,
    "UP": ROUND_UP,
}

# Realistic anchor rates: how many USD per 1 major unit of currency
_USD_PER_UNIT_ANCHOR: dict[str, Decimal] = {
    "USD": Decimal("1.000000"),
    "EUR": Decimal("1.085000"),
    "GBP": Decimal("1.265000"),
    "CHF": Decimal("1.125000"),
    "KZT": Decimal("0.002060"),  # ~485.44 KZT per USD
    "JPY": Decimal("0.006650"),  # ~150.38 JPY per USD
}


def extract_iso_date(date_or_timestamp: str) -> dt.date:
    """Extract a `datetime.date` from an ISO-8601 date or timestamp string."""
    clean = date_or_timestamp.strip()[:10]
    return dt.date.fromisoformat(clean)


def resolve_effective_fx_date(
    transaction_date: str,
    policy: PolicyConfig | None = None,
    invoice_issue_date: str | None = None,
) -> str:
    """Compute the effective YYYY-MM-DD date for FX lookup under the given accounting policy."""
    rule = policy.fx_rate_date_rule if policy is not None else "transaction_date"

    if rule == "invoice_issue_date" and invoice_issue_date is not None:
        base_date = extract_iso_date(invoice_issue_date)
    else:
        base_date = extract_iso_date(transaction_date)

    if rule == "previous_business_day":
        prev_day = base_date - dt.timedelta(days=1)
        while prev_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            prev_day -= dt.timedelta(days=1)
        return prev_day.isoformat()

    return base_date.isoformat()


def generate_fx_rates(
    seed: int,
    base_currency: str,
    currencies: list[str],
    period_start: str,
    period_end: str,
    padding_days: int = 15,
) -> list[FxRateRow]:
    """Generate deterministic daily FX rate rows for the reporting window plus padding days.

    Padding days before `period_start` and after `period_end` ensure that
    `previous_business_day` lookups and out-of-period cutoff transactions
    always resolve to valid open FX records.
    """
    if base_currency not in CURRENCY_EXPONENTS:
        raise ValueError(f"Unsupported base_currency '{base_currency}'")

    start_date = extract_iso_date(period_start) - dt.timedelta(days=padding_days)
    end_date = extract_iso_date(period_end) + dt.timedelta(days=padding_days)
    if end_date < start_date:
        raise ValueError(f"period_end ({period_end}) cannot precede period_start ({period_start})")

    rng = random.Random(seed)

    # Deduplicate and sort foreign currencies deterministically
    foreign_currencies = sorted({c for c in currencies if c != base_currency})
    for curr in foreign_currencies:
        if curr not in CURRENCY_EXPONENTS:
            raise ValueError(f"Unsupported currency '{curr}'")

    base_anchor = _USD_PER_UNIT_ANCHOR[base_currency]

    # Initialize starting rates: how many units of `base_currency` per 1 unit of `curr`
    current_rates: dict[str, Decimal] = {}
    for curr in foreign_currencies:
        curr_anchor = _USD_PER_UNIT_ANCHOR[curr]
        raw_cross = curr_anchor / base_anchor
        # Apply small deterministic initial perturbation (-2% to +2%)
        init_jitter = Decimal(str(1.0 + rng.uniform(-0.02, 0.02)))
        current_rates[curr] = raw_cross * init_jitter

    fx_rows: list[FxRateRow] = []
    num_days = (end_date - start_date).days + 1

    for day_offset in range(num_days):
        cur_date_str = (start_date + dt.timedelta(days=day_offset)).isoformat()
        for curr in foreign_currencies:
            if day_offset > 0:
                # Daily random walk step between -0.45% and +0.45%
                step_factor = Decimal(str(1.0 + rng.uniform(-0.0045, 0.0045)))
                current_rates[curr] = current_rates[curr] * step_factor

            rate_val = current_rates[curr]
            # Use 6 decimal places if rate is small (< 0.1), otherwise 4 decimal places
            quant = Decimal("0.000001") if rate_val < Decimal("0.1") else Decimal("0.0001")
            formatted_rate = str(rate_val.quantize(quant, rounding=ROUND_HALF_EVEN))

            fx_rows.append(
                FxRateRow(
                    date=cur_date_str,
                    source_currency=curr,
                    target_currency=base_currency,
                    rate=formatted_rate,
                )
            )

    return fx_rows


def convert_amount_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate_str: str,
    rounding_mode: str = "HALF_EVEN",
) -> int:
    """Convert an integer minor-unit amount between currencies using a decimal rate string.

    `rate_str` represents major units of `target_currency` per 1 major unit of `source_currency`.
    Accounts for differing currency exponents (e.g. JPY exponent 0 vs USD exponent 2).
    """
    if amount_minor < 0:
        raise ValueError(f"amount_minor must be non-negative, got {amount_minor}")
    if source_currency not in CURRENCY_EXPONENTS:
        raise ValueError(f"Unsupported source_currency '{source_currency}'")
    if target_currency not in CURRENCY_EXPONENTS:
        raise ValueError(f"Unsupported target_currency '{target_currency}'")
    if rounding_mode not in _ROUNDING_MAP:
        raise ValueError(f"Unsupported rounding_mode '{rounding_mode}'. Allowed: {list(_ROUNDING_MAP)}")

    if source_currency == target_currency and Decimal(rate_str) == Decimal("1"):
        return int(amount_minor)

    rate = Decimal(rate_str)
    if rate <= 0:
        raise ValueError(f"FX rate must be positive, got '{rate_str}'")

    src_exp = CURRENCY_EXPONENTS[source_currency]
    tgt_exp = CURRENCY_EXPONENTS[target_currency]
    scale_factor = Decimal(10) ** (tgt_exp - src_exp)

    raw_target_minor = Decimal(amount_minor) * rate * scale_factor
    rounded = raw_target_minor.quantize(Decimal("1"), rounding=_ROUNDING_MAP[rounding_mode])
    return int(rounded)


def lookup_fx_rate(
    fx_rows: list[FxRateRow],
    source_currency: str,
    target_currency: str,
    date: str,
    policy: PolicyConfig | None = None,
    invoice_issue_date: str | None = None,
) -> FxRateRow:
    """Resolve the applicable FX rate row according to company accounting policy."""
    if source_currency == target_currency:
        effective_date = resolve_effective_fx_date(date, policy, invoice_issue_date)
        return FxRateRow(
            date=effective_date,
            source_currency=source_currency,
            target_currency=target_currency,
            rate="1.0000",
        )

    effective_date = resolve_effective_fx_date(date, policy, invoice_issue_date)

    # Filter candidate rows matching currency pair
    pair_rows = [
        r
        for r in fx_rows
        if r.source_currency == source_currency and r.target_currency == target_currency
    ]
    if not pair_rows:
        raise KeyError(f"No FX rates found for pair {source_currency}->{target_currency}")

    # 1. Exact date match
    for row in pair_rows:
        if row.date == effective_date:
            return row

    # 2. Fallback: most recent prior date available in table
    prior_rows = [r for r in pair_rows if r.date <= effective_date]
    if prior_rows:
        prior_rows.sort(key=lambda r: r.date)
        return prior_rows[-1]

    # 3. Fallback: earliest available date if effective_date precedes table start
    pair_rows.sort(key=lambda r: r.date)
    return pair_rows[0]

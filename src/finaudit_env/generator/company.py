"""Clean synthetic company history generator (`company.py`).

Creates a consistent underlying company ledger, invoices, payments, and plausible
clean distractors (split payments, refunds, recurring legitimate amounts) that
satisfy accounting identities prior to causal error injection.
"""

from __future__ import annotations

from typing import Any


def generate_clean_company(seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a clean, balanced synthetic company episode state before error injection."""
    raise NotImplementedError("generate_clean_company is not implemented yet.")

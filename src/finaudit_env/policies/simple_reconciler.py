"""Simple heuristic reconciliation baseline policy (`simple_reconciler.py`).

Paginates through open records (`invoices`, `payments`, `fx`, `ledger`) using only
the environment's read-only tools and `calculator`, and performs rule-based audit
reconciliation across all 4 error types while filtering out clean distractors.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP
from typing import Any

from finaudit_env.policies.base import BasePolicy
from finaudit_env.schemas import CURRENCY_EXPONENTS, IssueType

_ROUNDING_MAP: dict[str, str] = {
    "HALF_EVEN": ROUND_HALF_EVEN,
    "HALF_UP": ROUND_HALF_UP,
    "DOWN": ROUND_DOWN,
    "UP": ROUND_UP,
}


def _extract_iso_date(ts: str) -> dt.date:
    return dt.date.fromisoformat(ts.strip()[:10])


def _resolve_fx_date(
    tx_ts: str,
    fx_rate_date_rule: str,
    invoice_issued_at: str | None = None,
) -> str:
    if fx_rate_date_rule == "invoice_issue_date" and invoice_issued_at:
        base_date = _extract_iso_date(invoice_issued_at)
    else:
        base_date = _extract_iso_date(tx_ts)

    if fx_rate_date_rule == "previous_business_day":
        prev = base_date - dt.timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= dt.timedelta(days=1)
        return prev.isoformat()
    return base_date.isoformat()


def _lookup_fx_dict(
    fx_rows: list[dict[str, Any]],
    source_currency: str,
    target_currency: str,
    tx_ts: str,
    fx_rate_date_rule: str,
    invoice_issued_at: str | None = None,
) -> dict[str, Any] | None:
    effective_date = _resolve_fx_date(tx_ts, fx_rate_date_rule, invoice_issued_at)
    pair_rows = [
        r
        for r in fx_rows
        if r.get("source_currency") == source_currency
        and r.get("target_currency") == target_currency
    ]
    if not pair_rows:
        return None
    for row in pair_rows:
        if row.get("date") == effective_date:
            return row
    prior = [r for r in pair_rows if str(r.get("date", "")) <= effective_date]
    if prior:
        prior.sort(key=lambda r: str(r.get("date", "")))
        return prior[-1]
    pair_rows.sort(key=lambda r: str(r.get("date", "")))
    return pair_rows[0]


def _convert_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate_str: str,
    rounding_mode: str,
    exponents: dict[str, int],
) -> int:
    if source_currency == target_currency:
        return int(amount_minor)
    rate = Decimal(rate_str)
    src_exp = exponents.get(source_currency, CURRENCY_EXPONENTS.get(source_currency, 2))
    tgt_exp = exponents.get(target_currency, CURRENCY_EXPONENTS.get(target_currency, 2))
    scale = Decimal(10) ** (tgt_exp - src_exp)
    raw = Decimal(amount_minor) * rate * scale
    rounding = _ROUNDING_MAP.get(rounding_mode, ROUND_HALF_EVEN)
    return int(raw.quantize(Decimal("1"), rounding=rounding))


class SimpleReconcilerPolicy(BasePolicy):
    """Heuristic reconciliation baseline policy operating strictly via open tools."""

    name: str = "simple_reconciler"

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self._tables_to_fetch: list[str] = []
        self._current_table: str | None = None
        self._current_cursor: str | None = None
        self._did_calculator_check: bool = False
        self._collected: dict[str, list[dict[str, Any]]] = {}

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset reconciliation buffers for a new episode."""
        super().reset(observation, info)
        self._tables_to_fetch = ["invoices", "payments", "fx", "ledger"]
        self._current_table = self._tables_to_fetch.pop(0)
        self._current_cursor = None
        self._did_calculator_check = False
        self._collected = {
            "invoices": [],
            "payments": [],
            "fx": [],
            "ledger": [],
        }

    def _ingest_last_observation(self, observation: dict[str, Any]) -> None:
        result = observation.get("result")
        if not isinstance(result, dict):
            return
        table = result.get("table")
        items = result.get("items")
        if isinstance(table, str) and table in self._collected and isinstance(items, list):
            self._collected[table].extend(items)
            next_c = result.get("next_cursor")
            if next_c is not None:
                self._current_cursor = str(next_c)
            else:
                self._current_cursor = None
                self._current_table = (
                    self._tables_to_fetch.pop(0) if self._tables_to_fetch else None
                )

    def _build_reconciled_findings(self) -> list[dict[str, Any]]:
        base_currency = str(self._manifest.get("base_currency", "USD"))
        exponents = dict(self._manifest.get("currency_exponents") or CURRENCY_EXPONENTS)
        period_start = str(self._policy.get("period_start") or self._manifest.get("period_start", "2026-01-01"))[:10]
        period_end = str(self._policy.get("period_end") or self._manifest.get("period_end", "2026-12-31"))[:10]
        fx_rule = str(self._policy.get("fx_rate_date_rule", "transaction_date"))
        rounding_mode = str(self._policy.get("rounding_mode") or self._manifest.get("rounding_mode", "HALF_EVEN"))
        tolerance = int(self._policy.get("amount_tolerance_minor", 1))

        invoices = self._collected["invoices"]
        payments = self._collected["payments"]
        fx_rows = self._collected["fx"]
        ledger = self._collected["ledger"]

        inv_by_id: dict[str, dict[str, Any]] = {
            inv["invoice_id"]: inv for inv in invoices if "invoice_id" in inv
        }

        # Map invoice_id -> accrual debit rows (payment_id is None, debit_minor > 0)
        accrual_debits_by_inv: dict[str, list[dict[str, Any]]] = {}
        # Map payment_id -> settlement rows
        settlement_credits_by_pay: dict[str, list[dict[str, Any]]] = {}
        settlement_debits_by_pay: dict[str, list[dict[str, Any]]] = {}

        for row in ledger:
            inv_id = row.get("invoice_id")
            pay_id = row.get("payment_id")
            debit = int(row.get("debit_minor", 0))
            credit = int(row.get("credit_minor", 0))
            if inv_id and not pay_id and debit > 0:
                accrual_debits_by_inv.setdefault(inv_id, []).append(row)
            if pay_id:
                if credit > 0:
                    settlement_credits_by_pay.setdefault(pay_id, []).append(row)
                if debit > 0:
                    settlement_debits_by_pay.setdefault(pay_id, []).append(row)

        paid_invoice_ids: set[str] = set()
        for pay in payments:
            for ref in pay.get("invoice_refs") or []:
                paid_invoice_ids.add(ref)

        findings: list[dict[str, Any]] = []
        duplicate_copy_inv_ids: set[str] = set()

        # -------------------------------------------------------------------
        # 1. Detect DUPLICATE_INVOICE
        # -------------------------------------------------------------------
        # Case A: Two distinct invoice documents with identical entity, currency, total_minor, and line_items
        sig_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for inv in invoices:
            items_sig = tuple(
                (
                    it.get("description", ""),
                    int(it.get("quantity", 0)),
                    int(it.get("unit_price_minor", 0)),
                    int(it.get("amount_minor", 0)),
                )
                for it in (inv.get("line_items") or [])
            )
            sig = (
                inv.get("entity_id"),
                inv.get("currency"),
                int(inv.get("total_minor", 0)),
                items_sig,
            )
            sig_groups.setdefault(sig, []).append(inv)

        for group in sig_groups.values():
            if len(group) < 2:
                continue
            # Sort so that the original (paid or earlier issued_at) comes first, and the duplicate comes second
            sorted_group = sorted(
                group,
                key=lambda x: (
                    0 if x["invoice_id"] in paid_invoice_ids else 1,
                    str(x.get("issued_at", "")),
                    str(x.get("invoice_id", "")),
                ),
            )
            orig_inv = sorted_group[0]
            for dup_inv in sorted_group[1:]:
                orig_id = orig_inv["invoice_id"]
                dup_id = dup_inv["invoice_id"]
                orig_ar_list = accrual_debits_by_inv.get(orig_id, [])
                dup_ar_list = accrual_debits_by_inv.get(dup_id, [])
                if not orig_ar_list or not dup_ar_list:
                    continue
                orig_ar = orig_ar_list[0]
                dup_ar = dup_ar_list[0]
                duplicate_copy_inv_ids.add(dup_id)

                findings.append(
                    {
                        "type": IssueType.DUPLICATE_INVOICE.value,
                        "affected_ids": sorted([orig_id, dup_id, dup_ar["row_id"]]),
                        "claimed_amount_minor": int(dup_ar["debit_minor"]),
                        "currency": base_currency,
                        "evidence_ids": sorted(
                            {
                                orig_inv["doc_id"],
                                dup_inv["doc_id"],
                                orig_ar["row_id"],
                                dup_ar["row_id"],
                            }
                        ),
                        "explanation": (
                            f"Duplicate invoice {dup_id} ({dup_inv['doc_id']}) duplicates original "
                            f"{orig_id} ({orig_inv['doc_id']}) in ledger row {dup_ar['row_id']}."
                        ),
                    }
                )

        # Case B: Single invoice posted multiple times in ledger accrual debit rows
        for inv_id, ar_rows in accrual_debits_by_inv.items():
            if len(ar_rows) >= 2 and inv_id in inv_by_id:
                inv = inv_by_id[inv_id]
                orig_ar = ar_rows[0]
                for dup_ar in ar_rows[1:]:
                    findings.append(
                        {
                            "type": IssueType.DUPLICATE_INVOICE.value,
                            "affected_ids": sorted([inv_id, dup_ar["row_id"]]),
                            "claimed_amount_minor": int(dup_ar["debit_minor"]),
                            "currency": base_currency,
                            "evidence_ids": sorted({inv["doc_id"], orig_ar["row_id"], dup_ar["row_id"]}),
                            "explanation": f"Invoice {inv_id} posted twice in ledger rows {orig_ar['row_id']} and {dup_ar['row_id']}.",
                        }
                    )

        # -------------------------------------------------------------------
        # 2. Detect PERIOD_CUTOFF_ERROR
        # -------------------------------------------------------------------
        for inv in invoices:
            inv_id = inv["invoice_id"]
            issued_date = str(inv.get("issued_at", ""))[:10]
            if issued_date < period_start or issued_date > period_end:
                ar_rows = accrual_debits_by_inv.get(inv_id, [])
                if not ar_rows:
                    continue
                ar_row = ar_rows[0]
                findings.append(
                    {
                        "type": IssueType.PERIOD_CUTOFF_ERROR.value,
                        "affected_ids": sorted([inv_id, ar_row["row_id"]]),
                        "claimed_amount_minor": int(ar_row["debit_minor"]),
                        "currency": base_currency,
                        "evidence_ids": sorted({inv["doc_id"], inv_id, ar_row["row_id"]}),
                        "explanation": (
                            f"Invoice {inv_id} issued at {inv.get('issued_at')} falls outside "
                            f"reporting period [{period_start}, {period_end}]."
                        ),
                    }
                )

        # -------------------------------------------------------------------
        # 3. Detect FX_CONVERSION_ERROR
        # -------------------------------------------------------------------
        for inv in invoices:
            inv_id = inv["invoice_id"]
            if inv_id in duplicate_copy_inv_ids:
                continue
            curr = str(inv.get("currency", base_currency))
            if curr == base_currency:
                continue
            ar_rows = accrual_debits_by_inv.get(inv_id, [])
            if not ar_rows:
                continue
            ar_row = ar_rows[0]
            fx_row = _lookup_fx_dict(
                fx_rows=fx_rows,
                source_currency=curr,
                target_currency=base_currency,
                tx_ts=str(inv.get("issued_at", "")),
                fx_rate_date_rule=fx_rule,
                invoice_issued_at=str(inv.get("issued_at", "")),
            )
            if fx_row is None or not fx_row.get("fx_id"):
                continue

            expected_base = _convert_minor(
                amount_minor=int(inv["total_minor"]),
                source_currency=curr,
                target_currency=base_currency,
                rate_str=str(fx_row["rate"]),
                rounding_mode=rounding_mode,
                exponents=exponents,
            )
            recorded_base = int(ar_row["debit_minor"])
            diff = abs(recorded_base - expected_base)
            if diff > tolerance:
                fx_id = str(fx_row["fx_id"])
                findings.append(
                    {
                        "type": IssueType.FX_CONVERSION_ERROR.value,
                        "affected_ids": sorted([inv_id, ar_row["row_id"], fx_id]),
                        "claimed_amount_minor": diff,
                        "currency": base_currency,
                        "evidence_ids": sorted({inv["doc_id"], inv_id, ar_row["row_id"], fx_id}),
                        "explanation": (
                            f"Foreign currency invoice {inv_id} ({curr}) converted with FX discrepancy "
                            f"of {diff} {base_currency} in ledger row {ar_row['row_id']} vs {fx_id}."
                        ),
                    }
                )

        # -------------------------------------------------------------------
        # 4. Detect UNALLOCATED_OR_PARTIAL_PAYMENT
        # -------------------------------------------------------------------
        for pay in payments:
            pay_id = pay["payment_id"]
            refs = pay.get("invoice_refs") or []
            if not refs:
                continue
            inv_id = refs[0]
            inv = inv_by_id.get(inv_id)
            if inv is None:
                continue

            cred_rows = settlement_credits_by_pay.get(pay_id, [])
            if not cred_rows:
                continue
            ar_row = cred_rows[0]
            deb_rows = settlement_debits_by_pay.get(pay_id, [])
            cash_row_id = deb_rows[0]["row_id"] if deb_rows else ar_row["row_id"]

            pay_curr = str(pay.get("currency", base_currency))
            pay_amount = int(pay.get("amount_minor", 0))
            if pay_curr == base_currency:
                expected_pay_base = pay_amount
            else:
                fx_row = _lookup_fx_dict(
                    fx_rows=fx_rows,
                    source_currency=pay_curr,
                    target_currency=base_currency,
                    tx_ts=str(pay.get("timestamp", "")),
                    fx_rate_date_rule=fx_rule,
                    invoice_issued_at=str(inv.get("issued_at", "")),
                )
                if fx_row is None:
                    continue
                expected_pay_base = _convert_minor(
                    amount_minor=pay_amount,
                    source_currency=pay_curr,
                    target_currency=base_currency,
                    rate_str=str(fx_row["rate"]),
                    rounding_mode=rounding_mode,
                    exponents=exponents,
                )

            recorded_credit = 0 if not ar_row.get("invoice_id") else int(ar_row["credit_minor"])
            diff = abs(expected_pay_base - recorded_credit)
            if diff > tolerance:
                findings.append(
                    {
                        "type": IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT.value,
                        "affected_ids": sorted([inv_id, pay_id, ar_row["row_id"]]),
                        "claimed_amount_minor": diff,
                        "currency": base_currency,
                        "evidence_ids": sorted(
                            {inv["doc_id"], pay["doc_id"], cash_row_id, ar_row["row_id"]}
                        ),
                        "explanation": (
                            f"Payment {pay_id} ({pay['doc_id']}) for invoice {inv_id} only allocated "
                            f"{recorded_credit} instead of {expected_pay_base} in ledger row {ar_row['row_id']}."
                        ),
                    }
                )

        return findings[:30]

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Execute next pagination/calculator step or submit reconciled findings."""
        self._ingest_last_observation(observation)
        steps_remaining = int(observation.get("steps_remaining", 100))

        # Keep paginating while we have tables left and sufficient step budget
        if self._current_table is not None and steps_remaining > 3:
            return {
                "action": "list_records",
                "table": self._current_table,
                "cursor": self._current_cursor,
                "limit": 50,
            }

        # Execute one safe calculator step to verify arithmetic tool integration
        if not self._did_calculator_check and steps_remaining > 2:
            self._did_calculator_check = True
            total_inv = sum(int(i.get("total_minor", 0)) for i in self._collected["invoices"][:5])
            return {
                "action": "calculator",
                "expression": f"{total_inv} * 1",
            }

        findings = self._build_reconciled_findings()
        return {
            "action": "submit",
            "submission": {
                "episode_id": self._episode_id,
                "findings": findings,
            },
        }


__all__ = ["SimpleReconcilerPolicy"]

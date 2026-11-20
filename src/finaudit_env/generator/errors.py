"""Causal financial error injection engine (`errors.py`).

Injects the 4 canonical `IssueType` errors (`duplicate_invoice`,
`unallocated_or_partial_payment`, `fx_conversion_error`, `period_cutoff_error`)
as well as interacting multi-error scenarios, recording `TruthIssue` and
`ProvenanceEvent` entries.
"""

from __future__ import annotations

import copy
import datetime as dt
import random
from typing import Any

from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    extract_iso_date,
    lookup_fx_rate,
)
from finaudit_env.schemas import (
    CURRENCY_EXPONENTS,
    InvoiceDoc,
    IssueType,
    LedgerRow,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ProvenanceEvent,
    TruthIssue,
)


def _collect_protected_ids(episode: dict[str, Any]) -> set[str]:
    """Return all IDs involved in clean distractors so error injection leaves distractors intact."""
    protected: set[str] = set()
    for evt in episode["provenance"].events:
        if evt.stage == "distractor_injection":
            protected.update(evt.affected_ids)
    return protected


def _find_invoice_accrual_rows(ledger: list[LedgerRow], invoice_id: str) -> tuple[LedgerRow, LedgerRow] | None:
    """Find the (debit_AR_row, credit_Revenue_row) pair for a given invoice accrual."""
    accrual_rows = [r for r in ledger if r.invoice_id == invoice_id and r.payment_id is None]
    debit_rows = [r for r in accrual_rows if r.debit_minor > 0]
    credit_rows = [r for r in accrual_rows if r.credit_minor > 0]
    if not debit_rows or not credit_rows:
        return None
    return debit_rows[0], credit_rows[0]


def _find_payment_settlement_rows(ledger: list[LedgerRow], payment_id: str) -> tuple[LedgerRow, LedgerRow] | None:
    """Find the (debit_Cash_row, credit_AR_row) pair for a given payment settlement."""
    pay_rows = [r for r in ledger if r.payment_id == payment_id]
    debit_rows = [r for r in pay_rows if r.debit_minor > 0]
    credit_rows = [r for r in pay_rows if r.credit_minor > 0]
    if not debit_rows or not credit_rows:
        return None
    return debit_rows[0], credit_rows[0]


def inject_errors(
    clean_episode: dict[str, Any],
    seed: int,
    error_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject configured causal errors into a clean company episode and build truth/provenance.

    Supported `error_config` keys:
    - `clean_only` (bool): if True, injects 0 errors and returns a clean company copy.
    - `duplicate_invoice` (int): count of `IssueType.DUPLICATE_INVOICE` errors to inject.
    - `unallocated_or_partial_payment` (int): count of `IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT`.
    - `fx_conversion_error` (int): count of `IssueType.FX_CONVERSION_ERROR`.
    - `period_cutoff_error` (int): count of `IssueType.PERIOD_CUTOFF_ERROR`.
    - `interacting_scenarios` (list[str]): list of compound scenarios to inject:
        * `"partial_fx_cutoff"`: foreign-currency invoice near period boundary with both
          an FX conversion error and a partial payment allocation discrepancy.
        * `"duplicate_fx_cutoff"`: foreign-currency invoice duplicated across the period
          cutoff boundary.
    """
    ep = copy.deepcopy(clean_episode)
    cfg = dict(error_config or {})
    rng = random.Random(seed)

    if cfg.get("clean_only", False):
        return ep

    n_dup = int(cfg.get("duplicate_invoice", 1))
    n_pay = int(cfg.get("unallocated_or_partial_payment", 1))
    n_fx = int(cfg.get("fx_conversion_error", 1))
    n_cutoff = int(cfg.get("period_cutoff_error", 1))
    interacting_scenarios: list[str] = list(cfg.get("interacting_scenarios", []))

    manifest: Manifest = ep["manifest"]
    policy: PolicyConfig = ep["policy"]
    ledger: list[LedgerRow] = ep["ledger"]
    invoices: list[InvoiceDoc] = ep["invoices"]
    payments: list[PaymentDoc] = ep["payments"]
    fx_rows = ep["fx"]
    base_currency = manifest.base_currency

    protected_ids = _collect_protected_ids(ep)
    used_invoice_ids: set[str] = set()

    # Determine next sequential numbers for newly injected rows/docs
    inv_seq = len(invoices) + 1
    led_seq = len(ledger) + 1
    issue_seq = len(ep["truth"].issues) + 1
    evt_seq = len(ep["provenance"].events) + 1

    start_date = extract_iso_date(manifest.period_start)
    end_date = extract_iso_date(manifest.period_end)
    year_str = start_date.strftime("%Y")

    def next_issue_id() -> str:
        nonlocal issue_seq
        iid = f"ISS-{issue_seq:03d}"
        issue_seq += 1
        return iid

    def next_evt_id() -> str:
        nonlocal evt_seq
        eid = f"EVT-{evt_seq:03d}"
        evt_seq += 1
        return eid

    def next_row_id() -> str:
        nonlocal led_seq
        rid = f"L-{led_seq:04d}"
        led_seq += 1
        return rid

    def next_dup_inv_ids() -> tuple[str, str]:
        nonlocal inv_seq
        doc_id = f"DOC-INV-{inv_seq:03d}"
        inv_id = f"INV-{year_str}-{inv_seq:03d}"
        inv_seq += 1
        return doc_id, inv_id

    def candidate_invoices(require_foreign: bool = False) -> list[InvoiceDoc]:
        cands: list[InvoiceDoc] = []
        for inv in invoices:
            if inv.invoice_id in used_invoice_ids or inv.invoice_id in protected_ids or inv.doc_id in protected_ids:
                continue
            if require_foreign and inv.currency == base_currency:
                continue
            if _find_invoice_accrual_rows(ledger, inv.invoice_id) is None:
                continue
            cands.append(inv)
        return cands

    # -----------------------------------------------------------------------
    # Helper 1: Inject DUPLICATE_INVOICE
    # -----------------------------------------------------------------------
    def _inject_duplicate_invoice(target_inv: InvoiceDoc) -> tuple[TruthIssue, InvoiceDoc, LedgerRow, LedgerRow]:
        used_invoice_ids.add(target_inv.invoice_id)
        accrual_pair = _find_invoice_accrual_rows(ledger, target_inv.invoice_id)
        assert accrual_pair is not None
        orig_ar_row, orig_rev_row = accrual_pair

        dup_doc_id, dup_inv_id = next_dup_inv_ids()
        # Duplicate issued 1 day later (or same day if at period end)
        orig_date = extract_iso_date(target_inv.issued_at)
        dup_date = min(end_date, orig_date + dt.timedelta(days=1))
        tz_suffix = target_inv.issued_at[19:] or "+00:00"
        dup_issued_at = f"{dup_date.isoformat()}T16:45:00{tz_suffix}"

        dup_inv = InvoiceDoc(
            doc_id=dup_doc_id,
            invoice_id=dup_inv_id,
            entity_id=target_inv.entity_id,
            issued_at=dup_issued_at,
            due_at=target_inv.due_at,
            currency=target_inv.currency,
            total_minor=target_inv.total_minor,
            line_items=copy.deepcopy(target_inv.line_items),
        )
        invoices.append(dup_inv)

        dup_ar_id = next_row_id()
        dup_rev_id = next_row_id()
        discrepancy_base_minor = orig_ar_row.debit_minor

        dup_ar_row = LedgerRow(
            row_id=dup_ar_id,
            timestamp=dup_issued_at,
            account=orig_ar_row.account,
            debit_minor=discrepancy_base_minor,
            credit_minor=0,
            currency=base_currency,
            entity_id=target_inv.entity_id,
            invoice_id=dup_inv_id,
            payment_id=None,
        )
        dup_rev_row = LedgerRow(
            row_id=dup_rev_id,
            timestamp=dup_issued_at,
            account=orig_rev_row.account,
            debit_minor=0,
            credit_minor=discrepancy_base_minor,
            currency=base_currency,
            entity_id=target_inv.entity_id,
            invoice_id=dup_inv_id,
            payment_id=None,
        )
        ledger.extend([dup_ar_row, dup_rev_row])

        iss_id = next_issue_id()
        affected = sorted([target_inv.invoice_id, dup_inv_id, dup_ar_id])
        valid_ev = [
            sorted([target_inv.doc_id, dup_doc_id, dup_ar_id]),
            sorted([target_inv.invoice_id, dup_inv_id, dup_ar_id]),
            sorted([target_inv.doc_id, dup_doc_id, orig_ar_row.row_id, dup_ar_id]),
        ]
        issue = TruthIssue(
            issue_id=iss_id,
            type=IssueType.DUPLICATE_INVOICE,
            affected_ids=affected,
            expected_amount_minor=discrepancy_base_minor,
            currency=base_currency,
            valid_evidence_sets=valid_ev,
            tolerance_minor=policy.amount_tolerance_minor,
            notes=f"Duplicate invoice {dup_inv_id} ({dup_doc_id}) duplicating {target_inv.invoice_id} ({target_inv.doc_id}).",
        )
        ep["truth"].issues.append(issue)
        ep["provenance"].events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="error_injection",
                issue_id=iss_id,
                issue_type=IssueType.DUPLICATE_INVOICE,
                description=f"Injected duplicate invoice {dup_inv_id} and ledger accrual {dup_ar_id} for original {target_inv.invoice_id}.",
                before_state={"original_invoice_id": target_inv.invoice_id, "original_row_id": orig_ar_row.row_id},
                after_state={
                    "duplicate_doc_id": dup_doc_id,
                    "duplicate_invoice_id": dup_inv_id,
                    "duplicate_row_id": dup_ar_id,
                    "discrepancy_base_minor": discrepancy_base_minor,
                },
                affected_ids=affected,
            )
        )
        return issue, dup_inv, dup_ar_row, dup_rev_row

    # -----------------------------------------------------------------------
    # Helper 2: Inject UNALLOCATED_OR_PARTIAL_PAYMENT
    # -----------------------------------------------------------------------
    def _inject_payment_error(target_inv: InvoiceDoc) -> TruthIssue | None:
        linked_payments = [p for p in payments if target_inv.invoice_id in p.invoice_refs]
        if not linked_payments:
            return None
        pay_doc = linked_payments[0]
        pay_rows = _find_payment_settlement_rows(ledger, pay_doc.payment_id)
        if pay_rows is None:
            return None
        cash_row, ar_row = pay_rows
        used_invoice_ids.add(target_inv.invoice_id)

        old_recorded_minor = ar_row.credit_minor
        # Record only ~60% of the payment in the ledger, leaving 40% unallocated
        new_recorded_minor = max(1, (old_recorded_minor * 60) // 100)
        if new_recorded_minor == old_recorded_minor:
            new_recorded_minor = max(0, old_recorded_minor - 100)
        discrepancy_minor = abs(old_recorded_minor - new_recorded_minor)

        cash_row.debit_minor = new_recorded_minor
        ar_row.credit_minor = new_recorded_minor

        iss_id = next_issue_id()
        affected = sorted([target_inv.invoice_id, pay_doc.payment_id, ar_row.row_id])
        valid_ev = [
            sorted([target_inv.doc_id, pay_doc.doc_id, ar_row.row_id]),
            sorted([target_inv.invoice_id, pay_doc.payment_id, ar_row.row_id]),
            sorted([pay_doc.doc_id, cash_row.row_id, ar_row.row_id]),
        ]
        issue = TruthIssue(
            issue_id=iss_id,
            type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
            affected_ids=affected,
            expected_amount_minor=discrepancy_minor,
            currency=base_currency,
            valid_evidence_sets=valid_ev,
            tolerance_minor=policy.amount_tolerance_minor,
            notes=(
                f"Payment {pay_doc.payment_id} for {target_inv.invoice_id} partially allocated in ledger row "
                f"{ar_row.row_id}: expected {old_recorded_minor}, recorded {new_recorded_minor}."
            ),
        )
        ep["truth"].issues.append(issue)
        ep["provenance"].events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="error_injection",
                issue_id=iss_id,
                issue_type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
                description=f"Reduced ledger settlement amount on {ar_row.row_id} for payment {pay_doc.payment_id}.",
                before_state={"row_id": ar_row.row_id, "credit_minor": old_recorded_minor},
                after_state={
                    "row_id": ar_row.row_id,
                    "credit_minor": new_recorded_minor,
                    "discrepancy_minor": discrepancy_minor,
                },
                affected_ids=affected,
            )
        )
        return issue

    # -----------------------------------------------------------------------
    # Helper 3: Inject FX_CONVERSION_ERROR
    # -----------------------------------------------------------------------
    def _inject_fx_error(target_inv: InvoiceDoc, keep_available: bool = False) -> TruthIssue:
        if not keep_available:
            used_invoice_ids.add(target_inv.invoice_id)

        # Ensure target invoice is in a foreign currency
        if target_inv.currency == base_currency:
            foreign_options = sorted({r.source_currency for r in fx_rows if r.source_currency != base_currency})
            new_curr = foreign_options[0] if foreign_options else "EUR"
            target_inv.currency = new_curr

        accrual_pair = _find_invoice_accrual_rows(ledger, target_inv.invoice_id)
        assert accrual_pair is not None
        ar_row, rev_row = accrual_pair

        correct_fx_row = lookup_fx_rate(
            fx_rows=fx_rows,
            source_currency=target_inv.currency,
            target_currency=base_currency,
            date=target_inv.issued_at,
            policy=policy,
            invoice_issue_date=target_inv.issued_at,
        )
        correct_base_minor = convert_amount_minor(
            amount_minor=target_inv.total_minor,
            source_currency=target_inv.currency,
            target_currency=base_currency,
            rate_str=correct_fx_row.rate,
            rounding_mode=policy.rounding_mode,
        )

        # Apply an erroneous FX conversion (+15% rate distortion)
        wrong_base_minor = int((correct_base_minor * 115) // 100)
        if wrong_base_minor == correct_base_minor:
            wrong_base_minor += 500
        discrepancy_minor = abs(wrong_base_minor - correct_base_minor)

        before_minor = ar_row.debit_minor
        ar_row.debit_minor = wrong_base_minor
        rev_row.credit_minor = wrong_base_minor

        iss_id = next_issue_id()
        assert correct_fx_row.fx_id is not None
        affected = sorted([target_inv.invoice_id, ar_row.row_id, correct_fx_row.fx_id])
        valid_ev = [
            sorted([target_inv.doc_id, ar_row.row_id, correct_fx_row.fx_id]),
            sorted([target_inv.invoice_id, ar_row.row_id, correct_fx_row.fx_id]),
            sorted([target_inv.doc_id, ar_row.row_id]),
        ]
        issue = TruthIssue(
            issue_id=iss_id,
            type=IssueType.FX_CONVERSION_ERROR,
            affected_ids=affected,
            expected_amount_minor=discrepancy_minor,
            currency=base_currency,
            valid_evidence_sets=valid_ev,
            tolerance_minor=policy.amount_tolerance_minor,
            notes=(
                f"Invoice {target_inv.invoice_id} in {target_inv.currency} converted at wrong FX rate in {ar_row.row_id}: "
                f"expected {correct_base_minor} (using {correct_fx_row.fx_id} rate {correct_fx_row.rate}), "
                f"recorded {wrong_base_minor}."
            ),
        )
        ep["truth"].issues.append(issue)
        ep["provenance"].events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="error_injection",
                issue_id=iss_id,
                issue_type=IssueType.FX_CONVERSION_ERROR,
                description=f"Applied wrong FX rate on ledger row {ar_row.row_id} for foreign invoice {target_inv.invoice_id}.",
                before_state={"row_id": ar_row.row_id, "debit_minor": before_minor, "fx_id": correct_fx_row.fx_id},
                after_state={
                    "row_id": ar_row.row_id,
                    "debit_minor": wrong_base_minor,
                    "expected_base_minor": correct_base_minor,
                    "discrepancy_minor": discrepancy_minor,
                },
                affected_ids=affected,
            )
        )
        return issue

    # -----------------------------------------------------------------------
    # Helper 4: Inject PERIOD_CUTOFF_ERROR
    # -----------------------------------------------------------------------
    def _inject_cutoff_error(target_inv: InvoiceDoc, ar_row: LedgerRow, rev_row: LedgerRow) -> TruthIssue:
        used_invoice_ids.add(target_inv.invoice_id)
        old_issued_at = target_inv.issued_at
        tz_suffix = old_issued_at[19:] or "+00:00"

        # Shift date 3 days past period_end (into next reporting period) while keeping it in current ledger
        out_of_period_date = end_date + dt.timedelta(days=3)
        new_ts = f"{out_of_period_date.isoformat()}T11:20:00{tz_suffix}"

        target_inv.issued_at = new_ts
        ar_row.timestamp = new_ts
        rev_row.timestamp = new_ts

        discrepancy_minor = ar_row.debit_minor
        iss_id = next_issue_id()
        affected = sorted([target_inv.invoice_id, ar_row.row_id])
        valid_ev = [
            sorted([target_inv.doc_id, ar_row.row_id]),
            sorted([target_inv.invoice_id, ar_row.row_id]),
        ]
        issue = TruthIssue(
            issue_id=iss_id,
            type=IssueType.PERIOD_CUTOFF_ERROR,
            affected_ids=affected,
            expected_amount_minor=discrepancy_minor,
            currency=base_currency,
            valid_evidence_sets=valid_ev,
            tolerance_minor=policy.amount_tolerance_minor,
            notes=(
                f"Invoice {target_inv.invoice_id} issued at {new_ts} outside reporting period "
                f"[{manifest.period_start}, {manifest.period_end}] but recorded in ledger row {ar_row.row_id}."
            ),
        )
        ep["truth"].issues.append(issue)
        ep["provenance"].events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="error_injection",
                issue_id=iss_id,
                issue_type=IssueType.PERIOD_CUTOFF_ERROR,
                description=f"Moved invoice {target_inv.invoice_id} and ledger row {ar_row.row_id} outside period end to {new_ts}.",
                before_state={"invoice_id": target_inv.invoice_id, "issued_at": old_issued_at},
                after_state={
                    "invoice_id": target_inv.invoice_id,
                    "issued_at": new_ts,
                    "row_id": ar_row.row_id,
                    "discrepancy_minor": discrepancy_minor,
                },
                affected_ids=affected,
            )
        )
        return issue

    # -----------------------------------------------------------------------
    # 1. Inject Interacting Scenarios First
    # -----------------------------------------------------------------------
    for scenario in interacting_scenarios:
        if scenario == "partial_fx_cutoff":
            # Foreign currency transaction near period boundary with both FX conversion error AND partial payment
            cands = candidate_invoices(require_foreign=True) or candidate_invoices(require_foreign=False)
            if cands:
                target = rng.choice(cands)
                # Move near period boundary (last day of period)
                tz_suffix = target.issued_at[19:] or "+00:00"
                boundary_ts = f"{end_date.isoformat()}T22:30:00{tz_suffix}"
                target.issued_at = boundary_ts
                accrual_pair = _find_invoice_accrual_rows(ledger, target.invoice_id)
                if accrual_pair is not None:
                    accrual_pair[0].timestamp = boundary_ts
                    accrual_pair[1].timestamp = boundary_ts
                _inject_fx_error(target, keep_available=True)
                _inject_payment_error(target)

        elif scenario == "duplicate_fx_cutoff":
            # Foreign currency invoice duplicated across the period boundary (Duplicate + Cutoff)
            cands = candidate_invoices(require_foreign=True) or candidate_invoices(require_foreign=False)
            if cands:
                target = rng.choice(cands)
                _, dup_inv, dup_ar, dup_rev = _inject_duplicate_invoice(target)
                _inject_cutoff_error(dup_inv, dup_ar, dup_rev)

    # -----------------------------------------------------------------------
    # 2. Inject Individual Standalone Error Types
    # -----------------------------------------------------------------------
    for _ in range(n_dup):
        cands = candidate_invoices(require_foreign=False)
        if cands:
            _inject_duplicate_invoice(rng.choice(cands))

    for _ in range(n_pay):
        cands = candidate_invoices(require_foreign=False)
        if cands:
            _inject_payment_error(rng.choice(cands))

    for _ in range(n_fx):
        cands = candidate_invoices(require_foreign=True) or candidate_invoices(require_foreign=False)
        if cands:
            _inject_fx_error(rng.choice(cands))

    for _ in range(n_cutoff):
        cands = candidate_invoices(require_foreign=False)
        if cands:
            target = rng.choice(cands)
            pair = _find_invoice_accrual_rows(ledger, target.invoice_id)
            if pair is not None:
                _inject_cutoff_error(target, pair[0], pair[1])

    # Update manifest counts after mutations (e.g. duplicate invoice adds 1 invoice and 2 ledger rows)
    manifest.ledger_rows_count = len(ledger)
    manifest.invoices_count = len(invoices)
    manifest.payments_count = len(payments)
    manifest.fx_rates_count = len(fx_rows)

    return ep

"""Independent episode and ground-truth verifier (`verifier.py`).

Verifies pre-mutation accounting identities, post-mutation discrepancy accuracy,
and independent sufficiency of `valid_evidence_sets` in `truth.json` using only
open records.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    extract_iso_date,
    lookup_fx_rate,
)
from finaudit_env.schemas import (
    FxRateRow,
    InvoiceDoc,
    IssueType,
    LedgerRow,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ProvenanceFile,
    TruthFile,
)


def verify_episode(episode_data: dict[str, Any]) -> dict[str, Any]:
    """Verify accounting invariants and evidence sufficiency for a generated episode.

    Returns a diagnostic dictionary:
    ```python
    {
        "valid": bool,
        "errors": list[str],
        "checks": {
            "manifest_counts_match": bool,
            "invoice_line_items_sum_ok": bool,
            "pre_mutation_trial_balance_ok": bool,
            "no_dangling_open_references": bool,
            "truth_ids_exist_in_open_records": bool,
            "independent_evidence_sufficiency": bool,
        },
        "issues_verified": int,
    }
    ```
    """
    errors: list[str] = []

    manifest: Manifest = episode_data["manifest"]
    policy: PolicyConfig = episode_data["policy"]
    ledger: list[LedgerRow] = episode_data["ledger"]
    invoices: list[InvoiceDoc] = episode_data["invoices"]
    payments: list[PaymentDoc] = episode_data["payments"]
    fx_rows: list[FxRateRow] = episode_data["fx"]
    truth: TruthFile = episode_data["truth"]
    provenance: ProvenanceFile = episode_data["provenance"]

    # 1. Manifest counts check
    manifest_counts_match = True
    if manifest.ledger_rows_count != len(ledger):
        manifest_counts_match = False
        errors.append(f"Manifest ledger_rows_count ({manifest.ledger_rows_count}) != len(ledger) ({len(ledger)})")
    if manifest.invoices_count != len(invoices):
        manifest_counts_match = False
        errors.append(f"Manifest invoices_count ({manifest.invoices_count}) != len(invoices) ({len(invoices)})")
    if manifest.payments_count != len(payments):
        manifest_counts_match = False
        errors.append(f"Manifest payments_count ({manifest.payments_count}) != len(payments) ({len(payments)})")
    if manifest.fx_rates_count != len(fx_rows):
        manifest_counts_match = False
        errors.append(f"Manifest fx_rates_count ({manifest.fx_rates_count}) != len(fx) ({len(fx_rows)})")

    # 2. Invoice line items math check
    invoice_line_items_sum_ok = True
    for inv in invoices:
        calc_total = sum(item.amount_minor for item in inv.line_items)
        if calc_total != inv.total_minor:
            invoice_line_items_sum_ok = False
            errors.append(f"Invoice {inv.invoice_id} total_minor ({inv.total_minor}) != sum(line_items) ({calc_total})")
        for item in inv.line_items:
            if item.quantity * item.unit_price_minor != item.amount_minor:
                invoice_line_items_sum_ok = False
                errors.append(f"Invoice {inv.invoice_id} line item math mismatch: {item}")

    # 3. Trial balance check
    total_debit = sum(r.debit_minor for r in ledger)
    total_credit = sum(r.credit_minor for r in ledger)
    pre_mutation_trial_balance_ok = provenance.pre_mutation_trial_balance_ok and (total_debit == total_credit)
    if not pre_mutation_trial_balance_ok:
        errors.append(
            f"Trial balance mismatch: total_debit={total_debit}, total_credit={total_credit}, "
            f"provenance.pre_mutation_trial_balance_ok={provenance.pre_mutation_trial_balance_ok}"
        )

    # Build lookup maps for open records
    ledger_by_id: dict[str, LedgerRow] = {r.row_id: r for r in ledger}
    inv_by_id: dict[str, InvoiceDoc] = {i.invoice_id: i for i in invoices}
    inv_by_doc: dict[str, InvoiceDoc] = {i.doc_id: i for i in invoices}
    pay_by_id: dict[str, PaymentDoc] = {p.payment_id: p for p in payments}
    pay_by_doc: dict[str, PaymentDoc] = {p.doc_id: p for p in payments}
    fx_by_id: dict[str, FxRateRow] = {f.fx_id: f for f in fx_rows if f.fx_id is not None}

    all_open_ids: set[str] = (
        set(ledger_by_id.keys())
        | set(inv_by_id.keys())
        | set(inv_by_doc.keys())
        | set(pay_by_id.keys())
        | set(pay_by_doc.keys())
        | set(fx_by_id.keys())
    )

    # 4. No dangling references in open files
    no_dangling_open_references = True
    for row in ledger:
        if row.invoice_id is not None and row.invoice_id not in inv_by_id:
            no_dangling_open_references = False
            errors.append(f"Ledger row {row.row_id} references unknown invoice_id '{row.invoice_id}'")
        if row.payment_id is not None and row.payment_id not in pay_by_id:
            no_dangling_open_references = False
            errors.append(f"Ledger row {row.row_id} references unknown payment_id '{row.payment_id}'")
    for pay in payments:
        for ref in pay.invoice_refs:
            if ref not in inv_by_id:
                no_dangling_open_references = False
                errors.append(f"Payment {pay.payment_id} references unknown invoice_id '{ref}'")

    # 5. Truth IDs exist in open records
    truth_ids_exist_in_open_records = True
    for iss in truth.issues:
        for aid in iss.affected_ids:
            if aid not in all_open_ids:
                truth_ids_exist_in_open_records = False
                errors.append(f"Issue {iss.issue_id} references non-existent affected_id '{aid}'")
        for ev_set in iss.valid_evidence_sets:
            if not ev_set:
                truth_ids_exist_in_open_records = False
                errors.append(f"Issue {iss.issue_id} has an empty evidence set")
            for eid in ev_set:
                if eid not in all_open_ids:
                    truth_ids_exist_in_open_records = False
                    errors.append(f"Issue {iss.issue_id} references non-existent evidence_id '{eid}'")

    # 6. Independent verification of each TruthIssue discrepancy and evidence sufficiency
    independent_evidence_sufficiency = True
    period_start_date = extract_iso_date(manifest.period_start)
    period_end_date = extract_iso_date(manifest.period_end)

    for iss in truth.issues:
        iss_tol = iss.tolerance_minor if iss.tolerance_minor is not None else policy.amount_tolerance_minor
        if iss.expected_amount_minor <= 0:
            independent_evidence_sufficiency = False
            errors.append(f"Issue {iss.issue_id} has non-positive expected_amount_minor={iss.expected_amount_minor}")

        # Gather all referenced objects across affected_ids and valid_evidence_sets
        all_ref_ids = set(iss.affected_ids)
        for ev_set in iss.valid_evidence_sets:
            all_ref_ids.update(ev_set)

        ref_rows = [ledger_by_id[rid] for rid in all_ref_ids if rid in ledger_by_id]
        ref_invs_map: dict[str, InvoiceDoc] = {}
        for rid in all_ref_ids:
            if rid in inv_by_id:
                ref_invs_map[inv_by_id[rid].invoice_id] = inv_by_id[rid]
            elif rid in inv_by_doc:
                ref_invs_map[inv_by_doc[rid].invoice_id] = inv_by_doc[rid]
        ref_invs = list(ref_invs_map.values())

        ref_pays_map: dict[str, PaymentDoc] = {}
        for rid in all_ref_ids:
            if rid in pay_by_id:
                ref_pays_map[pay_by_id[rid].payment_id] = pay_by_id[rid]
            elif rid in pay_by_doc:
                ref_pays_map[pay_by_doc[rid].payment_id] = pay_by_doc[rid]
        ref_pays = list(ref_pays_map.values())

        if not ref_rows:
            independent_evidence_sufficiency = False
            errors.append(f"Issue {iss.issue_id} does not reference any ledger row in affected/evidence IDs")
            continue

        if iss.type == IssueType.DUPLICATE_INVOICE:
            if len(ref_invs) < 2:
                independent_evidence_sufficiency = False
                errors.append(f"Duplicate invoice issue {iss.issue_id} must reference both original and duplicate invoices")
            else:
                inv_a, inv_b = ref_invs[0], ref_invs[1]
                if inv_a.entity_id != inv_b.entity_id or inv_a.total_minor != inv_b.total_minor:
                    independent_evidence_sufficiency = False
                    errors.append(f"Duplicate invoice issue {iss.issue_id} invoices do not match entity/amount")
            if not any(abs(r.debit_minor - iss.expected_amount_minor) <= iss_tol for r in ref_rows):
                independent_evidence_sufficiency = False
                errors.append(f"Duplicate invoice issue {iss.issue_id} ledger amount does not match expected_amount_minor")

        elif iss.type == IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT:
            if not ref_pays:
                independent_evidence_sufficiency = False
                errors.append(f"Payment error issue {iss.issue_id} does not reference a PaymentDoc")
            else:
                pay_doc = ref_pays[0]
                inv_ts = ref_invs[0].issued_at if ref_invs else pay_doc.timestamp
                if pay_doc.currency == manifest.base_currency:
                    expected_pay_base = pay_doc.amount_minor
                else:
                    fx_r = lookup_fx_rate(
                        fx_rows,
                        pay_doc.currency,
                        manifest.base_currency,
                        pay_doc.timestamp,
                        policy=policy,
                        invoice_issue_date=inv_ts,
                    )
                    expected_pay_base = convert_amount_minor(
                        pay_doc.amount_minor,
                        pay_doc.currency,
                        manifest.base_currency,
                        fx_r.rate,
                        policy.rounding_mode,
                    )
                allocated_amounts = [
                    0 if r.invoice_id is None else max(r.debit_minor, r.credit_minor)
                    for r in ref_rows
                ]
                if not any(abs((expected_pay_base - rec) - iss.expected_amount_minor) <= iss_tol for rec in allocated_amounts):
                    independent_evidence_sufficiency = False
                    errors.append(
                        f"Payment issue {iss.issue_id} discrepancy mismatch: expected_pay_base={expected_pay_base}, "
                        f"allocated={allocated_amounts}, truth={iss.expected_amount_minor}"
                    )

        elif iss.type == IssueType.FX_CONVERSION_ERROR:
            if not ref_invs:
                independent_evidence_sufficiency = False
                errors.append(f"FX error issue {iss.issue_id} does not reference an InvoiceDoc")
            else:
                inv_doc = ref_invs[0]
                fx_r = lookup_fx_rate(
                    fx_rows,
                    inv_doc.currency,
                    manifest.base_currency,
                    inv_doc.issued_at,
                    policy=policy,
                    invoice_issue_date=inv_doc.issued_at,
                )
                correct_base = convert_amount_minor(
                    inv_doc.total_minor,
                    inv_doc.currency,
                    manifest.base_currency,
                    fx_r.rate,
                    policy.rounding_mode,
                )
                recorded_amounts = [max(r.debit_minor, r.credit_minor) for r in ref_rows]
                if not any(abs(abs(rec - correct_base) - iss.expected_amount_minor) <= iss_tol for rec in recorded_amounts):
                    independent_evidence_sufficiency = False
                    errors.append(
                        f"FX issue {iss.issue_id} discrepancy mismatch: correct_base={correct_base}, "
                        f"recorded={recorded_amounts}, truth={iss.expected_amount_minor}"
                    )

        elif iss.type == IssueType.PERIOD_CUTOFF_ERROR:
            out_of_period_found = False
            for r in ref_rows:
                r_date = extract_iso_date(r.timestamp)
                if r_date < period_start_date or r_date > period_end_date:
                    out_of_period_found = True
            if not out_of_period_found:
                independent_evidence_sufficiency = False
                errors.append(f"Cutoff issue {iss.issue_id} has no ledger row timestamp outside reporting period")
            if not any(abs(max(r.debit_minor, r.credit_minor) - iss.expected_amount_minor) <= iss_tol for r in ref_rows):
                independent_evidence_sufficiency = False
                errors.append(f"Cutoff issue {iss.issue_id} ledger amount does not match expected_amount_minor")

    checks = {
        "manifest_counts_match": manifest_counts_match,
        "invoice_line_items_sum_ok": invoice_line_items_sum_ok,
        "pre_mutation_trial_balance_ok": pre_mutation_trial_balance_ok,
        "no_dangling_open_references": no_dangling_open_references,
        "truth_ids_exist_in_open_records": truth_ids_exist_in_open_records,
        "independent_evidence_sufficiency": independent_evidence_sufficiency,
    }

    return {
        "valid": all(checks.values()) and len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "issues_verified": len(truth.issues),
    }

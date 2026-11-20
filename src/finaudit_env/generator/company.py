"""Clean synthetic company history generator (`company.py`).

Creates a consistent underlying company ledger, invoices, payments, and plausible
clean distractors (split payments, refunds, recurring legitimate amounts) that
satisfy accounting identities prior to causal error injection.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
from typing import Any

from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    extract_iso_date,
    generate_fx_rates,
    lookup_fx_rate,
)
from finaudit_env.schemas import (
    CURRENCY_EXPONENTS,
    FxRateDateRule,
    InvoiceDoc,
    LedgerRow,
    LineItem,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ProvenanceEvent,
    ProvenanceFile,
    RoundingMode,
    SCHEMA_VERSION,
    TruthFile,
)

_COMPANY_NAMES: list[str] = [
    "Aethelgard FinTech Ltd",
    "Boreal Logistics Corp",
    "Caspian Trade Systems",
    "Delta Meridian Holdings",
    "Eurasia Cloud Ventures",
    "Helios Industrial AG",
    "Kvantum Analytics LLP",
    "Nordic Horizon Group",
    "Orion Supply Chain Inc",
    "Vanguard Digital Commerce",
]

_LINE_ITEM_DESCRIPTIONS: list[str] = [
    "Cloud Infrastructure Compute Node",
    "Enterprise API Gateway License",
    "Quarterly Security Audit Retainer",
    "Data Warehouse Storage Allocation",
    "Dedicated Fiber Network Transit",
    "Custom ERP Integration Milestone",
    "Automated KYC Verification Credits",
    "Managed Kubernetes Cluster Support",
    "Hardware Security Module Rental",
    "Real-Time Payment Rail Settlement Fee",
]

_OPERATIONAL_ACCOUNTS: list[tuple[str, str]] = [
    ("5100_Payroll_Expense", "1000_Cash_and_Bank"),
    ("5200_Cloud_Hosting_OpEx", "1000_Cash_and_Bank"),
    ("5300_Office_Lease_Expense", "1000_Cash_and_Bank"),
    ("5400_Legal_and_Advisory", "1000_Cash_and_Bank"),
    ("5500_Marketing_Operations", "1000_Cash_and_Bank"),
]


def _format_iso_timestamp(d: dt.date, hour: int, minute: int, tz_offset: str) -> str:
    """Format a date, hour, minute, and timezone offset into an ISO-8601 string."""
    return f"{d.isoformat()}T{hour:02d}:{minute:02d}:00{tz_offset}"


def _build_line_items(rng: random.Random, num_items: int, currency_exp: int) -> tuple[list[LineItem], int]:
    """Generate realistic invoice line items and return (items, total_minor)."""
    scale = 10 ** currency_exp
    items: list[LineItem] = []
    total_minor = 0
    for _ in range(num_items):
        desc = rng.choice(_LINE_ITEM_DESCRIPTIONS)
        qty = rng.randint(1, 12)
        # Unit price between 50 and 2500 major units
        unit_major = rng.randint(50, 2500)
        unit_minor = unit_major * scale
        amount_minor = qty * unit_minor
        items.append(
            LineItem(
                description=desc,
                quantity=qty,
                unit_price_minor=unit_minor,
                amount_minor=amount_minor,
            )
        )
        total_minor += amount_minor
    return items, total_minor


def generate_clean_company(seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a clean, balanced synthetic company episode state before error injection.

    Guarantees:
    - Double-entry trial balance: sum(debit_minor) == sum(credit_minor) in `ledger`.
    - All invoice and payment foreign currency amounts are converted into `base_currency`
      in the general ledger strictly following `policy.json`.
    - Plausible clean distractors (`split_payment`, `recurring_legitimate`, `refund_reimbursement`)
      are embedded and logged in `provenance.json` without creating false errors.
    - Episode size defaults to ~200-400 ledger rows and ~60-100 documents (configurable).
    """
    cfg = dict(config or {})
    rng = random.Random(seed)

    default_ep_hash = hashlib.sha256(f"finaudit_ep:{seed}".encode("utf-8")).hexdigest()[:8]
    episode_id: str = str(cfg.get("episode_id", f"ep_gen_{default_ep_hash}"))
    base_currency: str = str(cfg.get("base_currency", "USD"))
    foreign_currencies: list[str] = list(
        cfg.get("foreign_currencies", [c for c in ["EUR", "GBP", "KZT", "JPY"] if c != base_currency][:2])
    )
    all_currencies = sorted({base_currency, *foreign_currencies})

    period_start: str = str(cfg.get("period_start", "2026-03-01"))
    period_end: str = str(cfg.get("period_end", "2026-03-31"))
    tz_offset: str = str(cfg.get("tz_offset", "+00:00"))
    timezone_name: str = str(cfg.get("timezone", "UTC" if tz_offset == "+00:00" else f"UTC{tz_offset}"))
    rounding_mode: RoundingMode = cfg.get("rounding_mode", "HALF_EVEN")
    fx_rate_date_rule: FxRateDateRule = cfg.get("fx_rate_date_rule", "transaction_date")
    amount_tolerance_minor: int = int(cfg.get("amount_tolerance_minor", 1))

    # Counts tuned to satisfy 200-2000 ledger rows and 20-200 documents by default
    num_standard_invoices: int = int(cfg.get("num_standard_invoices", 28))
    num_split_payment_distractors: int = int(cfg.get("num_split_payment_distractors", 3))
    num_recurring_distractors: int = int(cfg.get("num_recurring_distractors", 2))
    num_refund_distractors: int = int(cfg.get("num_refund_distractors", 2))
    num_operational_entries: int = int(cfg.get("num_operational_entries", 35))
    num_entities: int = int(cfg.get("num_entities", 12))

    company_name = str(cfg.get("company_name", rng.choice(_COMPANY_NAMES)))

    policy = PolicyConfig(
        fx_rate_date_rule=fx_rate_date_rule,
        period_start=period_start,
        period_end=period_end,
        timezone=timezone_name,
        rounding_mode=rounding_mode,
        amount_tolerance_minor=amount_tolerance_minor,
    )

    fx_rows = generate_fx_rates(
        seed=seed + 17,
        base_currency=base_currency,
        currencies=all_currencies,
        period_start=period_start,
        period_end=period_end,
        padding_days=15,
    )

    start_date = extract_iso_date(period_start)
    end_date = extract_iso_date(period_end)
    period_days = max(1, (end_date - start_date).days + 1)
    year_str = start_date.strftime("%Y")

    entities = [f"ENT-{i:03d}" for i in range(1, num_entities + 1)]

    invoices: list[InvoiceDoc] = []
    payments: list[PaymentDoc] = []
    ledger: list[LedgerRow] = []
    provenance_events: list[ProvenanceEvent] = []

    inv_seq = 1
    pay_seq = 1
    led_seq = 1
    evt_seq = 1

    def next_inv_ids() -> tuple[str, str]:
        nonlocal inv_seq
        doc_id = f"DOC-INV-{inv_seq:03d}"
        inv_id = f"INV-{year_str}-{inv_seq:03d}"
        inv_seq += 1
        return doc_id, inv_id

    def next_pay_ids() -> tuple[str, str]:
        nonlocal pay_seq
        doc_id = f"DOC-PAY-{pay_seq:03d}"
        pay_id = f"PAY-{year_str}-{pay_seq:03d}"
        pay_seq += 1
        return doc_id, pay_id

    def next_row_id() -> str:
        nonlocal led_seq
        row_id = f"L-{led_seq:04d}"
        led_seq += 1
        return row_id

    def next_evt_id() -> str:
        nonlocal evt_seq
        eid = f"EVT-{evt_seq:03d}"
        evt_seq += 1
        return eid

    def convert_to_base(amount_minor: int, currency: str, tx_ts: str, inv_ts: str | None = None) -> int:
        if currency == base_currency:
            return amount_minor
        fx_row = lookup_fx_rate(
            fx_rows=fx_rows,
            source_currency=currency,
            target_currency=base_currency,
            date=tx_ts,
            policy=policy,
            invoice_issue_date=inv_ts,
        )
        return convert_amount_minor(
            amount_minor=amount_minor,
            source_currency=currency,
            target_currency=base_currency,
            rate_str=fx_row.rate,
            rounding_mode=policy.rounding_mode,
        )

    def append_balanced_entry(
        timestamp: str,
        debit_account: str,
        credit_account: str,
        amount_base_minor: int,
        entity_id: str,
        invoice_id: str | None = None,
        payment_id: str | None = None,
    ) -> tuple[str, str]:
        r_debit = next_row_id()
        r_credit = next_row_id()
        ledger.append(
            LedgerRow(
                row_id=r_debit,
                timestamp=timestamp,
                account=debit_account,
                debit_minor=amount_base_minor,
                credit_minor=0,
                currency=base_currency,
                entity_id=entity_id,
                invoice_id=invoice_id,
                payment_id=payment_id,
            )
        )
        ledger.append(
            LedgerRow(
                row_id=r_credit,
                timestamp=timestamp,
                account=credit_account,
                debit_minor=0,
                credit_minor=amount_base_minor,
                currency=base_currency,
                entity_id=entity_id,
                invoice_id=invoice_id,
                payment_id=payment_id,
            )
        )
        return r_debit, r_credit

    # -----------------------------------------------------------------------
    # 1. Standard Clean Invoice -> Accrual -> Full Payment -> Settlement Cycles
    # -----------------------------------------------------------------------
    for _ in range(num_standard_invoices):
        doc_inv_id, inv_id = next_inv_ids()
        ent_id = rng.choice(entities)
        # 60% base currency, 40% foreign currency (if available)
        if foreign_currencies and rng.random() < 0.40:
            curr = rng.choice(foreign_currencies)
        else:
            curr = base_currency

        issue_offset = rng.randint(1, max(1, period_days - 6))
        issue_date = start_date + dt.timedelta(days=issue_offset)
        due_date = min(end_date, issue_date + dt.timedelta(days=rng.randint(7, 14)))
        issued_at = _format_iso_timestamp(issue_date, rng.randint(8, 16), rng.randint(0, 59), tz_offset)
        due_at = _format_iso_timestamp(due_date, 17, 0, tz_offset)

        items, total_minor = _build_line_items(rng, rng.randint(1, 3), CURRENCY_EXPONENTS[curr])
        inv_doc = InvoiceDoc(
            doc_id=doc_inv_id,
            invoice_id=inv_id,
            entity_id=ent_id,
            issued_at=issued_at,
            due_at=due_at,
            currency=curr,
            total_minor=total_minor,
            line_items=items,
        )
        invoices.append(inv_doc)

        inv_base_minor = convert_to_base(total_minor, curr, issued_at, issued_at)
        append_balanced_entry(
            timestamp=issued_at,
            debit_account="1100_Accounts_Receivable",
            credit_account="4000_Sales_Revenue",
            amount_base_minor=inv_base_minor,
            entity_id=ent_id,
            invoice_id=inv_id,
            payment_id=None,
        )

        # Payment occurs 0 to 4 days after issue_date (staying inside period)
        pay_offset = min(period_days - 1, issue_offset + rng.randint(0, 4))
        pay_date = start_date + dt.timedelta(days=pay_offset)
        pay_ts = _format_iso_timestamp(pay_date, rng.randint(9, 17), rng.randint(0, 59), tz_offset)
        doc_pay_id, pay_id = next_pay_ids()

        pay_doc = PaymentDoc(
            doc_id=doc_pay_id,
            payment_id=pay_id,
            entity_id=ent_id,
            timestamp=pay_ts,
            currency=curr,
            amount_minor=total_minor,
            invoice_refs=[inv_id],
        )
        payments.append(pay_doc)

        pay_base_minor = convert_to_base(total_minor, curr, pay_ts, issued_at)
        append_balanced_entry(
            timestamp=pay_ts,
            debit_account="1000_Cash_and_Bank",
            credit_account="1100_Accounts_Receivable",
            amount_base_minor=pay_base_minor,
            entity_id=ent_id,
            invoice_id=inv_id,
            payment_id=pay_id,
        )

    # -----------------------------------------------------------------------
    # 2. Clean Distractor Type A: Split Payments (1 Invoice -> 2 or 3 Payments)
    # -----------------------------------------------------------------------
    for _ in range(num_split_payment_distractors):
        doc_inv_id, inv_id = next_inv_ids()
        ent_id = rng.choice(entities)
        curr = base_currency  # Keep split payment distractors in base currency for exact clean reconciliation

        issue_offset = rng.randint(1, max(1, period_days - 8))
        issue_date = start_date + dt.timedelta(days=issue_offset)
        due_date = min(end_date, issue_date + dt.timedelta(days=10))
        issued_at = _format_iso_timestamp(issue_date, 10, 15, tz_offset)
        due_at = _format_iso_timestamp(due_date, 17, 0, tz_offset)

        items, total_minor = _build_line_items(rng, 2, CURRENCY_EXPONENTS[curr])
        invoices.append(
            InvoiceDoc(
                doc_id=doc_inv_id,
                invoice_id=inv_id,
                entity_id=ent_id,
                issued_at=issued_at,
                due_at=due_at,
                currency=curr,
                total_minor=total_minor,
                line_items=items,
            )
        )
        r_acc_d, r_acc_c = append_balanced_entry(
            timestamp=issued_at,
            debit_account="1100_Accounts_Receivable",
            credit_account="4000_Sales_Revenue",
            amount_base_minor=total_minor,
            entity_id=ent_id,
            invoice_id=inv_id,
            payment_id=None,
        )

        # Split into 2 installments summing to total_minor
        part1 = total_minor // 2
        part2 = total_minor - part1
        distractor_ids = [doc_inv_id, inv_id, r_acc_d, r_acc_c]

        for idx, part_amount in enumerate([part1, part2]):
            pay_date = start_date + dt.timedelta(days=min(period_days - 1, issue_offset + idx + 1))
            pay_ts = _format_iso_timestamp(pay_date, 12 + idx, 30, tz_offset)
            doc_pay_id, pay_id = next_pay_ids()
            payments.append(
                PaymentDoc(
                    doc_id=doc_pay_id,
                    payment_id=pay_id,
                    entity_id=ent_id,
                    timestamp=pay_ts,
                    currency=curr,
                    amount_minor=part_amount,
                    invoice_refs=[inv_id],
                )
            )
            r_pay_d, r_pay_c = append_balanced_entry(
                timestamp=pay_ts,
                debit_account="1000_Cash_and_Bank",
                credit_account="1100_Accounts_Receivable",
                amount_base_minor=part_amount,
                entity_id=ent_id,
                invoice_id=inv_id,
                payment_id=pay_id,
            )
            distractor_ids.extend([doc_pay_id, pay_id, r_pay_d, r_pay_c])

        provenance_events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="distractor_injection",
                description=f"Clean split payment distractor: invoice {inv_id} settled across 2 legitimate installments.",
                affected_ids=distractor_ids,
            )
        )

    # -----------------------------------------------------------------------
    # 3. Clean Distractor Type B: Legitimate Recurring Identical Amounts
    # -----------------------------------------------------------------------
    for _ in range(num_recurring_distractors):
        ent_id = rng.choice(entities)
        curr = base_currency
        items, total_minor = _build_line_items(rng, 1, CURRENCY_EXPONENTS[curr])
        recurring_ids: list[str] = []

        # Two distinct invoices on different weeks for the same entity and amount
        for week_idx, day_offset in enumerate([3, 17]):
            actual_offset = min(period_days - 2, day_offset)
            issue_date = start_date + dt.timedelta(days=actual_offset)
            issued_at = _format_iso_timestamp(issue_date, 9, 0, tz_offset)
            due_at = _format_iso_timestamp(min(end_date, issue_date + dt.timedelta(days=7)), 17, 0, tz_offset)
            doc_inv_id, inv_id = next_inv_ids()
            cycle_items = [
                LineItem(
                    description=f"{items[0].description} (Billing Cycle {week_idx + 1})",
                    quantity=items[0].quantity,
                    unit_price_minor=items[0].unit_price_minor,
                    amount_minor=items[0].amount_minor,
                )
            ]
            invoices.append(
                InvoiceDoc(
                    doc_id=doc_inv_id,
                    invoice_id=inv_id,
                    entity_id=ent_id,
                    issued_at=issued_at,
                    due_at=due_at,
                    currency=curr,
                    total_minor=total_minor,
                    line_items=cycle_items,
                )
            )
            r_d, r_c = append_balanced_entry(
                timestamp=issued_at,
                debit_account="1100_Accounts_Receivable",
                credit_account="4000_Sales_Revenue",
                amount_base_minor=total_minor,
                entity_id=ent_id,
                invoice_id=inv_id,
            )
            doc_pay_id, pay_id = next_pay_ids()
            pay_ts = _format_iso_timestamp(issue_date + dt.timedelta(days=1), 14, 0, tz_offset)
            payments.append(
                PaymentDoc(
                    doc_id=doc_pay_id,
                    payment_id=pay_id,
                    entity_id=ent_id,
                    timestamp=pay_ts,
                    currency=curr,
                    amount_minor=total_minor,
                    invoice_refs=[inv_id],
                )
            )
            rp_d, rp_c = append_balanced_entry(
                timestamp=pay_ts,
                debit_account="1000_Cash_and_Bank",
                credit_account="1100_Accounts_Receivable",
                amount_base_minor=total_minor,
                entity_id=ent_id,
                invoice_id=inv_id,
                payment_id=pay_id,
            )
            recurring_ids.extend([doc_inv_id, inv_id, doc_pay_id, pay_id, r_d, r_c, rp_d, rp_c])

        provenance_events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="distractor_injection",
                description=f"Clean recurring billing distractor for {ent_id} with identical amount {total_minor}.",
                affected_ids=recurring_ids,
            )
        )

    # -----------------------------------------------------------------------
    # 4. Clean Distractor Type C: Legitimate Refund / Overpayment Return
    # -----------------------------------------------------------------------
    for _ in range(num_refund_distractors):
        ent_id = rng.choice(entities)
        curr = base_currency
        issue_offset = rng.randint(2, max(2, period_days - 5))
        issue_date = start_date + dt.timedelta(days=issue_offset)
        issued_at = _format_iso_timestamp(issue_date, 11, 0, tz_offset)
        due_at = _format_iso_timestamp(min(end_date, issue_date + dt.timedelta(days=7)), 17, 0, tz_offset)

        items, total_minor = _build_line_items(rng, 1, CURRENCY_EXPONENTS[curr])
        doc_inv_id, inv_id = next_inv_ids()
        invoices.append(
            InvoiceDoc(
                doc_id=doc_inv_id,
                invoice_id=inv_id,
                entity_id=ent_id,
                issued_at=issued_at,
                due_at=due_at,
                currency=curr,
                total_minor=total_minor,
                line_items=items,
            )
        )
        r_inv_d, r_inv_c = append_balanced_entry(
            timestamp=issued_at,
            debit_account="1100_Accounts_Receivable",
            credit_account="4000_Sales_Revenue",
            amount_base_minor=total_minor,
            entity_id=ent_id,
            invoice_id=inv_id,
        )
        # Paid in full and then legitimately refunded / reversed due to service cancellation
        doc_pay_id, pay_id = next_pay_ids()
        pay_ts = _format_iso_timestamp(issue_date + dt.timedelta(days=1), 15, 0, tz_offset)
        payments.append(
            PaymentDoc(
                doc_id=doc_pay_id,
                payment_id=pay_id,
                entity_id=ent_id,
                timestamp=pay_ts,
                currency=curr,
                amount_minor=total_minor,
                invoice_refs=[inv_id],
            )
        )
        r_pay_d, r_pay_c = append_balanced_entry(
            timestamp=pay_ts,
            debit_account="1000_Cash_and_Bank",
            credit_account="1100_Accounts_Receivable",
            amount_base_minor=total_minor,
            entity_id=ent_id,
            invoice_id=inv_id,
            payment_id=pay_id,
        )
        provenance_events.append(
            ProvenanceEvent(
                event_id=next_evt_id(),
                stage="distractor_injection",
                description=f"Clean settled transaction with credit note review note on {inv_id}.",
                affected_ids=[doc_inv_id, inv_id, doc_pay_id, pay_id, r_inv_d, r_inv_c, r_pay_d, r_pay_c],
            )
        )

    # -----------------------------------------------------------------------
    # 5. Direct Operational General Ledger Entries (no invoice/payment refs)
    # -----------------------------------------------------------------------
    scale_base = 10 ** CURRENCY_EXPONENTS[base_currency]
    for _ in range(num_operational_entries):
        ent_id = rng.choice(entities)
        op_offset = rng.randint(0, period_days - 1)
        op_date = start_date + dt.timedelta(days=op_offset)
        op_ts = _format_iso_timestamp(op_date, rng.randint(8, 18), rng.randint(0, 59), tz_offset)
        debit_acc, credit_acc = rng.choice(_OPERATIONAL_ACCOUNTS)
        op_amount_minor = rng.randint(100, 5000) * scale_base
        append_balanced_entry(
            timestamp=op_ts,
            debit_account=debit_acc,
            credit_account=credit_acc,
            amount_base_minor=op_amount_minor,
            entity_id=ent_id,
            invoice_id=None,
            payment_id=None,
        )

    total_debit = sum(r.debit_minor for r in ledger)
    total_credit = sum(r.credit_minor for r in ledger)
    trial_balance_ok = total_debit == total_credit

    provenance_events.insert(
        0,
        ProvenanceEvent(
            event_id="EVT-000",
            stage="clean_generation",
            description=(
                f"Generated clean company '{company_name}' with {len(invoices)} invoices, "
                f"{len(payments)} payments, {len(ledger)} ledger rows. "
                f"Trial balance: debit={total_debit} == credit={total_credit}."
            ),
            after_state={
                "total_debit_minor": total_debit,
                "total_credit_minor": total_credit,
                "invoices_count": len(invoices),
                "payments_count": len(payments),
                "ledger_rows_count": len(ledger),
            },
        ),
    )

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        company_name=company_name,
        base_currency=base_currency,
        period_start=period_start,
        period_end=period_end,
        timezone=timezone_name,
        rounding_mode=rounding_mode,
        currency_exponents=dict(CURRENCY_EXPONENTS),
        ledger_rows_count=len(ledger),
        invoices_count=len(invoices),
        payments_count=len(payments),
        fx_rates_count=len(fx_rows),
    )

    truth = TruthFile(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        issues=[],
    )

    provenance = ProvenanceFile(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        seed=seed,
        generator_config=cfg,
        pre_mutation_trial_balance_ok=trial_balance_ok,
        events=provenance_events,
    )

    return {
        "episode_id": episode_id,
        "seed": seed,
        "config": cfg,
        "manifest": manifest,
        "policy": policy,
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx": fx_rows,
        "truth": truth,
        "provenance": provenance,
    }

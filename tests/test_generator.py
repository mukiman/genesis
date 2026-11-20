"""Unit tests for Participant 1 (`feat/generator`) modules.

Tests Stage 1 (`fx_utils`), and will sequentially expand to cover
Stages 2-5 (`company`, `errors`, `transforms`, `verifier`, `build_datasets`).
"""

from __future__ import annotations

from finaudit_env.generator.company import generate_clean_company
from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    generate_fx_rates,
    lookup_fx_rate,
    resolve_effective_fx_date,
)
from finaudit_env.schemas import PolicyConfig


def test_fx_generation_determinism_and_padding() -> None:
    rows_a = generate_fx_rates(
        seed=42,
        base_currency="USD",
        currencies=["USD", "EUR", "JPY", "KZT"],
        period_start="2026-03-01",
        period_end="2026-03-31",
    )
    rows_b = generate_fx_rates(
        seed=42,
        base_currency="USD",
        currencies=["JPY", "EUR", "USD", "KZT"],
        period_start="2026-03-01",
        period_end="2026-03-31",
    )
    assert len(rows_a) == len(rows_b)
    assert [r.model_dump() for r in rows_a] == [r.model_dump() for r in rows_b]

    # Verify canonical fx_id auto-population and padding before/after March 2026
    dates = {r.date for r in rows_a}
    assert "2026-02-20" in dates
    assert "2026-04-05" in dates
    for r in rows_a:
        assert r.fx_id == f"FX-{r.date}-{r.source_currency}-{r.target_currency}"


def test_convert_amount_minor_exponents_and_rounding() -> None:
    # EUR (exp 2) -> USD (exp 2): 100.00 EUR (10000) * 1.0850 = 108.50 USD (10850)
    assert convert_amount_minor(10000, "EUR", "USD", "1.0850", "HALF_EVEN") == 10850

    # JPY (exp 0) -> USD (exp 2): 10,000 JPY (10000) * 0.006650 USD/JPY = 66.50 USD (6650 cents)
    assert convert_amount_minor(10000, "JPY", "USD", "0.006650", "HALF_EVEN") == 6650

    # USD (exp 2) -> JPY (exp 0): 100.00 USD (10000 cents) * 150.2500 JPY/USD = 15,025 JPY
    assert convert_amount_minor(10000, "USD", "JPY", "150.2500", "HALF_EVEN") == 15025

    # Rounding mode behavior on half-way boundary: 101 * 1.005 = 101.505 -> .5 rounds differently
    # 15 * 0.5 = 7.5
    assert convert_amount_minor(15, "USD", "EUR", "0.5000", "HALF_EVEN") == 8
    assert convert_amount_minor(15, "USD", "EUR", "0.5000", "DOWN") == 7
    assert convert_amount_minor(15, "USD", "EUR", "0.5000", "UP") == 8
    assert convert_amount_minor(14, "USD", "EUR", "0.5000", "HALF_EVEN") == 7
    assert convert_amount_minor(13, "USD", "EUR", "0.5000", "HALF_EVEN") == 6
    assert convert_amount_minor(13, "USD", "EUR", "0.5000", "HALF_UP") == 7


def test_lookup_fx_rate_policy_rules() -> None:
    rows = generate_fx_rates(
        seed=101,
        base_currency="USD",
        currencies=["USD", "EUR"],
        period_start="2026-03-01",
        period_end="2026-03-31",
    )

    # 1. transaction_date rule
    pol_tx = PolicyConfig(
        fx_rate_date_rule="transaction_date",
        period_start="2026-03-01",
        period_end="2026-03-31",
    )
    r_tx = lookup_fx_rate(rows, "EUR", "USD", "2026-03-16T14:20:00+00:00", policy=pol_tx)
    assert r_tx.date == "2026-03-16"

    # 2. invoice_issue_date rule
    pol_inv = PolicyConfig(
        fx_rate_date_rule="invoice_issue_date",
        period_start="2026-03-01",
        period_end="2026-03-31",
    )
    r_inv = lookup_fx_rate(
        rows,
        "EUR",
        "USD",
        "2026-03-20T10:00:00+00:00",
        policy=pol_inv,
        invoice_issue_date="2026-03-10T09:00:00+00:00",
    )
    assert r_inv.date == "2026-03-10"

    # 3. previous_business_day rule: 2026-03-16 is a Monday -> previous business day is Friday 2026-03-13
    pol_pbd = PolicyConfig(
        fx_rate_date_rule="previous_business_day",
        period_start="2026-03-01",
        period_end="2026-03-31",
    )
    assert resolve_effective_fx_date("2026-03-16T11:00:00+00:00", pol_pbd) == "2026-03-13"
    r_pbd = lookup_fx_rate(rows, "EUR", "USD", "2026-03-16T11:00:00+00:00", policy=pol_pbd)
    assert r_pbd.date == "2026-03-13"


def test_clean_company_trial_balance_and_distractors() -> None:
    ep1 = generate_clean_company(seed=777, config={"episode_id": "ep_test_777", "base_currency": "USD"})
    ep2 = generate_clean_company(seed=777, config={"episode_id": "ep_test_777", "base_currency": "USD"})

    # 1. Exact reproducibility for identical seed & config
    assert ep1["manifest"].model_dump() == ep2["manifest"].model_dump()
    assert [r.model_dump() for r in ep1["ledger"]] == [r.model_dump() for r in ep2["ledger"]]
    assert [i.model_dump() for i in ep1["invoices"]] == [i.model_dump() for i in ep2["invoices"]]
    assert [p.model_dump() for p in ep1["payments"]] == [p.model_dump() for p in ep2["payments"]]

    # 2. Trial balance invariant: sum(debit_minor) == sum(credit_minor)
    ledger = ep1["ledger"]
    total_debit = sum(r.debit_minor for r in ledger)
    total_credit = sum(r.credit_minor for r in ledger)
    assert total_debit > 0
    assert total_debit == total_credit
    assert ep1["provenance"].pre_mutation_trial_balance_ok is True

    # 3. Size limits compliance (200-2000 ledger rows, 20-200 documents)
    total_docs = len(ep1["invoices"]) + len(ep1["payments"])
    assert 200 <= len(ledger) <= 2000
    assert 20 <= total_docs <= 200
    assert ep1["manifest"].ledger_rows_count == len(ledger)
    assert ep1["manifest"].invoices_count == len(ep1["invoices"])
    assert ep1["manifest"].payments_count == len(ep1["payments"])

    # 4. Clean episode has 0 errors in truth.json, but includes logged distractors
    assert len(ep1["truth"].issues) == 0
    distractor_events = [e for e in ep1["provenance"].events if e.stage == "distractor_injection"]
    assert len(distractor_events) >= 5


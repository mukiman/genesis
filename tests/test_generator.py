"""Unit tests for Participant 1 (`feat/generator`) modules.

Tests Stage 1 (`fx_utils`), and will sequentially expand to cover
Stages 2-5 (`company`, `errors`, `transforms`, `verifier`, `build_datasets`).
"""

from __future__ import annotations

from finaudit_env.generator.company import generate_clean_company
from finaudit_env.generator.errors import inject_errors
from finaudit_env.generator.fx_utils import (
    convert_amount_minor,
    generate_fx_rates,
    lookup_fx_rate,
    resolve_effective_fx_date,
)
from finaudit_env.generator.transforms import permute_and_remap_episode, remap_submission_ids
from finaudit_env.generator.verifier import verify_episode
from finaudit_env.schemas import Finding, FindingsSubmission, IssueType, PolicyConfig


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


def test_error_injection_four_types_and_interacting_scenarios() -> None:
    clean = generate_clean_company(seed=2026, config={"episode_id": "ep_err_2026", "base_currency": "USD"})
    mutated = inject_errors(
        clean_episode=clean,
        seed=2026,
        error_config={
            "duplicate_invoice": 1,
            "unallocated_or_partial_payment": 1,
            "fx_conversion_error": 1,
            "period_cutoff_error": 1,
            "interacting_scenarios": ["partial_fx_cutoff", "duplicate_fx_cutoff"],
        },
    )

    # Original clean episode must remain untouched
    assert len(clean["truth"].issues) == 0

    issues = mutated["truth"].issues
    # 4 standalone + 2 from partial_fx_cutoff + 2 from duplicate_fx_cutoff = 8 issues
    assert len(issues) == 8

    present_types = {iss.type for iss in issues}
    assert present_types == {
        IssueType.DUPLICATE_INVOICE,
        IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
        IssueType.FX_CONVERSION_ERROR,
        IssueType.PERIOD_CUTOFF_ERROR,
    }

    # Build set of all open IDs in mutated episode
    open_ids: set[str] = set()
    open_ids.update(r.row_id for r in mutated["ledger"])
    open_ids.update(i.doc_id for i in mutated["invoices"])
    open_ids.update(i.invoice_id for i in mutated["invoices"])
    open_ids.update(p.doc_id for p in mutated["payments"])
    open_ids.update(p.payment_id for p in mutated["payments"])
    open_ids.update(f.fx_id for f in mutated["fx"] if f.fx_id)

    for iss in issues:
        assert iss.expected_amount_minor > 0
        assert iss.currency == "USD"
        for aid in iss.affected_ids:
            assert aid in open_ids, f"Affected ID {aid} missing from open records"
        for ev_set in iss.valid_evidence_sets:
            assert len(ev_set) >= 2
            for eid in ev_set:
                assert eid in open_ids, f"Evidence ID {eid} missing from open records"

    error_events = [e for e in mutated["provenance"].events if e.stage == "error_injection"]
    assert len(error_events) == 8


def test_verifier_and_isomorphic_transforms() -> None:
    clean = generate_clean_company(
        seed=3030,
        config={
            "episode_id": "ep_ver_3030",
            "base_currency": "USD",
            "fx_rate_date_rule": "invoice_issue_date",
        },
    )
    res_clean = verify_episode(clean)
    assert res_clean["valid"] is True, f"Clean verification failed: {res_clean['errors']}"

    mutated = inject_errors(
        clean_episode=clean,
        seed=3030,
        error_config={
            "duplicate_invoice": 1,
            "unallocated_or_partial_payment": 1,
            "fx_conversion_error": 1,
            "period_cutoff_error": 1,
            "interacting_scenarios": ["partial_fx_cutoff", "duplicate_fx_cutoff"],
        },
    )
    res_mutated = verify_episode(mutated)
    assert res_mutated["valid"] is True, f"Mutated verification failed: {res_mutated['errors']}"
    assert res_mutated["issues_verified"] == 8

    # Apply isomorphic ID remapping and row permutation
    transformed, id_map = permute_and_remap_episode(mutated, seed=9999)
    res_trans = verify_episode(transformed)
    assert res_trans["valid"] is True, f"Transformed verification failed: {res_trans['errors']}"
    assert res_trans["issues_verified"] == 8

    # Ensure IDs actually changed and row order changed
    orig_row_ids = [r.row_id for r in mutated["ledger"]]
    trans_row_ids = [r.row_id for r in transformed["ledger"]]
    assert set(orig_row_ids).isdisjoint(set(trans_row_ids))

    # Build an Oracle submission on original episode, remap its IDs, and verify it matches transformed truth
    oracle_sub = FindingsSubmission(
        episode_id=mutated["episode_id"],
        findings=[
            Finding(
                type=iss.type,
                affected_ids=iss.affected_ids,
                claimed_amount_minor=iss.expected_amount_minor,
                currency=iss.currency,
                evidence_ids=iss.valid_evidence_sets[0],
                explanation="Oracle verification finding",
            )
            for iss in mutated["truth"].issues
        ],
    )
    remapped_sub = remap_submission_ids(oracle_sub, id_map)
    for finding, trans_iss in zip(remapped_sub.findings, transformed["truth"].issues):
        assert finding.type == trans_iss.type
        assert finding.affected_ids == trans_iss.affected_ids
        assert finding.claimed_amount_minor == trans_iss.expected_amount_minor
        assert finding.evidence_ids == trans_iss.valid_evidence_sets[0]




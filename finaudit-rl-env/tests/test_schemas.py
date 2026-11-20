"""Comprehensive unit tests for shared Pydantic v2 schemas (`finaudit_env.schemas`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finaudit_env.schemas import (
    ACTION_ADAPTER,
    CURRENCY_EXPONENTS,
    SCHEMA_VERSION,
    CalculatorAction,
    Finding,
    FindingsSubmission,
    FxRateRow,
    GetDocumentAction,
    GetRecordAction,
    InvoiceDoc,
    IssueType,
    LedgerRow,
    LineItem,
    ListRecordsAction,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ProvenanceEvent,
    ProvenanceFile,
    SubmitAction,
    TruthFile,
    TruthIssue,
    parse_action,
)


def test_constants_and_issue_types() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    assert CURRENCY_EXPONENTS == {
        "USD": 2,
        "EUR": 2,
        "GBP": 2,
        "KZT": 2,
        "JPY": 0,
        "CHF": 2,
    }
    assert [member.value for member in IssueType] == [
        "duplicate_invoice",
        "unallocated_or_partial_payment",
        "fx_conversion_error",
        "period_cutoff_error",
    ]


def test_manifest_and_policy_config_valid_and_extra_forbid() -> None:
    policy = PolicyConfig(
        fx_rate_date_rule="transaction_date",
        period_start="2026-01-01",
        period_end="2026-03-31",
        timezone="UTC",
        rounding_mode="HALF_EVEN",
        amount_tolerance_minor=1,
    )
    assert policy.rounding_mode == "HALF_EVEN"

    manifest = Manifest(
        episode_id="ep_train_001",
        base_currency="USD",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-03-31T23:59:59+00:00",
        timezone="UTC",
        ledger_rows_count=250,
        invoices_count=40,
        payments_count=35,
        fx_rates_count=90,
    )
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.currency_exponents["JPY"] == 0

    with pytest.raises(ValidationError):
        Manifest(
            episode_id="ep_train_001",
            base_currency="USD",
            period_start="2026-01-01",
            period_end="2026-03-31",
            ledger_rows_count=10,
            invoices_count=5,
            payments_count=5,
            fx_rates_count=5,
            unexpected_field="forbidden",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        Manifest(
            episode_id="ep_train_001",
            base_currency="RUB",
            period_start="2026-01-01",
            period_end="2026-03-31",
            ledger_rows_count=10,
            invoices_count=5,
            payments_count=5,
            fx_rates_count=5,
        )


def test_open_records_models_valid_and_strict() -> None:
    ledger = LedgerRow(
        row_id="L-0001",
        timestamp="2026-02-10T12:00:00+00:00",
        account="1100_ACCOUNTS_RECEIVABLE",
        debit_minor=150000,
        credit_minor=0,
        currency="USD",
        entity_id="ENT-001",
        invoice_id="INV-2026-001",
        payment_id=None,
    )
    assert ledger.row_id == "L-0001"
    assert ledger.payment_id is None

    item = LineItem(
        description="Cloud Audit Subscription",
        quantity=2,
        unit_price_minor=75000,
        amount_minor=150000,
    )
    invoice = InvoiceDoc(
        doc_id="DOC-INV-001",
        invoice_id="INV-2026-001",
        entity_id="ENT-001",
        issued_at="2026-02-10T09:00:00+00:00",
        due_at="2026-03-10T09:00:00+00:00",
        currency="USD",
        total_minor=150000,
        line_items=[item],
    )
    assert invoice.line_items[0].amount_minor == 150000

    payment = PaymentDoc(
        doc_id="DOC-PAY-001",
        payment_id="PAY-2026-001",
        entity_id="ENT-001",
        timestamp="2026-02-20T15:30:00+00:00",
        currency="USD",
        amount_minor=150000,
        invoice_refs=["INV-2026-001"],
    )
    assert payment.invoice_refs == ["INV-2026-001"]

    fx = FxRateRow(
        date="2026-02-10",
        source_currency="EUR",
        target_currency="USD",
        rate="1.0845",
    )
    assert fx.fx_id == "FX-2026-02-10-EUR-USD"

    with pytest.raises(ValidationError):
        FxRateRow(
            date="2026-02-10",
            source_currency="EUR",
            target_currency="USD",
            rate="-1.05",
        )

    with pytest.raises(ValidationError):
        FxRateRow(
            date="2026-02-10",
            source_currency="EUR",
            target_currency="USD",
            rate="not_a_decimal",
        )


def test_closed_truth_and_provenance_models() -> None:
    issue = TruthIssue(
        issue_id="ISS-001",
        type=IssueType.DUPLICATE_INVOICE,
        affected_ids=["L-0001", "L-0042", "INV-2026-001"],
        expected_amount_minor=150000,
        currency="USD",
        valid_evidence_sets=[["DOC-INV-001", "DOC-INV-019", "L-0001", "L-0042"]],
    )
    truth_file = TruthFile(episode_id="ep_train_001", issues=[issue])
    assert truth_file.schema_version == SCHEMA_VERSION
    assert len(truth_file.issues) == 1

    with pytest.raises(ValidationError):
        TruthIssue(
            issue_id="ISS-002",
            type=IssueType.FX_CONVERSION_ERROR,
            affected_ids=["L-0005"],
            expected_amount_minor=500,
            currency="USD",
            valid_evidence_sets=[[]],
        )

    event = ProvenanceEvent(
        event_id="EVT-001",
        stage="error_injection",
        issue_id="ISS-001",
        issue_type=IssueType.DUPLICATE_INVOICE,
        description="Duplicated invoice INV-2026-001 into ledger row L-0042",
        before_state={"ledger_count": 249},
        after_state={"ledger_count": 250},
        affected_ids=["L-0042", "DOC-INV-019"],
    )
    prov_file = ProvenanceFile(
        episode_id="ep_train_001",
        seed=42,
        generator_config={"num_invoices": 40},
        events=[event],
    )
    assert prov_file.seed == 42
    assert prov_file.events[0].issue_type == IssueType.DUPLICATE_INVOICE


def test_findings_submission_limits_and_validation() -> None:
    finding = Finding(
        type=IssueType.PERIOD_CUTOFF_ERROR,
        affected_ids=["L-0099", "DOC-INV-012"],
        claimed_amount_minor=45000,
        currency="EUR",
        evidence_ids=["L-0099", "DOC-INV-012"],
        explanation="Invoice issued in April 2026 was booked into Q1 ledger.",
    )
    submission = FindingsSubmission(episode_id="ep_train_001", findings=[finding])
    assert len(submission.findings) == 1

    # Max 10 evidence IDs per finding
    with pytest.raises(ValidationError):
        Finding(
            type=IssueType.DUPLICATE_INVOICE,
            affected_ids=["L-0001"],
            claimed_amount_minor=100,
            currency="USD",
            evidence_ids=[f"L-{i:04d}" for i in range(11)],
            explanation="Too many evidence IDs",
        )

    # Max 1000 chars explanation
    with pytest.raises(ValidationError):
        Finding(
            type=IssueType.DUPLICATE_INVOICE,
            affected_ids=["L-0001"],
            claimed_amount_minor=100,
            currency="USD",
            evidence_ids=["L-0001"],
            explanation="x" * 1001,
        )

    # Max 30 findings per submission
    with pytest.raises(ValidationError):
        FindingsSubmission(
            episode_id="ep_train_001",
            findings=[finding for _ in range(31)],
        )


def test_action_discriminated_union() -> None:
    a1 = parse_action({"action": "list_records", "table": "ledger", "cursor": None, "limit": 50})
    assert isinstance(a1, ListRecordsAction)
    assert a1.table == "ledger"

    a2 = parse_action({"action": "get_record", "table": "fx", "id": "FX-2026-02-10-EUR-USD"})
    assert isinstance(a2, GetRecordAction)

    a3 = parse_action({"action": "get_document", "doc_id": "DOC-INV-001"})
    assert isinstance(a3, GetDocumentAction)

    a4 = parse_action({"action": "calculator", "expression": "(150000 - 125000) * 1.085"})
    assert isinstance(a4, CalculatorAction)

    a5 = parse_action(
        {
            "action": "submit",
            "submission": {
                "episode_id": "ep_train_001",
                "findings": [
                    {
                        "type": "fx_conversion_error",
                        "affected_ids": ["L-0010"],
                        "claimed_amount_minor": 1250,
                        "currency": "USD",
                        "evidence_ids": ["L-0010", "DOC-INV-003", "FX-2026-02-10-EUR-USD"],
                        "explanation": "Wrong FX date rate applied.",
                    }
                ],
            },
        }
    )
    assert isinstance(a5, SubmitAction)
    assert a5.submission.findings[0].type == IssueType.FX_CONVERSION_ERROR

    # Limit > 50 must fail
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"action": "list_records", "table": "ledger", "limit": 51})

    # Unknown action type must fail
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"action": "execute_sql", "query": "SELECT * FROM ledger"})

    # Extra field in action must fail
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(
            {"action": "get_document", "doc_id": "DOC-INV-001", "extra_param": True}
        )

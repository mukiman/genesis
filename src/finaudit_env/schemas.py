"""Shared Pydantic v2 schemas and strict data contracts for finaudit_env.

Defines open episode files (`Manifest`, `PolicyConfig`, `LedgerRow`, `LineItem`,
`InvoiceDoc`, `PaymentDoc`, `FxRateRow`), closed evaluator files (`TruthIssue`,
`TruthFile`, `ProvenanceEvent`, `ProvenanceFile`), and agent interaction contracts
(`Finding`, `FindingsSubmission`, `Action`).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

SCHEMA_VERSION: str = "1.0.0"

CURRENCY_EXPONENTS: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "JPY": 0,
    "CHF": 2,
}

ValidTable = Literal["ledger", "invoices", "payments", "fx"]
RoundingMode = Literal["HALF_EVEN", "HALF_UP", "DOWN", "UP"]
FxRateDateRule = Literal["transaction_date", "invoice_issue_date", "previous_business_day"]


class IssueType(str, Enum):
    """Canonical enumeration of financial audit error types."""

    DUPLICATE_INVOICE = "duplicate_invoice"
    UNALLOCATED_OR_PARTIAL_PAYMENT = "unallocated_or_partial_payment"
    FX_CONVERSION_ERROR = "fx_conversion_error"
    PERIOD_CUTOFF_ERROR = "period_cutoff_error"


class StrictBaseModel(BaseModel):
    """Base Pydantic v2 model with strict extra='forbid' validation."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


def _validate_currency_code(currency: str) -> str:
    if currency not in CURRENCY_EXPONENTS:
        allowed = ", ".join(sorted(CURRENCY_EXPONENTS.keys()))
        raise ValueError(f"Unsupported currency '{currency}'. Allowed currencies: {allowed}")
    return currency


# ---------------------------------------------------------------------------
# Open Episode Models (visible to the agent via narrow read-only tools)
# ---------------------------------------------------------------------------


class PolicyConfig(StrictBaseModel):
    """Company accounting policy configuration (`policy.json`)."""

    fx_rate_date_rule: FxRateDateRule = "transaction_date"
    period_start: str = Field(..., min_length=10, description="Reporting period start (ISO-8601 date or timestamp)")
    period_end: str = Field(..., min_length=10, description="Reporting period end (ISO-8601 date or timestamp)")
    timezone: str = Field(default="UTC", min_length=1, description="Company reporting timezone")
    rounding_mode: RoundingMode = "HALF_EVEN"
    amount_tolerance_minor: int = Field(default=1, ge=0, description="Allowed minor-unit rounding tolerance")


class Manifest(StrictBaseModel):
    """Episode metadata manifest (`manifest.json`)."""

    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    episode_id: str = Field(..., min_length=1)
    company_name: str = Field(default="Synthetic Corp", min_length=1)
    base_currency: str = Field(..., min_length=3, max_length=3)
    period_start: str = Field(..., min_length=10)
    period_end: str = Field(..., min_length=10)
    timezone: str = Field(default="UTC", min_length=1)
    rounding_mode: RoundingMode = "HALF_EVEN"
    currency_exponents: dict[str, int] = Field(default_factory=lambda: dict(CURRENCY_EXPONENTS))
    ledger_rows_count: int = Field(..., ge=0)
    invoices_count: int = Field(..., ge=0)
    payments_count: int = Field(..., ge=0)
    fx_rates_count: int = Field(..., ge=0)

    @field_validator("base_currency")
    @classmethod
    def validate_base_currency(cls, value: str) -> str:
        return _validate_currency_code(value)


class LedgerRow(StrictBaseModel):
    """Single general ledger row (`ledger.csv`)."""

    row_id: str = Field(..., min_length=1, description="Canonical ledger row ID, e.g. L-0001")
    timestamp: str = Field(..., min_length=10, description="ISO-8601 timestamp with timezone offset")
    account: str = Field(..., min_length=1, description="Account code or name")
    debit_minor: int = Field(..., ge=0, description="Debit amount in integer minor units")
    credit_minor: int = Field(..., ge=0, description="Credit amount in integer minor units")
    currency: str = Field(..., min_length=3, max_length=3)
    entity_id: str = Field(..., min_length=1, description="Counterparty entity ID, e.g. ENT-001")
    invoice_id: str | None = Field(default=None, description="Optional linked invoice ID, e.g. INV-2026-001")
    payment_id: str | None = Field(default=None, description="Optional linked payment ID, e.g. PAY-2026-001")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return _validate_currency_code(value)


class LineItem(StrictBaseModel):
    """Line item inside an invoice document."""

    description: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    unit_price_minor: int = Field(..., ge=0)
    amount_minor: int = Field(..., ge=0)


class InvoiceDoc(StrictBaseModel):
    """Invoice document (`invoices.jsonl`)."""

    doc_id: str = Field(..., min_length=1, description="Document ID, e.g. DOC-INV-001")
    invoice_id: str = Field(..., min_length=1, description="Invoice business ID, e.g. INV-2026-001")
    entity_id: str = Field(..., min_length=1, description="Counterparty entity ID, e.g. ENT-001")
    issued_at: str = Field(..., min_length=10, description="ISO-8601 timestamp with offset")
    due_at: str = Field(..., min_length=10, description="ISO-8601 timestamp with offset")
    currency: str = Field(..., min_length=3, max_length=3)
    total_minor: int = Field(..., ge=0)
    line_items: list[LineItem] = Field(..., min_length=1)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return _validate_currency_code(value)


class PaymentDoc(StrictBaseModel):
    """Payment document (`payments.jsonl`)."""

    doc_id: str = Field(..., min_length=1, description="Document ID, e.g. DOC-PAY-001")
    payment_id: str = Field(..., min_length=1, description="Payment business ID, e.g. PAY-2026-001")
    entity_id: str | None = Field(default=None, description="Optional counterparty entity ID")
    timestamp: str = Field(..., min_length=10, description="ISO-8601 timestamp with offset")
    currency: str = Field(..., min_length=3, max_length=3)
    amount_minor: int = Field(..., ge=0)
    invoice_refs: list[str] = Field(default_factory=list, description="Referenced invoice IDs")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return _validate_currency_code(value)


class FxRateRow(StrictBaseModel):
    """Foreign exchange rate row (`fx.csv`)."""

    fx_id: str | None = Field(
        default=None,
        description="Canonical FX record ID: FX-YYYY-MM-DD-SRC-TGT (auto-populated if omitted)",
    )
    date: str = Field(..., min_length=10, max_length=10, description="ISO date YYYY-MM-DD")
    source_currency: str = Field(..., min_length=3, max_length=3)
    target_currency: str = Field(..., min_length=3, max_length=3)
    rate: str = Field(..., min_length=1, description="Decimal exchange rate string, e.g. '1.0850'")

    @field_validator("source_currency", "target_currency")
    @classmethod
    def validate_currencies(cls, value: str) -> str:
        return _validate_currency_code(value)

    @field_validator("rate")
    @classmethod
    def validate_decimal_rate(cls, value: str) -> str:
        try:
            dec = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"FX rate must be a valid decimal string, got '{value}'") from exc
        if dec <= 0:
            raise ValueError(f"FX rate must be positive, got '{value}'")
        return value

    @model_validator(mode="after")
    def populate_fx_id(self) -> FxRateRow:
        expected_id = f"FX-{self.date}-{self.source_currency}-{self.target_currency}"
        if self.fx_id is None:
            self.fx_id = expected_id
        return self


# ---------------------------------------------------------------------------
# Closed Evaluator Models (NEVER visible to the agent)
# ---------------------------------------------------------------------------


class TruthIssue(StrictBaseModel):
    """Single ground-truth financial issue inside `truth.json`."""

    issue_id: str = Field(..., min_length=1, description="Ground truth issue ID, e.g. ISS-001")
    type: IssueType
    affected_ids: list[str] = Field(..., min_length=1, description="Canonical affected record/document IDs")
    expected_amount_minor: int = Field(..., ge=0, description="Discrepancy amount in minor units")
    currency: str = Field(..., min_length=3, max_length=3)
    valid_evidence_sets: list[list[str]] = Field(
        ...,
        min_length=1,
        description="At least one complete minimal set of evidence IDs sufficient to verify the issue",
    )
    tolerance_minor: int = Field(default=1, ge=0, description="Allowed minor-unit tolerance for matching")
    notes: str | None = Field(default=None, description="Evaluator-only diagnostic note")

    @model_validator(mode="before")
    @classmethod
    def normalize_truth_issue_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "valid_evidence_sets" not in d:
            if "allowed_evidence_sets" in d:
                d["valid_evidence_sets"] = d.pop("allowed_evidence_sets")
            elif "evidence_sets" in d:
                d["valid_evidence_sets"] = d.pop("evidence_sets")
        if "tolerance_minor" not in d and "amount_tolerance_minor" in d:
            d["tolerance_minor"] = d.pop("amount_tolerance_minor")
        return d

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return _validate_currency_code(value)

    @field_validator("valid_evidence_sets")
    @classmethod
    def validate_non_empty_evidence_sets(cls, value: list[list[str]]) -> list[list[str]]:
        for evidence_set in value:
            if not evidence_set:
                raise ValueError("Each valid evidence set in valid_evidence_sets must contain at least one ID")
        return value


class TruthFile(StrictBaseModel):
    """Top-level ground-truth file (`truth.json`)."""

    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    episode_id: str = Field(..., min_length=1)
    issues: list[TruthIssue] = Field(default_factory=list)


class ProvenanceEvent(StrictBaseModel):
    """Single causal generation or error-injection event inside `provenance.json`."""

    event_id: str = Field(..., min_length=1)
    stage: Literal["clean_generation", "distractor_injection", "error_injection", "transform"] = "error_injection"
    issue_id: str | None = Field(default=None, description="Linked TruthIssue.issue_id if stage=='error_injection'")
    issue_type: IssueType | None = None
    description: str = Field(..., min_length=1)
    before_state: dict[str, Any] = Field(default_factory=dict)
    after_state: dict[str, Any] = Field(default_factory=dict)
    affected_ids: list[str] = Field(default_factory=list)


class ProvenanceFile(StrictBaseModel):
    """Closed generation & causal mutation history (`provenance.json`)."""

    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    episode_id: str = Field(..., min_length=1)
    seed: int
    generator_config: dict[str, Any] = Field(default_factory=dict)
    pre_mutation_trial_balance_ok: bool = True
    events: list[ProvenanceEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output & Action Contracts (Agent <-> Environment interface)
# ---------------------------------------------------------------------------


class Finding(StrictBaseModel):
    """Single audit finding submitted by an agent."""

    type: IssueType
    affected_ids: list[str] = Field(..., min_length=1)
    claimed_amount_minor: int = Field(..., ge=0)
    currency: str = Field(..., min_length=3, max_length=3)
    evidence_ids: list[str] = Field(..., min_length=1, max_length=10)
    explanation: str = Field(..., min_length=1, max_length=1000)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return _validate_currency_code(value)


class FindingsSubmission(StrictBaseModel):
    """Payload for `submit` action (`findings.json` format, max 30 findings)."""

    episode_id: str = Field(..., min_length=1)
    findings: list[Finding] = Field(default_factory=list, max_length=30)


class ListRecordsAction(StrictBaseModel):
    """Action: paginate through an open table (`ledger`, `invoices`, `payments`, `fx`)."""

    action: Literal["list_records"] = "list_records"
    table: ValidTable
    cursor: str | None = Field(default=None, description="Opaque pagination cursor or string offset")
    limit: int = Field(default=50, ge=1, le=50, description="Page size (max 50 rows)")


class GetRecordAction(StrictBaseModel):
    """Action: fetch a single record by canonical ID from a table."""

    action: Literal["get_record"] = "get_record"
    table: ValidTable
    id: str = Field(..., min_length=1)


class GetDocumentAction(StrictBaseModel):
    """Action: fetch a primary invoice or payment document by `doc_id`."""

    action: Literal["get_document"] = "get_document"
    doc_id: str = Field(..., min_length=1)


class CalculatorAction(StrictBaseModel):
    """Action: evaluate a safe decimal arithmetic expression."""

    action: Literal["calculator"] = "calculator"
    expression: str = Field(..., min_length=1, max_length=500)


class SubmitAction(StrictBaseModel):
    """Action: submit final audit findings and terminate the episode."""

    action: Literal["submit"] = "submit"
    submission: FindingsSubmission


Action = Annotated[
    Union[
        ListRecordsAction,
        GetRecordAction,
        GetDocumentAction,
        CalculatorAction,
        SubmitAction,
    ],
    Field(discriminator="action"),
]

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(
    payload: dict[str, Any] | Action,
    *,
    default_episode_id: str | None = None,
) -> Action:
    """Validate and parse a raw dictionary or model into a discriminated `Action`."""
    if isinstance(payload, (ListRecordsAction, GetRecordAction, GetDocumentAction, CalculatorAction, SubmitAction)):
        return payload
    if isinstance(payload, dict):
        d = dict(payload)
        # Support {"tool": "...", "args": {...}} or {"name": "...", "arguments": {...}} wrappers
        if "action" not in d:
            for key_alias in ("tool", "name", "type"):
                if key_alias in d and isinstance(d[key_alias], str):
                    d["action"] = d.pop(key_alias)
                    break
        for args_alias in ("args", "arguments", "params"):
            if args_alias in d and isinstance(d[args_alias], dict):
                nested_args = d.pop(args_alias)
                for k, v in nested_args.items():
                    d.setdefault(k, v)

        act_name = d.get("action")
        if act_name == "list_records":
            if isinstance(d.get("cursor"), int) and not isinstance(d.get("cursor"), bool) and d["cursor"] >= 0:
                d["cursor"] = str(d["cursor"])
            if "limit" in d and d["limit"] is None:
                d.pop("limit")
        elif act_name == "get_record":
            if "id" not in d and "record_id" in d:
                d["id"] = d.pop("record_id")
        elif act_name == "get_document":
            if "doc_id" not in d:
                if "document_id" in d:
                    d["doc_id"] = d.pop("document_id")
                elif "id" in d:
                    d["doc_id"] = d.pop("id")
        elif act_name == "calculator":
            if "expression" not in d and "expr" in d:
                d["expression"] = d.pop("expr")
        elif act_name == "submit":
            if "submission" not in d:
                for sub_alias in ("findings.json", "findings_json"):
                    if sub_alias in d and isinstance(d[sub_alias], dict):
                        d["submission"] = d.pop(sub_alias)
                        break
            if "submission" not in d and "findings" in d:
                raw_findings_val = d.pop("findings")
                if isinstance(raw_findings_val, dict) and "findings" in raw_findings_val:
                    sub_dict = dict(raw_findings_val)
                else:
                    sub_dict = {"findings": raw_findings_val}
                if "episode_id" in d:
                    sub_dict.setdefault("episode_id", d.pop("episode_id"))
                elif default_episode_id:
                    sub_dict.setdefault("episode_id", default_episode_id)
                d["submission"] = sub_dict
            elif isinstance(d.get("submission"), dict):
                sub_copy = dict(d["submission"])
                if "episode_id" not in sub_copy and "findings" in sub_copy and default_episode_id:
                    sub_copy["episode_id"] = default_episode_id
                d["submission"] = sub_copy
        return ACTION_ADAPTER.validate_python(d)
    return ACTION_ADAPTER.validate_python(payload)


__all__ = [
    "SCHEMA_VERSION",
    "CURRENCY_EXPONENTS",
    "IssueType",
    "Manifest",
    "PolicyConfig",
    "LedgerRow",
    "LineItem",
    "InvoiceDoc",
    "PaymentDoc",
    "FxRateRow",
    "TruthIssue",
    "TruthFile",
    "ProvenanceEvent",
    "ProvenanceFile",
    "Finding",
    "FindingsSubmission",
    "ListRecordsAction",
    "GetRecordAction",
    "GetDocumentAction",
    "CalculatorAction",
    "SubmitAction",
    "Action",
    "ACTION_ADAPTER",
    "parse_action",
]

"""Contract and security unit tests for `EpisodeDataStore` and `safe_calculate`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from finaudit_env.env.calculator import CalculatorError, evaluate_expression, safe_calculate
from finaudit_env.env.datastore import EpisodeDataStore


def _write_sample_episode(ep_dir: Path, *, num_ledger_rows: int = 65) -> None:
    """Create a complete synthetic episode directory with 6 open files plus closed truth/provenance files."""
    ep_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0.0",
        "episode_id": "ep_test_001",
        "company_name": "Test Audit Corp",
        "base_currency": "USD",
        "period_start": "2026-03-01",
        "period_end": "2026-03-31",
        "timezone": "UTC",
        "rounding_mode": "HALF_EVEN",
        "ledger_rows_count": num_ledger_rows,
        "invoices_count": 2,
        "payments_count": 1,
        "fx_rates_count": 2,
    }
    (ep_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    policy = {
        "fx_rate_date_rule": "transaction_date",
        "period_start": "2026-03-01",
        "period_end": "2026-03-31",
        "timezone": "UTC",
        "rounding_mode": "HALF_EVEN",
        "amount_tolerance_minor": 1,
    }
    (ep_dir / "policy.json").write_text(json.dumps(policy), encoding="utf-8")

    with (ep_dir / "ledger.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_id",
                "timestamp",
                "account",
                "debit_minor",
                "credit_minor",
                "currency",
                "entity_id",
                "invoice_id",
                "payment_id",
            ]
        )
        for idx in range(1, num_ledger_rows + 1):
            inv_id = "INV-2026-001" if idx == 1 else ""
            pay_id = "PAY-2026-001" if idx == 2 else ""
            writer.writerow(
                [
                    f"L-{idx:04d}",
                    "2026-03-10T12:00:00+00:00",
                    "1100-AR" if idx % 2 == 1 else "4000-REV",
                    10000 if idx % 2 == 1 else 0,
                    0 if idx % 2 == 1 else 10000,
                    "USD",
                    "ENT-001",
                    inv_id,
                    pay_id,
                ]
            )

    invoices = [
        {
            "doc_id": "DOC-INV-001",
            "invoice_id": "INV-2026-001",
            "entity_id": "ENT-001",
            "issued_at": "2026-03-10T09:00:00+00:00",
            "due_at": "2026-03-25T09:00:00+00:00",
            "currency": "USD",
            "total_minor": 10000,
            "line_items": [
                {
                    "description": "Cloud audit subscription",
                    "quantity": 1,
                    "unit_price_minor": 10000,
                    "amount_minor": 10000,
                }
            ],
        },
        {
            "doc_id": "DOC-INV-002",
            "invoice_id": "INV-2026-002",
            "entity_id": "ENT-002",
            "issued_at": "2026-03-15T09:00:00+00:00",
            "due_at": "2026-03-30T09:00:00+00:00",
            "currency": "EUR",
            "total_minor": 20000,
            "line_items": [
                {
                    "description": "Advisory fee",
                    "quantity": 2,
                    "unit_price_minor": 10000,
                    "amount_minor": 20000,
                }
            ],
        },
    ]
    with (ep_dir / "invoices.jsonl").open("w", encoding="utf-8") as f:
        for inv in invoices:
            f.write(json.dumps(inv) + "\n")

    payments = [
        {
            "doc_id": "DOC-PAY-001",
            "payment_id": "PAY-2026-001",
            "entity_id": "ENT-001",
            "timestamp": "2026-03-12T15:00:00+00:00",
            "currency": "USD",
            "amount_minor": 10000,
            "invoice_refs": ["INV-2026-001"],
        }
    ]
    with (ep_dir / "payments.jsonl").open("w", encoding="utf-8") as f:
        for pay in payments:
            f.write(json.dumps(pay) + "\n")

    with (ep_dir / "fx.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "source_currency", "target_currency", "rate"])
        writer.writerow(["2026-03-10", "EUR", "USD", "1.0850"])
        writer.writerow(["2026-03-15", "EUR", "USD", "1.0925"])

    # Closed files placed in the same directory to verify they are completely ignored by datastore
    (ep_dir / "truth.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "episode_id": "ep_test_001",
                "issues": [
                    {
                        "issue_id": "ISS-SECRET-999",
                        "type": "duplicate_invoice",
                        "affected_ids": ["L-0001"],
                        "expected_amount_minor": 10000,
                        "currency": "USD",
                        "valid_evidence_sets": [["DOC-INV-001", "L-0001"]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (ep_dir / "provenance.json").write_text(
        json.dumps({"episode_id": "ep_test_001", "seed": 424242, "events": []}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# EpisodeDataStore Tests
# ---------------------------------------------------------------------------


def test_datastore_loads_open_files_and_ignores_closed_files(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=65)
    store = EpisodeDataStore(episode_dir=tmp_path)

    manifest = store.get_manifest()
    policy = store.get_policy()
    assert manifest["episode_id"] == "ep_test_001"
    assert policy["fx_rate_date_rule"] == "transaction_date"

    # Verify closed truth/provenance IDs and seed are not loaded into known IDs or attributes
    known_ids = store.all_known_ids()
    assert "ISS-SECRET-999" not in known_ids
    assert not hasattr(store, "_truth")
    assert not hasattr(store, "_provenance")

    # Verify CSV empty strings become None
    first_row = store.get_record("ledger", "L-0001")
    assert first_row is not None
    assert first_row["invoice_id"] == "INV-2026-001"
    assert first_row["payment_id"] is None

    third_row = store.get_record("ledger", "L-0003")
    assert third_row is not None
    assert third_row["invoice_id"] is None
    assert third_row["payment_id"] is None


def test_datastore_pagination_and_cursor_stability(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=65)
    store = EpisodeDataStore(episode_dir=tmp_path)

    page1 = store.list_records("ledger", cursor=None, limit=50)
    assert len(page1["items"]) == 50
    assert page1["items"][0]["row_id"] == "L-0001"
    assert page1["items"][-1]["row_id"] == "L-0050"
    assert page1["next_cursor"] == "50"
    assert page1["total_count"] == 65

    # Repeating the same query yields the exact same items and cursor
    page1_repeat = store.list_records("ledger", cursor="0", limit=50)
    assert page1_repeat == page1

    page2 = store.list_records("ledger", cursor=page1["next_cursor"], limit=50)
    assert len(page2["items"]) == 15
    assert page2["items"][0]["row_id"] == "L-0051"
    assert page2["items"][-1]["row_id"] == "L-0065"
    assert page2["next_cursor"] is None


def test_datastore_rejects_invalid_limits_tables_and_cursors(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    store = EpisodeDataStore(episode_dir=tmp_path)

    with pytest.raises(ValueError, match="Invalid limit"):
        store.list_records("ledger", limit=51)

    with pytest.raises(ValueError, match="Invalid limit"):
        store.list_records("ledger", limit=0)

    with pytest.raises(ValueError, match="Unknown table"):
        store.list_records("truth")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid pagination cursor"):
        store.list_records("ledger", cursor="-5")

    with pytest.raises(ValueError, match="Invalid pagination cursor"):
        store.list_records("ledger", cursor="abc")

    with pytest.raises(ValueError, match="exceeds table size"):
        store.list_records("ledger", cursor="999")


def test_datastore_get_record_and_get_document_across_tables(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    store = EpisodeDataStore(episode_dir=tmp_path)

    # Ledger by row_id
    assert store.get_record("ledger", "L-0002")["payment_id"] == "PAY-2026-001"
    assert store.get_record("ledger", "L-9999") is None

    # Invoices by invoice_id (not doc_id)
    inv_rec = store.get_record("invoices", "INV-2026-002")
    assert inv_rec is not None
    assert inv_rec["doc_id"] == "DOC-INV-002"
    assert store.get_record("invoices", "DOC-INV-002") is None

    # Payments by payment_id
    pay_rec = store.get_record("payments", "PAY-2026-001")
    assert pay_rec is not None
    assert pay_rec["doc_id"] == "DOC-PAY-001"

    # FX by canonical fx_id
    fx_rec = store.get_record("fx", "FX-2026-03-10-EUR-USD")
    assert fx_rec is not None
    assert fx_rec["rate"] == "1.0850"

    # Documents by doc_id
    doc_inv = store.get_document("DOC-INV-001")
    assert doc_inv is not None
    assert doc_inv["invoice_id"] == "INV-2026-001"

    doc_pay = store.get_document("DOC-PAY-001")
    assert doc_pay is not None
    assert doc_pay["payment_id"] == "PAY-2026-001"

    assert store.get_document("INV-2026-001") is None


def test_datastore_returns_defensive_copies(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=5)
    store = EpisodeDataStore(episode_dir=tmp_path)

    rec = store.get_record("ledger", "L-0001")
    assert rec is not None
    rec["debit_minor"] = 999999999

    fresh = store.get_record("ledger", "L-0001")
    assert fresh is not None
    assert fresh["debit_minor"] == 10000


# ---------------------------------------------------------------------------
# Safe Decimal AST Calculator Tests
# ---------------------------------------------------------------------------


def test_calculator_exact_decimal_arithmetic() -> None:
    # Classic float trap: 0.1 + 0.2 == 0.3 exactly in Decimal
    res = safe_calculate("0.1 + 0.2")
    assert res["result"] == "0.3"

    # Complex expression with parentheses and unary minus
    assert evaluate_expression("(125000 * 1.0850) - 135625") == "0"
    assert evaluate_expression("-500 + (200 * 3) / 4") == "-350"
    assert evaluate_expression("+42.500") == "42.5"


@pytest.mark.parametrize(
    "bad_expr",
    [
        "",
        "   ",
        "10 / 0",
        "100 / (5 - 5)",
        "2 ** 10",
        "10 // 3",
        "10 % 3",
        "__import__('os').system('echo hacked')",
        "open('truth.json').read()",
        "a + b",
        "True + 1",
        "'hello' + 'world'",
        "[1, 2, 3]",
        "1 + " * 300 + "1",
    ],
)
def test_calculator_rejects_unsafe_or_invalid_expressions(bad_expr: str) -> None:
    with pytest.raises(CalculatorError):
        safe_calculate(bad_expr)

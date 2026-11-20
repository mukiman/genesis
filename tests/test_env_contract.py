"""25+ Jury verification tests for RL Contract & Isolation (`test_env_contract.py`).

Covers all Section 04 and Section 10 ("Контракт и изоляция" - 25 pts) requirements:
- EpisodeDataStore read-only access, 50-row pagination limit, cursor stability, defensive copying.
- Safe Decimal AST calculator (no float errors, blocks code execution, division by zero, AST depth).
- FinAuditEnv `reset(seed, config)` and `step(action)` Gymnasium-style contract.
- Reproducibility across identical resets and action sequences.
- State isolation across consecutive resets.
- All 5 actions (`list_records`, `get_record`, `get_document`, `calculator`, `submit`).
- Unknown ID / invalid action returns bounded error and consumes 1 step.
- Invalid `submit` structure immediately terminates with reward -1.0.
- Duplicate `submit` after termination is rejected without altering evaluation.
- 100-step limit without submit sets `truncated=True` and evaluates empty submission.
- Timeout limit sets `truncated=True` and evaluates empty submission.
- Zero intermediate reward during inspection steps.
- Open `train` mode vs closed `eval` mode reward/diagnostic isolation.
- Ground-truth `leak_checker` across observations, info, errors, and `trajectories.jsonl`.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from finaudit_env.env.calculator import CalculatorError, evaluate_expression, safe_calculate
from finaudit_env.env.core import FinAuditEnv
from finaudit_env.env.datastore import EpisodeDataStore
from finaudit_env.env.isolated_client import IsolatedFinAuditEnv
from finaudit_env.env.leak_checker import (
    assert_no_leaks,
    check_observation_for_leaks,
    scan_payload_for_leaks,
    verify_trajectory_file,
)


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

    # Closed files placed in the same directory to verify they are completely isolated from observations
    (ep_dir / "truth.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "episode_id": "ep_test_001",
                "issues": [
                    {
                        "issue_id": "ISS-SECRET-999",
                        "type": "duplicate_invoice",
                        "affected_ids": ["L-0001", "INV-2026-001"],
                        "expected_amount_minor": 10000,
                        "currency": "USD",
                        "valid_evidence_sets": [["DOC-INV-001", "L-0001"]],
                        "notes": "HIDDEN_NOTE_DO_NOT_LEAK",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (ep_dir / "provenance.json").write_text(
        json.dumps({"episode_id": "ep_test_001", "seed": 987654321, "events": []}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# EpisodeDataStore Tests
# ---------------------------------------------------------------------------


def test_01_datastore_loads_open_files_and_ignores_closed_files(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=65)
    store = EpisodeDataStore(episode_dir=tmp_path)

    manifest = store.get_manifest()
    policy = store.get_policy()
    assert manifest["episode_id"] == "ep_test_001"
    assert policy["fx_rate_date_rule"] == "transaction_date"

    known_ids = store.all_known_ids()
    assert "ISS-SECRET-999" not in known_ids
    assert not hasattr(store, "_truth")
    assert not hasattr(store, "_provenance")

    first_row = store.get_record("ledger", "L-0001")
    assert first_row is not None
    assert first_row["invoice_id"] == "INV-2026-001"
    assert first_row["payment_id"] is None


def test_02_datastore_pagination_and_cursor_stability(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=65)
    store = EpisodeDataStore(episode_dir=tmp_path)

    page1 = store.list_records("ledger", cursor=None, limit=50)
    assert len(page1["items"]) == 50
    assert page1["items"][0]["row_id"] == "L-0001"
    assert page1["items"][-1]["row_id"] == "L-0050"
    assert page1["next_cursor"] == "50"
    assert page1["total_count"] == 65

    page1_repeat = store.list_records("ledger", cursor="0", limit=50)
    assert page1_repeat == page1

    page2 = store.list_records("ledger", cursor=page1["next_cursor"], limit=50)
    assert len(page2["items"]) == 15
    assert page2["items"][0]["row_id"] == "L-0051"
    assert page2["items"][-1]["row_id"] == "L-0065"
    assert page2["next_cursor"] is None


def test_03_datastore_rejects_invalid_limits_tables_and_cursors(tmp_path: Path) -> None:
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


def test_04_datastore_get_record_and_get_document_across_tables(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    store = EpisodeDataStore(episode_dir=tmp_path)

    assert store.get_record("ledger", "L-0002")["payment_id"] == "PAY-2026-001"
    assert store.get_record("ledger", "L-9999") is None

    inv_rec = store.get_record("invoices", "INV-2026-002")
    assert inv_rec is not None
    assert inv_rec["doc_id"] == "DOC-INV-002"
    assert store.get_record("invoices", "DOC-INV-002") is None

    pay_rec = store.get_record("payments", "PAY-2026-001")
    assert pay_rec is not None
    assert pay_rec["doc_id"] == "DOC-PAY-001"

    fx_rec = store.get_record("fx", "FX-2026-03-10-EUR-USD")
    assert fx_rec is not None
    assert fx_rec["rate"] == "1.0850"

    assert store.get_document("DOC-INV-001")["invoice_id"] == "INV-2026-001"
    assert store.get_document("DOC-PAY-001")["payment_id"] == "PAY-2026-001"
    assert store.get_document("INV-2026-001") is None


def test_05_datastore_returns_defensive_copies(tmp_path: Path) -> None:
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


def test_06_calculator_exact_decimal_arithmetic() -> None:
    res = safe_calculate("0.1 + 0.2")
    assert res["result"] == "0.3"
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
def test_07_calculator_rejects_unsafe_or_invalid_expressions(bad_expr: str) -> None:
    with pytest.raises(CalculatorError):
        safe_calculate(bad_expr)


# ---------------------------------------------------------------------------
# FinAuditEnv Contract & Isolation Jury Tests
# ---------------------------------------------------------------------------


def test_08_env_reset_returns_initial_observation_and_isolates_state(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)

    obs, info = env.reset(seed=101)
    assert "task" in obs
    assert obs["manifest"]["episode_id"] == "ep_test_001"
    assert obs["policy"]["fx_rate_date_rule"] == "transaction_date"
    assert obs["steps_remaining"] == 100
    assert info == {"episode_id": "ep_test_001", "step": 0, "steps_remaining": 100}

    # Take a step, then reset again and verify step counter returns to 0
    env.step({"action": "list_records", "table": "ledger", "limit": 5})
    assert len(env.get_trajectory()) == 1

    obs2, info2 = env.reset(seed=101)
    assert obs2 == obs
    assert info2["step"] == 0
    assert len(env.get_trajectory()) == 0


def test_09_env_step_before_reset_raises_runtime_error(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=5)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    with pytest.raises(RuntimeError, match="reset"):
        env.step({"action": "list_records", "table": "ledger"})


def test_10_env_all_five_actions_work_end_to_end(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    env.reset(seed=42)

    # 1. list_records
    obs1, r1, term1, trunc1, _ = env.step({"action": "list_records", "table": "ledger", "limit": 5})
    assert r1 == 0.0 and not term1 and not trunc1
    assert len(obs1["result"]["items"]) == 5
    assert obs1["steps_remaining"] == 99

    # 2. get_record
    obs2, r2, term2, trunc2, _ = env.step({"action": "get_record", "table": "ledger", "id": "L-0001"})
    assert r2 == 0.0 and not term2 and not trunc2
    assert obs2["result"]["row_id"] == "L-0001"
    assert obs2["steps_remaining"] == 98

    # 3. get_document
    obs3, r3, term3, trunc3, _ = env.step({"action": "get_document", "doc_id": "DOC-INV-001"})
    assert r3 == 0.0 and not term3 and not trunc3
    assert obs3["result"]["invoice_id"] == "INV-2026-001"
    assert obs3["steps_remaining"] == 97

    # 4. calculator
    obs4, r4, term4, trunc4, _ = env.step({"action": "calculator", "expression": "10000 * 1.0850"})
    assert r4 == 0.0 and not term4 and not trunc4
    assert obs4["result"]["result"] == "10850"
    assert obs4["steps_remaining"] == 96

    # 5. submit perfect finding (4 inspection steps occurred before submit)
    obs5, r5, term5, trunc5, info5 = env.step(
        {
            "action": "submit",
            "submission": {
                "episode_id": "ep_test_001",
                "findings": [
                    {
                        "type": "duplicate_invoice",
                        "affected_ids": ["L-0001", "INV-2026-001"],
                        "claimed_amount_minor": 10000,
                        "currency": "USD",
                        "evidence_ids": ["DOC-INV-001", "L-0001"],
                        "explanation": "Duplicate invoice posted in L-0001.",
                    }
                ],
            },
        }
    )
    assert term5 is True and trunc5 is False
    assert obs5["result"]["status"] == "submitted"
    assert r5 == pytest.approx(1.0 - 0.002 * 4)
    assert info5["evaluation"]["f1"] == 1.0


def test_11_env_reproducibility_with_same_seed_and_actions(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    actions = [
        {"action": "list_records", "table": "invoices", "limit": 2},
        {"action": "get_record", "table": "fx", "id": "FX-2026-03-10-EUR-USD"},
        {"action": "calculator", "expression": "20000 * 1.085"},
        {"action": "submit", "submission": {"episode_id": "ep_test_001", "findings": []}},
    ]

    env1 = FinAuditEnv(mode="train", episode_dir=tmp_path)
    obs_init_1, _ = env1.reset(seed=777)
    traj1 = [env1.step(a) for a in actions]

    env2 = FinAuditEnv(mode="train", episode_dir=tmp_path)
    obs_init_2, _ = env2.reset(seed=777)
    traj2 = [env2.step(a) for a in actions]

    assert obs_init_1 == obs_init_2
    assert traj1 == traj2


def test_12_env_unknown_id_and_unknown_action_consume_step_with_bounded_error(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    env.reset(seed=1)

    # Unknown record ID
    obs1, r1, term1, trunc1, _ = env.step({"action": "get_record", "table": "ledger", "id": "L-9999"})
    assert "error" in obs1 and "not found" in obs1["error"]
    assert r1 == 0.0 and not term1 and not trunc1
    assert obs1["steps_remaining"] == 99

    # Unknown document ID
    obs2, r2, term2, trunc2, _ = env.step({"action": "get_document", "doc_id": "DOC-UNKNOWN"})
    assert "error" in obs2 and "not found" in obs2["error"]
    assert r2 == 0.0 and not term2 and not trunc2
    assert obs2["steps_remaining"] == 98

    # Unknown action name
    obs3, r3, term3, trunc3, _ = env.step({"action": "run_sql", "query": "SELECT * FROM truth"})
    assert "error" in obs3
    assert r3 == 0.0 and not term3 and not trunc3
    assert obs3["steps_remaining"] == 97


def test_13_env_invalid_submit_structure_immediately_terminates_with_minus_one(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        {
            "action": "submit",
            "submission": {"bad_field": "malformed"},
        }
    )
    assert terminated is True
    assert truncated is False
    assert reward == -1.0
    assert "error" in obs
    assert info["evaluation"]["invalid_structure"] is True


def test_14_env_duplicate_submit_after_termination_is_rejected(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    env.reset(seed=1)

    # First submit: empty findings (F1=0, reward=-1.0)
    _, r1, term1, _, _ = env.step(
        {"action": "submit", "submission": {"episode_id": "ep_test_001", "findings": []}}
    )
    assert term1 is True and r1 == -1.0

    # Second submit: attempts to overwrite with perfect finding
    obs2, r2, term2, _, _ = env.step(
        {
            "action": "submit",
            "submission": {
                "episode_id": "ep_test_001",
                "findings": [
                    {
                        "type": "duplicate_invoice",
                        "affected_ids": ["L-0001", "INV-2026-001"],
                        "claimed_amount_minor": 10000,
                        "currency": "USD",
                        "evidence_ids": ["DOC-INV-001", "L-0001"],
                        "explanation": "Late overwrite attempt.",
                    }
                ],
            },
        }
    )
    assert "already finished" in obs2["error"]
    assert r2 == 0.0
    assert term2 is True
    # Internal evaluation must still reflect the first submission (TP=0)
    assert env.get_last_evaluation()["tp"] == 0


def test_15_env_100_step_limit_truncates_with_empty_answer(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    env.reset(seed=1)

    for step_idx in range(1, 100):
        obs, reward, terminated, truncated, _ = env.step(
            {"action": "get_record", "table": "ledger", "id": "L-0001"}
        )
        assert not terminated and not truncated
        assert reward == 0.0
        assert obs["steps_remaining"] == 100 - step_idx

    # 100th step is still an inspection step -> budget exhausted -> truncated=True
    obs100, reward100, terminated100, truncated100, _ = env.step(
        {"action": "get_record", "table": "ledger", "id": "L-0001"}
    )
    assert terminated100 is False
    assert truncated100 is True
    assert obs100["steps_remaining"] == 0
    assert reward100 == -1.0  # empty submission on erroneous episode
    assert env.get_last_evaluation()["truncation_reason"] == "step_limit_reached"


def test_16_env_timeout_truncates_with_empty_answer(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", timeout_seconds=0.01, episode_dir=tmp_path)
    env.reset(seed=1)

    time.sleep(0.03)
    obs, reward, terminated, truncated, _ = env.step(
        {"action": "list_records", "table": "ledger", "limit": 5}
    )
    assert terminated is False
    assert truncated is True
    assert "time limit exceeded" in obs["error"]
    assert reward == -1.0
    assert env.get_last_evaluation()["truncation_reason"] == "timeout"


def test_17_env_eval_mode_hides_terminal_reward_and_diagnostics_from_agent(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="eval", episode_dir=tmp_path)
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        {
            "action": "submit",
            "submission": {
                "episode_id": "ep_test_001",
                "findings": [
                    {
                        "type": "duplicate_invoice",
                        "affected_ids": ["L-0001", "INV-2026-001"],
                        "claimed_amount_minor": 10000,
                        "currency": "USD",
                        "evidence_ids": ["DOC-INV-001", "L-0001"],
                        "explanation": "Duplicate invoice in eval mode.",
                    }
                ],
            },
        }
    )
    assert terminated is True and not truncated
    # In eval mode, policy receives 0.0 reward and no evaluation dict in info
    assert reward == 0.0
    assert "evaluation" not in info
    assert "tp" not in json.dumps(obs)
    # Evaluator still holds the true score internally
    eval_res = env.get_last_evaluation()
    assert eval_res is not None
    assert eval_res["f1"] == 1.0
    assert eval_res["reward"] == 1.0


def test_18_trajectory_dump_and_leak_checker_validation(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)
    env = FinAuditEnv(mode="train", episode_dir=tmp_path)
    obs0, info0 = env.reset(seed=987654321)

    assert check_observation_for_leaks(obs0, info0) == []

    env.step({"action": "list_records", "table": "ledger", "limit": 3})
    env.step({"action": "get_record", "table": "ledger", "id": "L-9999"})
    env.step({"action": "calculator", "expression": "10 / 0"})
    env.step({"action": "submit", "submission": {"episode_id": "ep_test_001", "findings": []}})

    traj_path = tmp_path / "trajectories.jsonl"
    env.dump_trajectory_jsonl(traj_path, append=False)

    violations = verify_trajectory_file(traj_path, forbidden_tokens={"ISS-SECRET-999", "987654321"})
    assert violations == []


def test_19_leak_checker_catches_planted_truth_and_path_leaks() -> None:
    assert scan_payload_for_leaks({"msg": "Found ISS-001 in truth.json"}) != []
    assert scan_payload_for_leaks({"path": "C:\\Users\\admin\\truth.json"}) != []
    assert scan_payload_for_leaks({"path": "/app/data/ep_001"}) != []

    with pytest.raises(AssertionError, match="Ground-truth leak detected"):
        assert_no_leaks({"debug": "expected_amount_minor=10000"})


# ---------------------------------------------------------------------------
# Subprocess JSON-RPC Evaluator Isolation Tests
# ---------------------------------------------------------------------------


def test_20_isolated_client_full_lifecycle_and_memory_isolation(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)

    with IsolatedFinAuditEnv(mode="train", episode_dir=tmp_path) as client:
        obs, info = client.reset(seed=555)
        assert obs["manifest"]["episode_id"] == "ep_test_001"
        assert info["steps_remaining"] == 100

        # Ensure client process memory holds no truth attributes or secret issue IDs
        assert not hasattr(client, "_evaluator_truth")
        assert not hasattr(client, "_datastore")
        client_state_str = str(vars(client))
        assert "ISS-SECRET-999" not in client_state_str
        assert "HIDDEN_NOTE_DO_NOT_LEAK" not in client_state_str

        # Perform all 5 actions over subprocess JSON-RPC
        obs1, r1, term1, trunc1, _ = client.step({"action": "list_records", "table": "ledger", "limit": 3})
        assert r1 == 0.0 and not term1 and not trunc1
        assert len(obs1["result"]["items"]) == 3

        obs2, r2, _, _, _ = client.step({"action": "get_record", "table": "ledger", "id": "L-0001"})
        assert r2 == 0.0 and obs2["result"]["row_id"] == "L-0001"

        obs3, r3, _, _, _ = client.step({"action": "get_document", "doc_id": "DOC-INV-001"})
        assert r3 == 0.0 and obs3["result"]["invoice_id"] == "INV-2026-001"

        obs4, r4, _, _, _ = client.step({"action": "calculator", "expression": "5000 + 5000"})
        assert r4 == 0.0 and obs4["result"]["result"] == "10000"

        obs5, r5, term5, trunc5, info5 = client.step(
            {
                "action": "submit",
                "submission": {
                    "episode_id": "ep_test_001",
                    "findings": [
                        {
                            "type": "duplicate_invoice",
                            "affected_ids": ["L-0001", "INV-2026-001"],
                            "claimed_amount_minor": 10000,
                            "currency": "USD",
                            "evidence_ids": ["DOC-INV-001", "L-0001"],
                            "explanation": "Isolated subprocess submit.",
                        }
                    ],
                },
            }
        )
        assert term5 is True and not trunc5
        assert r5 == pytest.approx(1.0 - 0.002 * 4)
        assert info5["evaluation"]["f1"] == 1.0

        traj = client.get_trajectory()
        assert len(traj) == 5


def test_21_isolated_client_eval_mode_and_bounded_errors_over_ipc(tmp_path: Path) -> None:
    _write_sample_episode(tmp_path, num_ledger_rows=10)

    with IsolatedFinAuditEnv(mode="eval", episode_dir=tmp_path) as client:
        client.reset(seed=123)

        # Invalid action over IPC does not crash subprocess
        obs_err, r_err, term_err, trunc_err, _ = client.step({"action": "non_existent_tool"})
        assert "error" in obs_err
        assert r_err == 0.0 and not term_err and not trunc_err

        # Malformed submit over IPC terminates episode cleanly with internal reward -1.0
        obs_sub, r_sub, term_sub, _, info_sub = client.step(
            {"action": "submit", "submission": {"invalid": "payload"}}
        )
        assert term_sub is True
        assert "error" in obs_sub
        # In eval mode, returned reward to agent is 0.0 and evaluation is omitted from info
        assert r_sub == 0.0
        assert "evaluation" not in info_sub

        # Evaluator side still recorded -1.0
        eval_res = client.get_last_evaluation()
        assert eval_res is not None
        assert eval_res["reward"] == -1.0
        assert eval_res["invalid_structure"] is True


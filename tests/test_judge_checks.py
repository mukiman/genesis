"""Official Genesis FinTech Judge Verification Suite (`test_judge_checks.py`).

Tests the complete specification published at `https://genesis-exe.com/fintech-checks.html`
via `judge_adapter.py`:
- Block 1: 20 Contract & Isolation tests (`C01` .. `C20`) — 25 pts
- Block 2: 20 Closed Episode generation & invariant tests (`D01` .. `D20`) — 25 pts
- Block 3: 20 Matcher & Reward tests (`01` .. `20`) + 12 Open Examples (`M01` .. `M12`) — 25 pts
- Block 4: 20 Paired permutation & ID-remapping checks (`P01` .. `P20`) — 15 pts
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

import judge_adapter
from finaudit_env.build_datasets import load_episode_from_dir, save_episode_to_dir
from finaudit_env.env.leak_checker import scan_payload_for_leaks
from finaudit_env.generator import permute_and_remap_episode, verify_episode
from finaudit_env.policies.oracle import OraclePolicy
from finaudit_env.schemas import IssueType

ABS_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Fixtures for Block 1 (C, E, L, U) and Block 3
# ---------------------------------------------------------------------------


@pytest.fixture()
def block1_fixtures(tmp_path: Path) -> dict[str, Any]:
    """Build common fixtures C (clean), E (1 error), L (>=101 rows), and nonexistent ID U."""
    dir_c = tmp_path / "fixture_C"
    dir_e = tmp_path / "fixture_E"
    dir_l = tmp_path / "fixture_L"

    judge_adapter.generate_episode(
        seed=101,
        config={"episode_id": "ep_fix_C", "clean_only": True, "num_invoices": 25},
        output_dir=str(dir_c),
    )
    judge_adapter.generate_episode(
        seed=102,
        config={
            "episode_id": "ep_fix_E",
            "duplicate_invoice": 1,
            "unallocated_or_partial_payment": 0,
            "fx_conversion_error": 0,
            "period_cutoff_error": 0,
            "num_invoices": 25,
        },
        output_dir=str(dir_e),
    )
    judge_adapter.generate_episode(
        seed=103,
        config={"episode_id": "ep_fix_L", "clean_only": True, "num_invoices": 45},
        output_dir=str(dir_l),
    )

    ep_c = load_episode_from_dir(dir_c)
    ep_e = load_episode_from_dir(dir_e)
    ep_l = load_episode_from_dir(dir_l)
    assert len(ep_c["truth"].issues) == 0
    assert len(ep_e["truth"].issues) == 1
    assert len(ep_l["ledger"]) >= 101

    return {
        "dir_c": dir_c,
        "dir_e": dir_e,
        "dir_l": dir_l,
        "r1": ep_c["ledger"][0].row_id,
        "d1": ep_c["invoices"][0].doc_id,
        "u": "NONEXISTENT-ID-999999",
        "ep_c": ep_c,
        "ep_e": ep_e,
        "ep_l": ep_l,
    }


def _write_block3_fixture(
    ep_dir: Path,
    *,
    issues: list[dict[str, Any]],
    extra_record_ids: list[str] | None = None,
    extra_doc_ids: list[str] | None = None,
) -> Path:
    """Create a deterministic Block 3 test fixture directory with R1..R5 and D1..D3 in KZT."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    rec_ids = ["L-0001", "L-0002", "L-0003", "L-0004", "L-0005"]
    if extra_record_ids:
        for rid in extra_record_ids:
            if rid not in rec_ids:
                rec_ids.append(rid)

    doc_ids = ["DOC-INV-001", "DOC-INV-002", "DOC-INV-003"]
    if extra_doc_ids:
        for did in extra_doc_ids:
            if did not in doc_ids:
                doc_ids.append(did)

    manifest = {
        "schema_version": "1.0.0",
        "episode_id": "ep_block3_test",
        "company_name": "Block3 Test KZ",
        "base_currency": "KZT",
        "period_start": "2026-03-01",
        "period_end": "2026-03-31",
        "timezone": "Asia/Almaty",
        "rounding_mode": "HALF_EVEN",
        "ledger_rows_count": len(rec_ids),
        "invoices_count": len(doc_ids),
        "payments_count": 0,
        "fx_rates_count": 1,
    }
    (ep_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    policy = {
        "fx_rate_date_rule": "transaction_date",
        "period_start": "2026-03-01",
        "period_end": "2026-03-31",
        "timezone": "Asia/Almaty",
        "rounding_mode": "HALF_EVEN",
        "amount_tolerance_minor": 0,
    }
    (ep_dir / "policy.json").write_text(json.dumps(policy), encoding="utf-8")

    ledger_lines = ["row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id"]
    for idx, rid in enumerate(rec_ids):
        debit = 10000 if idx % 2 == 0 else 0
        credit = 0 if idx % 2 == 0 else 10000
        ledger_lines.append(
            f"{rid},2026-03-10T10:00:00+05:00,1100-AR,{debit},{credit},KZT,ENT-001,INV-2026-001,"
        )
    (ep_dir / "ledger.csv").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    inv_lines = []
    for idx, did in enumerate(doc_ids, start=1):
        inv_obj = {
            "doc_id": did,
            "invoice_id": f"INV-2026-{idx:03d}",
            "entity_id": "ENT-001",
            "issued_at": "2026-03-10T10:00:00+05:00",
            "due_at": "2026-03-25T10:00:00+05:00",
            "currency": "KZT",
            "total_minor": 10000,
            "line_items": [
                {
                    "description": "Service",
                    "quantity": 1,
                    "unit_price_minor": 10000,
                    "amount_minor": 10000,
                }
            ],
        }
        inv_lines.append(json.dumps(inv_obj))
    (ep_dir / "invoices.jsonl").write_text("\n".join(inv_lines) + "\n", encoding="utf-8")
    (ep_dir / "payments.jsonl").write_text("", encoding="utf-8")
    (ep_dir / "fx.csv").write_text(
        "date,source_currency,target_currency,rate\n2026-03-10,USD,KZT,495.0000\n",
        encoding="utf-8",
    )

    truth_payload = {
        "schema_version": "1.0.0",
        "episode_id": "ep_block3_test",
        "issues": issues,
    }
    (ep_dir / "truth.json").write_text(json.dumps(truth_payload), encoding="utf-8")
    (ep_dir / "provenance.json").write_text(
        json.dumps({"episode_id": "ep_block3_test", "seed": 1, "events": []}),
        encoding="utf-8",
    )
    return ep_dir


# ---------------------------------------------------------------------------
# BLOCK 1: 20 Contract & Isolation Tests (C01 .. C20) — 25 points
# ---------------------------------------------------------------------------


def test_block1_c01_to_c20_contract_and_isolation(block1_fixtures: dict[str, Any], tmp_path: Path) -> None:
    dir_c: Path = block1_fixtures["dir_c"]
    dir_e: Path = block1_fixtures["dir_e"]
    dir_l: Path = block1_fixtures["dir_l"]
    r1: str = block1_fixtures["r1"]
    d1: str = block1_fixtures["d1"]
    u: str = block1_fixtures["u"]
    ep_c = block1_fixtures["ep_c"]
    ep_l = block1_fixtures["ep_l"]

    env = judge_adapter.make_env()

    # C01: reset on C -> returns JSON-compatible (observation, info)
    obs, info = env.reset(seed=101, config={"mode": "closed", "episode_dir": str(dir_c)})
    json.dumps(obs)
    json.dumps(info)

    # C02: First observation has task, manifest, policy, remaining_steps=100; no seed or truth
    assert isinstance(obs["task"], str) and len(obs["task"]) > 0
    assert isinstance(obs["manifest"], dict)
    assert isinstance(obs["policy"], dict)
    assert obs["remaining_steps"] == 100
    assert "truth" not in obs and "seed" not in obs

    # C03: Manifest has required fields and counts match actual data
    m = obs["manifest"]
    for key in (
        "schema_version",
        "episode_id",
        "base_currency",
        "period_start",
        "period_end",
        "timezone",
        "rounding_mode",
        "ledger_rows_count",
        "invoices_count",
        "payments_count",
        "fx_rates_count",
    ):
        assert key in m
    assert m["ledger_rows_count"] == len(ep_c["ledger"])
    assert m["invoices_count"] == len(ep_c["invoices"])
    assert m["payments_count"] == len(ep_c["payments"])
    assert m["fx_rates_count"] == len(ep_c["fx"])

    # C04: One valid list_records on C -> step returns (obs, reward, terminated, truncated, info)
    s_obs, reward, terminated, truncated, s_info = env.step(
        {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
    )
    assert isinstance(terminated, bool) and not terminated
    assert isinstance(truncated, bool) and not truncated
    assert reward == 0
    assert s_obs["remaining_steps"] == 99
    json.dumps(s_obs)
    json.dumps(s_info)

    # C05 & C06: list_records on L with limit=50; paginate all cursors + repeat same cursor
    env.reset(seed=103, config={"mode": "closed", "episode_dir": str(dir_l)})
    collected_ids: list[str] = []
    cursor: str | None = None
    first_page_records: list[dict[str, Any]] | None = None
    second_cursor: str | None = None

    while True:
        p_obs, _, term, trunc, _ = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}}
        )
        assert not term and not trunc
        assert p_obs["error"] is None
        recs = p_obs["result"]["records"]
        assert len(recs) <= 50
        if first_page_records is None:
            first_page_records = recs
            second_cursor = p_obs["result"]["next_cursor"]
        collected_ids.extend(r["row_id"] for r in recs)
        cursor = p_obs["result"]["next_cursor"]
        if cursor is None:
            break

    assert collected_ids == [r.row_id for r in ep_l["ledger"]]

    # C06: Request the same page of L twice using the same cursor in one episode
    assert second_cursor is not None
    rep1, _, _, _, _ = env.step(
        {"name": "list_records", "args": {"table": "ledger", "cursor": second_cursor, "limit": 50}}
    )
    rep2, _, _, _, _ = env.step(
        {"name": "list_records", "args": {"table": "ledger", "cursor": second_cursor, "limit": 50}}
    )
    assert rep1["result"]["records"] == rep2["result"]["records"]
    assert rep2["remaining_steps"] == rep1["remaining_steps"] - 1

    # C07: get_record for existing R1 on C
    env.reset(seed=101, config={"mode": "closed", "episode_dir": str(dir_c)})
    r_obs, _, _, _, _ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1}})
    assert r_obs["error"] is None
    assert r_obs["result"]["row_id"] == r1

    # C08: get_document for existing D1 on C
    d_obs, _, _, _, _ = env.step({"name": "get_document", "args": {"doc_id": d1}})
    assert d_obs["error"] is None
    assert d_obs["result"]["doc_id"] == d1

    # C09: calculator with "0.1+0.2" -> decimal "0.3"
    c_obs, _, _, _, _ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
    assert c_obs["error"] is None
    assert c_obs["result"] == "0.3"

    # C10: calculator with non-arithmetic expression (file access attempt)
    bad_calc_obs, _, _, _, _ = env.step(
        {"name": "calculator", "args": {"expression": "open('truth.json').read()"}}
    )
    assert bad_calc_obs["result"] is None
    assert bad_calc_obs["error"] is not None

    # C11: get_record for nonexistent U
    rem_before = bad_calc_obs["remaining_steps"]
    u_rec_obs, _, _, _, _ = env.step({"name": "get_record", "args": {"table": "ledger", "id": u}})
    assert u_rec_obs["result"] is None
    assert u_rec_obs["error"] is not None
    assert u_rec_obs["remaining_steps"] == rem_before - 1

    # C12: get_document for nonexistent U
    u_doc_obs, _, _, _, _ = env.step({"name": "get_document", "args": {"doc_id": u}})
    assert u_doc_obs["result"] is None
    assert u_doc_obs["error"] is not None
    assert u_doc_obs["remaining_steps"] == rem_before - 2

    # C13: step with unknown action name
    unk_obs, _, _, _, _ = env.step({"name": "execute_shell_sql", "args": {"cmd": "ls"}})
    assert unk_obs["result"] is None
    assert unk_obs["error"] is not None
    assert unk_obs["remaining_steps"] == rem_before - 3

    # C14: After valid and errored actions, request valid record again -> budget decremented by 1 per call
    again_obs, _, _, _, _ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1}})
    assert again_obs["error"] is None
    assert again_obs["result"]["row_id"] == r1
    assert again_obs["remaining_steps"] == rem_before - 4

    # C15: submit with valid structure before limit -> terminated=True; subsequent calls do not change findings/score
    sub_obs, sub_rew, sub_term, sub_trunc, _ = env.step(
        {"name": "submit", "args": {"findings": {"episode_id": "ep_fix_C", "findings": []}}}
    )
    assert sub_term is True and sub_trunc is False
    assert sub_obs["result"] == {"accepted": True}
    eval_snap = env.get_last_evaluation()
    post_obs, _, post_term, post_trunc, _ = env.step(
        {"name": "get_record", "args": {"table": "ledger", "id": r1}}
    )
    assert post_term is True and post_trunc is False
    assert post_obs["error"] is not None
    assert env.get_last_evaluation() == eval_snap

    # C16: 99 inspection actions, then valid submit as action 100 -> accepted, terminated=True
    env.reset(seed=101, config={"mode": "closed", "episode_dir": str(dir_c)})
    for _ in range(99):
        _, _, t_flag, tr_flag, _ = env.step({"name": "calculator", "args": {"expression": "1+1"}})
        assert not t_flag and not tr_flag
    obs_100, _, term_100, trunc_100, _ = env.step(
        {"name": "submit", "args": {"findings": {"episode_id": "ep_fix_C", "findings": []}}}
    )
    assert term_100 is True and trunc_100 is False
    assert obs_100["result"] == {"accepted": True}
    assert env.get_last_evaluation()["inspection_steps"] == 99

    # C17: 100 inspection actions without submit -> truncated=True, empty findings
    env.reset(seed=101, config={"mode": "closed", "episode_dir": str(dir_c)})
    for step_idx in range(100):
        _, _, t_flag, tr_flag, _ = env.step({"name": "calculator", "args": {"expression": "2+2"}})
        if step_idx < 99:
            assert not t_flag and not tr_flag
        else:
            assert not t_flag and tr_flag is True
    # Post-truncation submit does not alter recorded score
    snap_100 = env.get_last_evaluation()
    env.step({"name": "submit", "args": {"findings": {"episode_id": "ep_fix_C", "findings": []}}})
    assert env.get_last_evaluation() == snap_100

    # C18: Reproducible scenario reaching time limit -> truncated=True, empty findings
    env.reset(
        seed=101,
        config={"mode": "closed", "episode_dir": str(dir_c), "timeout_seconds": 0.001},
    )
    time.sleep(0.03)
    t_obs, _, t_term, t_trunc, _ = env.step({"name": "calculator", "args": {"expression": "3+3"}})
    assert t_term is False and t_trunc is True
    assert t_obs["error"] is not None

    # C19: Inspection steps + submit on E in closed mode -> all returned rewards are numeric 0, info has no evaluation
    env.reset(seed=102, config={"mode": "closed", "episode_dir": str(dir_e)})
    _, r_step1, _, _, info_step1 = env.step(
        {"name": "list_records", "args": {"table": "invoices", "cursor": None, "limit": 10}}
    )
    assert r_step1 == 0 and isinstance(r_step1, (int, float))
    assert "evaluation" not in info_step1

    oracle_findings = OraclePolicy.build_oracle_submission(
        block1_fixtures["ep_e"]["truth"]
    ).model_dump(mode="json")
    f_obs, r_final, term_e, trunc_e, info_final = env.step(
        {"name": "submit", "args": {"findings": oracle_findings}}
    )
    assert term_e is True and trunc_e is False
    assert r_final == 0
    assert "evaluation" not in info_final
    assert f_obs["result"] == {"accepted": True}
    # Meanwhile evaluator score_episode gets the true reward:
    true_score = judge_adapter.score_episode(str(dir_e), oracle_findings, inspection_steps=1)
    assert true_score["tp"] == 1 and true_score["f1"] == 1.0 and true_score["return"] > 0.99

    # C20: Iterate all open records/docs on C and E, check errors & info -> 0 leaks of truth/provenance/seed
    for ep_dir in (dir_c, dir_e):
        obs0, inf0 = env.reset(seed=999, config={"mode": "closed", "episode_dir": str(ep_dir)})
        assert not scan_payload_for_leaks({"obs": obs0, "info": inf0})
        for tbl in ("ledger", "invoices", "payments", "fx"):
            cur = None
            while True:
                o_page, _, _, _, i_page = env.step(
                    {"name": "list_records", "args": {"table": tbl, "cursor": cur, "limit": 50}}
                )
                assert not scan_payload_for_leaks({"obs": o_page, "info": i_page})
                cur = o_page["result"]["next_cursor"]
                if cur is None:
                    break


# ---------------------------------------------------------------------------
# BLOCK 2: 20 Closed Episodes Matrix (D01 .. D20) — 25 points
# ---------------------------------------------------------------------------


def test_block2_d01_to_d20_closed_episodes_matrix(tmp_path: Path) -> None:
    """Verify all 20 closed episode scenarios D01..D20 against all 6 pass conditions."""
    matrix_configs: list[tuple[str, int, dict[str, Any], int]] = [
        # D01-D05: 5 clean companies (truth empty)
        ("D01", 3001, {"clean_only": True, "base_currency": "USD"}, 0),
        ("D02", 3002, {"clean_only": True, "enable_split_payments": True, "base_currency": "EUR"}, 0),
        ("D03", 3003, {"clean_only": True, "enable_refunds": True, "base_currency": "GBP"}, 0),
        ("D04", 3004, {"clean_only": True, "enable_recurring_payments": True, "base_currency": "KZT"}, 0),
        (
            "D05",
            3005,
            {
                "clean_only": True,
                "base_currency": "CHF",
                "fx_rate_date_rule": "previous_business_day",
                "timezone": "Asia/Almaty",
            },
            0,
        ),
        # D06-D08: 3 duplicate invoice episodes
        ("D06", 3006, {"duplicate_invoice": 1, "base_currency": "USD"}, 1),
        ("D07", 3007, {"duplicate_invoice": 2, "base_currency": "EUR"}, 2),
        ("D08", 3008, {"duplicate_invoice": 1, "base_currency": "KZT"}, 1),
        # D09-D11: 3 missed or partial payment allocation episodes (at least 1 missed and 1 partial)
        ("D09", 3009, {"unallocated_payment": 1, "base_currency": "USD"}, 1),
        ("D10", 3010, {"partial_payment": 1, "base_currency": "EUR"}, 1),
        ("D11", 3011, {"unallocated_payment": 1, "partial_payment": 1, "base_currency": "GBP"}, 2),
        # D12-D13: 2 wrong FX date or conversion rate episodes
        ("D12", 3012, {"fx_conversion_error": 1, "fx_rate_date_rule": "transaction_date"}, 1),
        ("D13", 3013, {"fx_conversion_error": 2, "fx_rate_date_rule": "previous_business_day"}, 2),
        # D14-D15: 2 period cutoff error episodes
        ("D14", 3014, {"period_cutoff_error": 1, "timezone": "UTC"}, 1),
        ("D15", 3015, {"period_cutoff_error": 2, "timezone": "Asia/Almaty"}, 2),
        # D16-D18: 3 episodes of 1st documented interaction scenario (partial_fx_cutoff)
        ("D16", 3016, {"interacting_scenarios": ["partial_fx_cutoff"], "base_currency": "USD"}, 2),
        ("D17", 3017, {"interacting_scenarios": ["partial_fx_cutoff"], "base_currency": "KZT"}, 2),
        ("D18", 3018, {"interacting_scenarios": ["partial_fx_cutoff"], "base_currency": "EUR"}, 2),
        # D19-D20: 2 episodes of 2nd documented interaction scenario (duplicate_fx_cutoff)
        ("D19", 3019, {"interacting_scenarios": ["duplicate_fx_cutoff"], "base_currency": "USD"}, 2),
        ("D20", 3020, {"interacting_scenarios": ["duplicate_fx_cutoff"], "base_currency": "GBP"}, 2),
    ]

    for code, seed, cfg, expected_issues in matrix_configs:
        ep_dir = tmp_path / code
        cfg_with_id = dict(cfg, episode_id=f"ep_{code.lower()}")
        judge_adapter.generate_episode(seed=seed, config=cfg_with_id, output_dir=str(ep_dir))
        ep = load_episode_from_dir(ep_dir)

        assert len(ep["truth"].issues) == expected_issues, f"{code}: unexpected issue count"
        ver = verify_episode(ep)
        assert ver["valid"] is True, f"{code} failed verification: {ver['errors']}"

        # Verify Oracle achieves F1 = 1.0 via judge_adapter.score_episode
        oracle_sub = OraclePolicy.build_oracle_submission(ep["truth"]).model_dump(mode="json")
        score = judge_adapter.score_episode(str(ep_dir), oracle_sub, inspection_steps=5)
        assert score["f1"] == pytest.approx(1.0, abs=ABS_TOL)
        assert score["fn"] == 0 and score["fp"] == 0


# ---------------------------------------------------------------------------
# BLOCK 3: 20 Matcher & Reward Tests (1 .. 20) + Open Examples (M01 .. M12)
# ---------------------------------------------------------------------------


def test_block3_01_to_20_matcher_and_reward(tmp_path: Path) -> None:
    """Verify all 20 official matcher & reward tests from Block 3 of `fintech-checks`."""
    issue_a = {
        "issue_id": "ISS-A",
        "type": IssueType.DUPLICATE_INVOICE.value,
        "affected_ids": ["L-0001"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "valid_evidence_sets": [["DOC-INV-001", "DOC-INV-002"]],
        "tolerance_minor": 0,
    }
    issue_b = {
        "issue_id": "ISS-B",
        "type": IssueType.PERIOD_CUTOFF_ERROR.value,
        "affected_ids": ["L-0002"],
        "expected_amount_minor": 20000,
        "currency": "KZT",
        "valid_evidence_sets": [["DOC-INV-002", "DOC-INV-003"]],
        "tolerance_minor": 0,
    }

    dir_single = _write_block3_fixture(tmp_path / "b3_single", issues=[issue_a])
    dir_clean = _write_block3_fixture(tmp_path / "b3_clean", issues=[])
    dir_two = _write_block3_fixture(tmp_path / "b3_two", issues=[issue_a, issue_b])

    valid_f = {
        "type": IssueType.DUPLICATE_INVOICE.value,
        "affected_ids": ["L-0001"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["DOC-INV-001", "DOC-INV-002"],
        "explanation": "Valid duplicate invoice finding.",
    }

    # Test 1: Truth A; 1 valid F -> TP=1, FP=0, FN=0; P=R=F1=1; reward=0.98
    r1 = judge_adapter.score_episode(dir_single, {"findings": [valid_f]}, inspection_steps=10)
    assert (r1["tp"], r1["fp"], r1["fn"]) == (1, 0, 0)
    assert r1["precision"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r1["recall"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r1["f1"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r1["return"] == pytest.approx(0.98, abs=ABS_TOL)

    # Test 2: Truth A; empty findings -> TP=0, FP=0, FN=1; P=R=F1=0; reward=-1
    r2 = judge_adapter.score_episode(dir_single, {"findings": []}, inspection_steps=10)
    assert (r2["tp"], r2["fp"], r2["fn"]) == (0, 0, 1)
    assert r2["precision"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r2["recall"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r2["f1"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r2["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 3: Empty truth; empty findings -> TP=FP=FN=0; P=R=0; F1=1; reward=0.98
    r3 = judge_adapter.score_episode(dir_clean, {"findings": []}, inspection_steps=10)
    assert (r3["tp"], r3["fp"], r3["fn"]) == (0, 0, 0)
    assert r3["precision"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r3["recall"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r3["f1"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r3["return"] == pytest.approx(0.98, abs=ABS_TOL)

    # Test 4: Empty truth; 1 structurally valid false finding -> TP=0, FP=1, FN=0; F1=0; reward=-1
    r4 = judge_adapter.score_episode(dir_clean, {"findings": [valid_f]}, inspection_steps=10)
    assert (r4["tp"], r4["fp"], r4["fn"]) == (0, 1, 0)
    assert r4["f1"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r4["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 5: Truth A; 2 identical F -> TP=1, FP=1, FN=0; P=1/2, R=1, F1=2/3; reward=47/150
    r5 = judge_adapter.score_episode(dir_single, {"findings": [valid_f, valid_f]}, inspection_steps=10)
    assert (r5["tp"], r5["fp"], r5["fn"]) == (1, 1, 0)
    assert r5["precision"] == pytest.approx(0.5, abs=ABS_TOL)
    assert r5["recall"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r5["f1"] == pytest.approx(2.0 / 3.0, abs=ABS_TOL)
    assert r5["return"] == pytest.approx(47.0 / 150.0, abs=ABS_TOL)

    # Test 6: Truth A; F with wrong type -> TP=0, FP=1, FN=1; F1=0; reward=-1
    f_wrong_type = dict(valid_f, type=IssueType.PERIOD_CUTOFF_ERROR.value)
    r6 = judge_adapter.score_episode(dir_single, {"findings": [f_wrong_type]}, inspection_steps=10)
    assert (r6["tp"], r6["fp"], r6["fn"]) == (0, 1, 1)
    assert r6["f1"] == pytest.approx(0.0, abs=ABS_TOL)
    assert r6["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 7: Truth A; F with another existing but unaffected record in affected_ids -> TP=0, FP=1, FN=1
    f_wrong_aff = dict(valid_f, affected_ids=["L-0001", "L-0002"])
    r7 = judge_adapter.score_episode(dir_single, {"findings": [f_wrong_aff]}, inspection_steps=10)
    assert (r7["tp"], r7["fp"], r7["fn"]) == (0, 1, 1)
    assert r7["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 8: Truth A; F with claimed_amount_minor=10001 at tolerance 0 -> TP=0, FP=1, FN=1
    f_wrong_amt = dict(valid_f, claimed_amount_minor=10001)
    r8 = judge_adapter.score_episode(dir_single, {"findings": [f_wrong_amt]}, inspection_steps=10)
    assert (r8["tp"], r8["fp"], r8["fn"]) == (0, 1, 1)
    assert r8["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 9: Truth A; F with different currency -> TP=0, FP=1, FN=1
    f_wrong_curr = dict(valid_f, currency="USD")
    r9 = judge_adapter.score_episode(dir_single, {"findings": [f_wrong_curr]}, inspection_steps=10)
    assert (r9["tp"], r9["fp"], r9["fn"]) == (0, 1, 1)
    assert r9["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 10: Truth A; F has only D1 instead of full {D1, D2} -> TP=0, FP=1, FN=1
    f_partial_ev = dict(valid_f, evidence_ids=["DOC-INV-001"])
    r10 = judge_adapter.score_episode(dir_single, {"findings": [f_partial_ev]}, inspection_steps=10)
    assert (r10["tp"], r10["fp"], r10["fn"]) == (0, 1, 1)
    assert r10["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 11: Truth A; F has nonexistent evidence_id instead of D2 -> TP=0, FP=1, FN=1
    f_missing_ev = dict(valid_f, evidence_ids=["DOC-INV-001", "DOC-NONEXISTENT-999"])
    r11 = judge_adapter.score_episode(dir_single, {"findings": [f_missing_ev]}, inspection_steps=10)
    assert (r11["tp"], r11["fp"], r11["fn"]) == (0, 1, 1)
    assert r11["return"] == pytest.approx(-1.0, abs=ABS_TOL)

    # Test 12: 5 records & 1 error A; mark every record with all 4 types (20 findings, 1 matches A)
    attack_findings = []
    for rid in ("L-0001", "L-0002", "L-0003", "L-0004", "L-0005"):
        for itype in IssueType:
            attack_findings.append(
                {
                    "type": itype.value,
                    "affected_ids": [rid],
                    "claimed_amount_minor": 10000,
                    "currency": "KZT",
                    "evidence_ids": ["DOC-INV-001", "DOC-INV-002"],
                    "explanation": f"Flag all {rid} {itype.value}",
                }
            )
    assert len(attack_findings) == 20
    r12 = judge_adapter.score_episode(dir_single, {"findings": attack_findings}, inspection_steps=10)
    assert (r12["tp"], r12["fp"], r12["fn"]) == (1, 19, 0)
    assert r12["precision"] == pytest.approx(1.0 / 20.0, abs=ABS_TOL)
    assert r12["recall"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r12["f1"] == pytest.approx(2.0 / 21.0, abs=ABS_TOL)
    assert r12["return"] == pytest.approx(-871.0 / 1050.0, abs=ABS_TOL)

    # Test 13: Two distinct errors A and B; only valid F for A -> TP=1, FP=0, FN=1; P=1, R=1/2, F1=2/3; reward=47/150
    r13 = judge_adapter.score_episode(dir_two, {"findings": [valid_f]}, inspection_steps=10)
    assert (r13["tp"], r13["fp"], r13["fn"]) == (1, 0, 1)
    assert r13["precision"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r13["recall"] == pytest.approx(0.5, abs=ABS_TOL)
    assert r13["f1"] == pytest.approx(2.0 / 3.0, abs=ABS_TOL)
    assert r13["return"] == pytest.approx(47.0 / 150.0, abs=ABS_TOL)

    # Test 14: Two distinct errors A and B; valid F for A + 1 false finding -> TP=1, FP=1, FN=1; P=R=F1=1/2; reward=-0.02
    r14 = judge_adapter.score_episode(
        dir_two, {"findings": [valid_f, f_wrong_type]}, inspection_steps=10
    )
    assert (r14["tp"], r14["fp"], r14["fn"]) == (1, 1, 1)
    assert r14["precision"] == pytest.approx(0.5, abs=ABS_TOL)
    assert r14["recall"] == pytest.approx(0.5, abs=ABS_TOL)
    assert r14["f1"] == pytest.approx(0.5, abs=ABS_TOL)
    assert r14["return"] == pytest.approx(-0.02, abs=ABS_TOL)

    # Test 15: Truth A; valid F; inspection_steps=0 -> F1=1; reward=1.0
    r15 = judge_adapter.score_episode(dir_single, {"findings": [valid_f]}, inspection_steps=0)
    assert r15["f1"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r15["return"] == pytest.approx(1.0, abs=ABS_TOL)

    # Test 16: Truth A; valid F; inspection_steps=50 -> F1=1; reward=0.90
    r16 = judge_adapter.score_episode(dir_single, {"findings": [valid_f]}, inspection_steps=50)
    assert r16["f1"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r16["return"] == pytest.approx(0.90, abs=ABS_TOL)

    # Test 17: Truth A; valid F; inspection_steps=100 -> F1=1; reward=0.80
    r17 = judge_adapter.score_episode(dir_single, {"findings": [valid_f]}, inspection_steps=100)
    assert r17["f1"] == pytest.approx(1.0, abs=ABS_TOL)
    assert r17["return"] == pytest.approx(0.80, abs=ABS_TOL)

    # Test 18: submit with 31 findings -> invalid structure: return=-1, other fields None
    r18 = judge_adapter.score_episode(dir_single, {"findings": [valid_f] * 31}, inspection_steps=10)
    assert r18 == {
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "return": -1,
    }

    # Test 19: 1 finding with 11 distinct evidence IDs -> invalid structure: return=-1
    f_11_ev = dict(valid_f, evidence_ids=[f"DOC-INV-{i:03d}" for i in range(1, 12)])
    r19 = judge_adapter.score_episode(dir_single, {"findings": [f_11_ev]}, inspection_steps=10)
    assert r19["return"] == -1 and r19["tp"] is None

    # Test 20: 1 finding with 1001 ASCII-char explanation -> invalid structure: return=-1
    f_1001_exp = dict(valid_f, explanation="A" * 1001)
    r20 = judge_adapter.score_episode(dir_single, {"findings": [f_1001_exp]}, inspection_steps=10)
    assert r20["return"] == -1 and r20["tp"] is None


def test_block3_open_examples_m01_to_m12(tmp_path: Path) -> None:
    """Verify all 12 open matcher examples M01..M12 from `fintech-checks`."""
    issue_m = {
        "issue_id": "ISS-A",
        "type": IssueType.DUPLICATE_INVOICE.value,
        "affected_ids": ["L-0001", "L-0002"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "valid_evidence_sets": [["DOC-INV-001", "DOC-INV-002"]],
        "tolerance_minor": 0,
    }
    dir_m = _write_block3_fixture(tmp_path / "m_base", issues=[issue_m])
    base_f = {
        "type": IssueType.DUPLICATE_INVOICE.value,
        "affected_ids": ["L-0001", "L-0002"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["DOC-INV-001", "DOC-INV-002"],
        "explanation": "Finding for M examples.",
    }

    # M01: affected_ids=[R2, R1, R1] -> set {R1, R2} -> TP=1
    m01 = judge_adapter.score_episode(
        dir_m, {"findings": [dict(base_f, affected_ids=["L-0002", "L-0001", "L-0001"])]}, 10
    )
    assert (m01["tp"], m01["fp"], m01["fn"]) == (1, 0, 0)

    # M02: affected_ids=[R1, R2, R3] with existing unaffected R3 -> FP=1, FN=1
    m02 = judge_adapter.score_episode(
        dir_m, {"findings": [dict(base_f, affected_ids=["L-0001", "L-0002", "L-0003"])]}, 10
    )
    assert (m02["tp"], m02["fp"], m02["fn"]) == (0, 1, 1)

    # M03: evidence_ids=[D2, D1, D3] with accessible extra D3 -> TP=1
    m03 = judge_adapter.score_episode(
        dir_m,
        {"findings": [dict(base_f, evidence_ids=["DOC-INV-002", "DOC-INV-001", "DOC-INV-003"])]},
        10,
    )
    assert (m03["tp"], m03["fp"], m03["fn"]) == (1, 0, 0)

    # M04: evidence_ids=[D1, D2, U] with nonexistent U -> FP=1, FN=1
    m04 = judge_adapter.score_episode(
        dir_m,
        {"findings": [dict(base_f, evidence_ids=["DOC-INV-001", "DOC-INV-002", "DOC-U-999"])]},
        10,
    )
    assert (m04["tp"], m04["fp"], m04["fn"]) == (0, 1, 1)

    # M05: evidence_ids of 11 elements (10 repeating D1 + D2) -> structural error before dedup: return=-1
    m05 = judge_adapter.score_episode(
        dir_m,
        {"findings": [dict(base_f, evidence_ids=["DOC-INV-001"] * 10 + ["DOC-INV-002"])]},
        10,
    )
    assert m05["return"] == -1 and m05["tp"] is None

    # M06: currency differs by case ("kzt") or whitespace ("KZT ") -> FP=1, FN=1 (not structural crash)
    for bad_curr in ("kzt", "KZT "):
        m06 = judge_adapter.score_episode(
            dir_m, {"findings": [dict(base_f, currency=bad_curr)]}, 10
        )
        assert (m06["tp"], m06["fp"], m06["fn"]) == (0, 1, 1)

    # M07: amount=10001 at tau=0 -> FP=1, FN=1
    m07 = judge_adapter.score_episode(
        dir_m, {"findings": [dict(base_f, claimed_amount_minor=10001)]}, 10
    )
    assert (m07["tp"], m07["fp"], m07["fn"]) == (0, 1, 1)

    # M08: Separate fixture with tau=1; amount=10001 -> TP=1; amount=10002 -> FP=1, FN=1
    dir_m08 = _write_block3_fixture(
        tmp_path / "m_tau1", issues=[dict(issue_m, tolerance_minor=1)]
    )
    m08_ok = judge_adapter.score_episode(
        dir_m08, {"findings": [dict(base_f, claimed_amount_minor=10001)]}, 10
    )
    m08_fail = judge_adapter.score_episode(
        dir_m08, {"findings": [dict(base_f, claimed_amount_minor=10002)]}, 10
    )
    assert (m08_ok["tp"], m08_ok["fp"], m08_ok["fn"]) == (1, 0, 0)
    assert (m08_fail["tp"], m08_fail["fp"], m08_fail["fn"]) == (0, 1, 1)

    # M09: amount is string "10000" or boolean True -> structural error: return=-1
    for invalid_amt in ("10000", True):
        m09 = judge_adapter.score_episode(
            dir_m, {"findings": [dict(base_f, claimed_amount_minor=invalid_amt)]}, 10
        )
        assert m09["return"] == -1 and m09["tp"] is None

    # M10 & M11: Two issues A and B; verify maximum-cardinality matching & deterministic tie-breaking
    issue_a_amb = dict(issue_m, issue_id="ISS-A", expected_amount_minor=10000, tolerance_minor=5)
    issue_b_amb = dict(issue_m, issue_id="ISS-B", expected_amount_minor=10004, tolerance_minor=2)
    dir_m10 = _write_block3_fixture(tmp_path / "m10", issues=[issue_a_amb, issue_b_amb])
    # F0 (10003) matches both A (|10003-10000|<=5) and B (|10003-10004|<=2)
    # F1 (9996) matches ONLY A (|9996-10000|<=5, |9996-10004|=8>2)
    f0 = dict(base_f, claimed_amount_minor=10003)
    f1 = dict(base_f, claimed_amount_minor=9996)
    m10 = judge_adapter.score_episode(dir_m10, {"findings": [f0, f1]}, 10)
    assert (m10["tp"], m10["fp"], m10["fn"]) == (2, 0, 0)

    # M11: Both F0 and F1 match both A and B (A < B) -> deterministic A->F0, B->F1, TP=2
    from finaudit_env.scoring.matcher import match_findings

    f0_both = dict(base_f, claimed_amount_minor=10003)
    f1_both = dict(base_f, claimed_amount_minor=10003)
    raw_m11 = match_findings(
        {"episode_id": "ep_block3_test", "findings": [f0_both, f1_both]},
        {"episode_id": "ep_block3_test", "issues": [issue_a_amb, issue_b_amb]},
    )
    assert raw_m11["tp"] == 2
    pairs_m11 = [(p["issue_id"], p["finding_index"]) for p in raw_m11["matched_pairs"]]
    assert pairs_m11 == [("ISS-A", 0), ("ISS-B", 1)]

    # M12: Single error A; 2 identical valid findings -> TP=1, FP=1, FN=0
    m12 = judge_adapter.score_episode(dir_m, {"findings": [base_f, base_f]}, 10)
    assert (m12["tp"], m12["fp"], m12["fn"]) == (1, 1, 0)


# ---------------------------------------------------------------------------
# BLOCK 4: 20 Paired Checks (P01 .. P20) — 15 points
# ---------------------------------------------------------------------------


def test_block4_p01_to_p20_paired_checks(tmp_path: Path) -> None:
    """Verify P01..P05 (permute order), P06..P10 (remap IDs), and P11..P20 (both) via judge_adapter."""
    for idx in range(1, 21):
        p_code = f"P{idx:02d}"
        seed = 4000 + idx
        if idx <= 5:
            # P01-P05 on clean companies: permute record order ONLY
            cfg = {"clean_only": True, "episode_id": f"ep_{p_code.lower()}"}
            permute_order, remap_ids = True, False
        elif idx <= 10:
            # P06-P10: bijective ID replacement ONLY
            cfg = {"duplicate_invoice": 1, "unallocated_payment": 1, "episode_id": f"ep_{p_code.lower()}"}
            permute_order, remap_ids = False, True
        else:
            # P11-P20: both order permutation and bijective ID replacement
            scenario = "partial_fx_cutoff" if idx <= 15 else "duplicate_fx_cutoff"
            cfg = {"interacting_scenarios": [scenario], "episode_id": f"ep_{p_code.lower()}"}
            permute_order, remap_ids = True, True

        orig_dir = tmp_path / f"{p_code}_orig"
        trans_dir = tmp_path / f"{p_code}_trans"
        judge_adapter.generate_episode(seed=seed, config=cfg, output_dir=str(orig_dir))

        orig_ep = load_episode_from_dir(orig_dir)
        trans_ep, id_map = permute_and_remap_episode(
            orig_ep,
            seed=seed + 500,
            permute_order=permute_order,
            remap_ids=remap_ids,
        )
        save_episode_to_dir(trans_ep, trans_dir)

        # 1. Verify transformed episode still passes all accounting & evidence sufficiency checks
        ver_trans = verify_episode(trans_ep)
        assert ver_trans["valid"] is True, f"{p_code} transformed verification failed: {ver_trans['errors']}"

        # 2. Exact replay on original episode produces identical observations and transitions
        env = judge_adapter.make_env()
        obs_a, _ = env.reset(seed=seed, config={"mode": "closed", "episode_dir": str(orig_dir)})
        step_a, _, _, _, _ = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 20}}
        )
        obs_b, _ = env.reset(seed=seed, config={"mode": "closed", "episode_dir": str(orig_dir)})
        step_b, _, _, _, _ = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 20}}
        )
        assert obs_a == obs_b
        assert step_a == step_b

        # 3. Oracle submission on original vs transformed copy achieves identical metrics & reward
        oracle_orig = OraclePolicy.build_oracle_submission(orig_ep["truth"]).model_dump(mode="json")
        oracle_trans = OraclePolicy.build_oracle_submission(trans_ep["truth"]).model_dump(mode="json")

        score_orig = judge_adapter.score_episode(str(orig_dir), oracle_orig, inspection_steps=7)
        score_trans = judge_adapter.score_episode(str(trans_dir), oracle_trans, inspection_steps=7)
        assert score_orig["tp"] == score_trans["tp"]
        assert score_orig["fp"] == score_trans["fp"]
        assert score_orig["fn"] == score_trans["fn"]
        assert score_orig["f1"] == pytest.approx(score_trans["f1"], abs=ABS_TOL)
        assert score_orig["return"] == pytest.approx(score_trans["return"], abs=ABS_TOL)

        # 4. Inverse ID mapping of transformed Oracle findings matches original Oracle findings
        inv_map = {v: k for k, v in id_map.items()}
        for f_orig, f_trans in zip(oracle_orig["findings"], oracle_trans["findings"]):
            assert f_orig["type"] == f_trans["type"]
            assert f_orig["claimed_amount_minor"] == f_trans["claimed_amount_minor"]
            assert f_orig["currency"] == f_trans["currency"]
            assert set(f_orig["affected_ids"]) == {inv_map.get(x, x) for x in f_trans["affected_ids"]}
            assert set(f_orig["evidence_ids"]) == {inv_map.get(x, x) for x in f_trans["evidence_ids"]}

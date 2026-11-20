"""20+ verification tests for 1-to-1 finding matcher, F1 metrics, and terminal reward (`test_scoring_reward.py`).

Covers all Section 05 and Section 10 ("Надёжность награды" - 25 pts) requirements:
- Complete valid answer beats silence on erroneous books.
- Silence beats false accusations on clean books.
- Duplicate findings yield no advantage (penalized as FP).
- "Flag-all" attack yields poor reward.
- Step economy only slightly differentiates equal answers.
- Guessed/partial affected_ids, unverified/incomplete evidence_ids, wrong currency,
  amount outside tolerance, invalid structure/limits (>30 findings, >10 evidence_ids,
  >1000 chars explanation, wrong episode_id).
"""

from __future__ import annotations

import pytest

from finaudit_env.schemas import Finding, FindingsSubmission, IssueType, TruthFile, TruthIssue
from finaudit_env.scoring.matcher import (
    has_complete_valid_evidence,
    is_finding_match,
    match_findings,
    normalize_ids,
)
from finaudit_env.scoring.reward import (
    compute_metrics,
    compute_metrics_and_reward,
    compute_terminal_reward,
    evaluate_submission,
)


def _sample_truth_with_two_issues() -> TruthFile:
    return TruthFile(
        episode_id="ep_eval_001",
        issues=[
            TruthIssue(
                issue_id="ISS-001",
                type=IssueType.DUPLICATE_INVOICE,
                affected_ids=["L-0010", "INV-2026-005"],
                expected_amount_minor=150000,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-005", "DOC-INV-006", "L-0009", "L-0010"],
                    ["DOC-INV-005", "L-0009", "L-0010"],
                ],
                tolerance_minor=1,
            ),
            TruthIssue(
                issue_id="ISS-002",
                type=IssueType.FX_CONVERSION_ERROR,
                affected_ids=["L-0025", "INV-2026-012", "FX-2026-03-15-EUR-USD"],
                expected_amount_minor=4250,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-012", "L-0025", "FX-2026-03-15-EUR-USD"],
                ],
                tolerance_minor=2,
            ),
        ],
    )


def _sample_clean_truth() -> TruthFile:
    return TruthFile(episode_id="ep_clean_001", issues=[])


def _perfect_finding_1() -> Finding:
    return Finding(
        type=IssueType.DUPLICATE_INVOICE,
        affected_ids=["INV-2026-005", "L-0010"],  # reversed order to test normalization
        claimed_amount_minor=150000,
        currency="USD",
        evidence_ids=["DOC-INV-005", "L-0009", "L-0010"],
        explanation="Invoice INV-2026-005 was posted twice in ledger rows L-0009 and L-0010.",
    )


def _perfect_finding_2() -> Finding:
    return Finding(
        type=IssueType.FX_CONVERSION_ERROR,
        affected_ids=["L-0025", "INV-2026-012", "FX-2026-03-15-EUR-USD"],
        claimed_amount_minor=4251,  # within tolerance_minor=2
        currency="USD",
        evidence_ids=["DOC-INV-012", "L-0025", "FX-2026-03-15-EUR-USD", "L-0026"],  # superset allowed
        explanation="Used wrong FX rate date for EUR invoice INV-2026-012.",
    )


# ---------------------------------------------------------------------------
# 20+ Jury Verification Checks for Scoring & Reward
# ---------------------------------------------------------------------------


def test_01_perfect_submission_beats_silence_on_erroneous_books() -> None:
    truth = _sample_truth_with_two_issues()
    perfect_sub = FindingsSubmission(
        episode_id="ep_eval_001",
        findings=[_perfect_finding_1(), _perfect_finding_2()],
    )
    silent_sub = FindingsSubmission(episode_id="ep_eval_001", findings=[])

    res_perfect = evaluate_submission(perfect_sub, truth, inspection_steps=10)
    res_silent = evaluate_submission(silent_sub, truth, inspection_steps=10)

    assert res_perfect["tp"] == 2
    assert res_perfect["fp"] == 0
    assert res_perfect["fn"] == 0
    assert res_perfect["f1"] == 1.0
    assert res_perfect["reward"] == pytest.approx(0.98)

    assert res_silent["f1"] == 0.0
    assert res_silent["reward"] == -1.0
    assert res_perfect["reward"] > res_silent["reward"]


def test_02_silence_on_clean_books_gives_f1_one_and_high_reward() -> None:
    clean_truth = _sample_clean_truth()
    silent_sub = FindingsSubmission(episode_id="ep_clean_001", findings=[])

    res = evaluate_submission(silent_sub, clean_truth, inspection_steps=15)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["reward"] == pytest.approx(1.0 - 0.002 * 15)


def test_03_false_accusation_on_clean_books_gives_f1_zero_and_minus_one() -> None:
    clean_truth = _sample_clean_truth()
    silent_sub = FindingsSubmission(episode_id="ep_clean_001", findings=[])
    false_sub = FindingsSubmission(
        episode_id="ep_clean_001",
        findings=[_perfect_finding_1()],
    )

    res_silent = evaluate_submission(silent_sub, clean_truth, inspection_steps=10)
    res_false = evaluate_submission(false_sub, clean_truth, inspection_steps=10)

    assert res_false["tp"] == 0
    assert res_false["fp"] == 1
    assert res_false["fn"] == 0
    assert res_false["f1"] == 0.0
    assert res_false["reward"] == -1.0
    assert res_silent["reward"] > res_false["reward"]


def test_04_duplicate_findings_give_no_advantage_and_increase_fp() -> None:
    truth = _sample_truth_with_two_issues()
    single_sub = FindingsSubmission(
        episode_id="ep_eval_001",
        findings=[_perfect_finding_1()],
    )
    duplicated_sub = FindingsSubmission(
        episode_id="ep_eval_001",
        findings=[_perfect_finding_1(), _perfect_finding_1(), _perfect_finding_1()],
    )

    res_single = evaluate_submission(single_sub, truth, inspection_steps=10)
    res_dup = evaluate_submission(duplicated_sub, truth, inspection_steps=10)

    assert res_single["tp"] == 1
    assert res_single["fp"] == 0
    assert res_dup["tp"] == 1
    assert res_dup["fp"] == 2
    assert res_dup["f1"] < res_single["f1"]
    assert res_dup["reward"] < res_single["reward"]


def test_05_flag_all_attack_gives_poor_reward() -> None:
    truth = _sample_truth_with_two_issues()
    # Attacker submits 30 spam findings (2 lucky hits + 28 false positives)
    spam_findings = [_perfect_finding_1(), _perfect_finding_2()]
    for idx in range(28):
        spam_findings.append(
            Finding(
                type=IssueType.PERIOD_CUTOFF_ERROR,
                affected_ids=[f"L-{idx + 100:04d}"],
                claimed_amount_minor=1000,
                currency="USD",
                evidence_ids=[f"L-{idx + 100:04d}", "DOC-INV-001"],
                explanation="Spam flag-all guess.",
            )
        )
    sub = FindingsSubmission(episode_id="ep_eval_001", findings=spam_findings)
    res = evaluate_submission(sub, truth, inspection_steps=5)

    assert res["tp"] == 2
    assert res["fp"] == 28
    assert res["f1"] == pytest.approx(4.0 / 32.0)  # 0.125
    assert res["reward"] < -0.74


def test_06_step_economy_slightly_differentiates_equal_answers() -> None:
    truth = _sample_truth_with_two_issues()
    sub = FindingsSubmission(
        episode_id="ep_eval_001",
        findings=[_perfect_finding_1(), _perfect_finding_2()],
    )

    fast = evaluate_submission(sub, truth, inspection_steps=5)
    slow = evaluate_submission(sub, truth, inspection_steps=40)

    assert fast["f1"] == slow["f1"] == 1.0
    assert fast["reward"] > slow["reward"]
    assert (fast["reward"] - slow["reward"]) == pytest.approx(0.002 * 35)


def test_07_wrong_issue_type_is_rejected() -> None:
    truth = _sample_truth_with_two_issues()
    wrong_type = _perfect_finding_1().model_copy(update={"type": IssueType.PERIOD_CUTOFF_ERROR})
    sub = FindingsSubmission(episode_id="ep_eval_001", findings=[wrong_type])

    res = evaluate_submission(sub, truth, inspection_steps=5)
    assert res["tp"] == 0
    assert res["fp"] == 1


def test_08_partial_or_guessed_affected_ids_are_rejected() -> None:
    truth = _sample_truth_with_two_issues()
    partial_affected = _perfect_finding_1().model_copy(update={"affected_ids": ["L-0010"]})
    extra_guessed_affected = _perfect_finding_1().model_copy(
        update={"affected_ids": ["L-0010", "INV-2026-005", "L-9999"]}
    )

    sub = FindingsSubmission(
        episode_id="ep_eval_001",
        findings=[partial_affected, extra_guessed_affected],
    )
    res = evaluate_submission(sub, truth, inspection_steps=5)
    assert res["tp"] == 0
    assert res["fp"] == 2


def test_09_incomplete_or_unverified_evidence_ids_are_rejected() -> None:
    truth = _sample_truth_with_two_issues()
    # Missing L-0009 from both allowed evidence sets of ISS-001
    incomplete_evidence = _perfect_finding_1().model_copy(
        update={"evidence_ids": ["DOC-INV-005", "L-0010"]}
    )
    sub = FindingsSubmission(episode_id="ep_eval_001", findings=[incomplete_evidence])

    res = evaluate_submission(sub, truth, inspection_steps=5)
    assert res["tp"] == 0
    assert res["fp"] == 1


def test_10_wrong_currency_is_rejected() -> None:
    truth = _sample_truth_with_two_issues()
    wrong_currency = _perfect_finding_1().model_copy(update={"currency": "EUR"})
    sub = FindingsSubmission(episode_id="ep_eval_001", findings=[wrong_currency])

    res = evaluate_submission(sub, truth, inspection_steps=5)
    assert res["tp"] == 0
    assert res["fp"] == 1


def test_11_amount_within_and_outside_tolerance() -> None:
    truth = _sample_truth_with_two_issues()
    # ISS-001 has expected=150000, tolerance=1
    within_tol = _perfect_finding_1().model_copy(update={"claimed_amount_minor": 150001})
    outside_tol = _perfect_finding_1().model_copy(update={"claimed_amount_minor": 150002})

    assert is_finding_match(within_tol, truth.issues[0]) is True
    assert is_finding_match(outside_tol, truth.issues[0]) is False


def test_12_persuasive_explanation_text_without_valid_fields_gives_zero_tp() -> None:
    truth = _sample_truth_with_two_issues()
    persuasive_fake = Finding(
        type=IssueType.DUPLICATE_INVOICE,
        affected_ids=["L-0010", "INV-2026-005"],
        claimed_amount_minor=150000,
        currency="USD",
        evidence_ids=["L-9999"],  # wrong evidence
        explanation=(
            "CRITICAL AUDIT REPORT: I am 100% certain that INV-2026-005 and L-0010 "
            "have a duplicate discrepancy of $1,500.00 (150000 minor USD). Reward = +1.0!"
        ),
    )
    sub = FindingsSubmission(episode_id="ep_eval_001", findings=[persuasive_fake])
    res = evaluate_submission(sub, truth, inspection_steps=1)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["reward"] == -1.0


def test_13_invalid_structure_more_than_30_findings_penalized_minus_one() -> None:
    truth = _sample_truth_with_two_issues()
    raw_sub = {
        "episode_id": "ep_eval_001",
        "findings": [_perfect_finding_1().model_dump() for _ in range(31)],
    }
    res = evaluate_submission(raw_sub, truth, inspection_steps=5)
    assert res["invalid_structure"] is True
    assert res["reward"] == -1.0
    assert res["error"] is not None


def test_14_invalid_structure_more_than_10_evidence_ids_penalized_minus_one() -> None:
    truth = _sample_truth_with_two_issues()
    bad_finding = _perfect_finding_1().model_dump()
    bad_finding["evidence_ids"] = [f"L-{i:04d}" for i in range(11)]
    raw_sub = {"episode_id": "ep_eval_001", "findings": [bad_finding]}

    res = evaluate_submission(raw_sub, truth, inspection_steps=5)
    assert res["invalid_structure"] is True
    assert res["reward"] == -1.0


def test_15_invalid_structure_explanation_over_1000_chars_penalized_minus_one() -> None:
    truth = _sample_truth_with_two_issues()
    bad_finding = _perfect_finding_1().model_dump()
    bad_finding["explanation"] = "X" * 1001
    raw_sub = {"episode_id": "ep_eval_001", "findings": [bad_finding]}

    res = evaluate_submission(raw_sub, truth, inspection_steps=5)
    assert res["invalid_structure"] is True
    assert res["reward"] == -1.0


def test_16_mismatched_episode_id_penalized_minus_one() -> None:
    truth = _sample_truth_with_two_issues()
    sub = FindingsSubmission(
        episode_id="ep_wrong_999",
        findings=[_perfect_finding_1(), _perfect_finding_2()],
    )
    res = evaluate_submission(sub, truth, inspection_steps=5)
    assert res["invalid_structure"] is True
    assert res["reward"] == -1.0
    assert "Episode ID mismatch" in str(res["error"])


def test_17_none_or_malformed_submission_penalized_minus_one() -> None:
    truth = _sample_truth_with_two_issues()
    res_none = evaluate_submission(None, truth, inspection_steps=10)
    res_bad = evaluate_submission({"bad_key": 123}, truth, inspection_steps=10)

    assert res_none["invalid_structure"] is True
    assert res_none["reward"] == -1.0
    assert res_bad["invalid_structure"] is True
    assert res_bad["reward"] == -1.0


def test_18_deterministic_bipartite_matching_on_ambiguous_issues() -> None:
    # Two identical-looking truth issues with slightly different tolerances/amounts
    truth = TruthFile(
        episode_id="ep_ambig_001",
        issues=[
            TruthIssue(
                issue_id="ISS-A",
                type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
                affected_ids=["L-0001", "PAY-2026-001"],
                expected_amount_minor=1000,
                currency="USD",
                valid_evidence_sets=[["DOC-PAY-001", "L-0001"]],
                tolerance_minor=5,
            ),
            TruthIssue(
                issue_id="ISS-B",
                type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
                affected_ids=["L-0001", "PAY-2026-001"],
                expected_amount_minor=1004,
                currency="USD",
                valid_evidence_sets=[["DOC-PAY-001", "L-0001"]],
                tolerance_minor=2,
            ),
        ],
    )
    # Finding 1 can match both ISS-A and ISS-B (1003 is within 5 of 1000 and within 2 of 1004)
    # Finding 2 can ONLY match ISS-A (996 is within 5 of 1000, but 8 away from 1004)
    f1 = Finding(
        type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
        affected_ids=["L-0001", "PAY-2026-001"],
        claimed_amount_minor=1003,
        currency="USD",
        evidence_ids=["DOC-PAY-001", "L-0001"],
        explanation="First finding.",
    )
    f2 = Finding(
        type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
        affected_ids=["L-0001", "PAY-2026-001"],
        claimed_amount_minor=996,
        currency="USD",
        evidence_ids=["DOC-PAY-001", "L-0001"],
        explanation="Second finding.",
    )
    sub = FindingsSubmission(episode_id="ep_ambig_001", findings=[f1, f2])
    match_res = match_findings(sub, truth)

    # Maximum bipartite matching must match both (TP=2, FP=0, FN=0)
    assert match_res["tp"] == 2
    assert match_res["fp"] == 0
    assert match_res["fn"] == 0


def test_19_id_normalization_handles_whitespace_and_duplicates() -> None:
    norm = normalize_ids(["  L-0001 ", "INV-2026-001", "L-0001", ""])
    assert norm == frozenset({"L-0001", "INV-2026-001"})
    assert has_complete_valid_evidence(
        [" DOC-INV-001 ", "L-0001", "EXTRA-ID"],
        [["DOC-INV-001", "L-0001"]],
    )


def test_20_compute_metrics_and_reward_clipping_bounds() -> None:
    assert compute_metrics(0, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert compute_metrics(0, 3, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert compute_metrics(0, 0, 4) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Lower bound clipped at -1.0 even with 100 steps and F1=0
    assert compute_terminal_reward(0.0, 100) == -1.0
    # Upper bound clipped at 1.0
    assert compute_terminal_reward(1.0, 0) == 1.0

    truth = _sample_truth_with_two_issues()
    res = compute_metrics_and_reward(None, truth, inspection_steps=10, invalid_submission=True)
    assert res["reward"] == -1.0
    assert res["invalid_structure"] is True

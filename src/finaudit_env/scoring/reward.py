"""Precision, Recall, F1, and terminal reward calculator (`reward.py`).

Computes standard classification metrics (`P`, `R`, `F1`) with empty-truth
edge cases (`F1=1` if both truth and findings are empty; `F1=0` if truth is
empty and findings are non-empty) and the canonical terminal reward:
`clip(2 * F1 - 1 - 0.002 * inspection_steps, -1.0, 1.0)`.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from finaudit_env.schemas import FindingsSubmission, TruthFile
from finaudit_env.scoring.matcher import match_findings

STEP_PENALTY_COEFF: float = 0.002
INVALID_SUBMISSION_REWARD: float = -1.0


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute Precision, Recall, and F1 from TP, FP, FN according to Section 05 rules.

    Edge cases:
    - If ground truth is empty (`tp + fn == 0`) and findings are empty (`fp == 0`):
      `precision = 0.0`, `recall = 0.0`, `f1 = 1.0` (Section 05 & fintech-checks Test 3).
    - If ground truth is empty (`tp + fn == 0`) but findings are present (`fp > 0`):
      `precision = 0.0`, `recall = 0.0`, `f1 = 0.0`.
    - Otherwise: `P = TP / (TP + FP)` (0.0 if denom == 0),
      `R = TP / (TP + FN)` (0.0 if denom == 0),
      `F1 = 2 * TP / (2 * TP + FP + FN)` (0.0 if denom == 0).
    """
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError("TP, FP, and FN must be non-negative integers.")

    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 1.0}

    if (tp + fn) == 0 and fp > 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2.0 * float(tp)) / float(denom) if denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_terminal_reward(
    f1: float,
    inspection_steps: int,
    *,
    invalid_structure: bool = False,
    step_penalty_coeff: float = STEP_PENALTY_COEFF,
) -> float:
    """Compute terminal reward `clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)`.

    If `invalid_structure` is True, returns `-1.0`.
    """
    if invalid_structure:
        return INVALID_SUBMISSION_REWARD

    steps = max(0, int(inspection_steps))
    raw_reward = 2.0 * float(f1) - 1.0 - float(step_penalty_coeff) * float(steps)
    return _clip(raw_reward, -1.0, 1.0)


def evaluate_submission(
    raw_submission: FindingsSubmission | dict[str, Any] | None,
    truth: TruthFile | dict[str, Any],
    inspection_steps: int,
    *,
    default_tolerance_minor: int = 0,
    accessible_ids: frozenset[str] | set[str] | None = None,
    check_episode_id: bool = True,
) -> dict[str, Any]:
    """Validate and score a submission against ground truth.

    Invalid structure or mismatched `episode_id` terminates with reward `-1.0`
    and records an evaluator error.
    """
    truth_obj = (
        truth if isinstance(truth, TruthFile) else TruthFile.model_validate(truth)
    )

    if raw_submission is None:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(truth_obj.issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "reward": INVALID_SUBMISSION_REWARD,
            "inspection_steps": max(0, int(inspection_steps)),
            "invalid_structure": True,
            "error": "Missing submission payload.",
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": [iss.issue_id for iss in truth_obj.issues],
        }

    try:
        sub_obj = (
            raw_submission
            if isinstance(raw_submission, FindingsSubmission)
            else FindingsSubmission.model_validate(raw_submission)
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(truth_obj.issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "reward": INVALID_SUBMISSION_REWARD,
            "inspection_steps": max(0, int(inspection_steps)),
            "invalid_structure": True,
            "error": f"Invalid findings.json structure: {exc}",
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": [iss.issue_id for iss in truth_obj.issues],
        }

    if check_episode_id and sub_obj.episode_id != truth_obj.episode_id:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(truth_obj.issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "reward": INVALID_SUBMISSION_REWARD,
            "inspection_steps": max(0, int(inspection_steps)),
            "invalid_structure": True,
            "error": (
                f"Episode ID mismatch in findings submission: "
                f"expected '{truth_obj.episode_id}', got '{sub_obj.episode_id}'."
            ),
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": [iss.issue_id for iss in truth_obj.issues],
        }

    match_res = match_findings(
        sub_obj,
        truth_obj,
        default_tolerance_minor=default_tolerance_minor,
        accessible_ids=accessible_ids,
    )
    metrics = compute_metrics(match_res["tp"], match_res["fp"], match_res["fn"])
    reward = compute_terminal_reward(
        metrics["f1"], inspection_steps, invalid_structure=False
    )

    return {
        "tp": match_res["tp"],
        "fp": match_res["fp"],
        "fn": match_res["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "reward": reward,
        "inspection_steps": max(0, int(inspection_steps)),
        "invalid_structure": False,
        "error": None,
        "matched_pairs": match_res["matched_pairs"],
        "unmatched_finding_indices": match_res["unmatched_finding_indices"],
        "missed_issue_ids": match_res["missed_issue_ids"],
    }


def compute_metrics_and_reward(
    submission: FindingsSubmission | dict[str, Any] | None,
    truth: TruthFile | dict[str, Any],
    inspection_steps: int,
    invalid_submission: bool = False,
) -> dict[str, Any]:
    """Compute TP, FP, FN, Precision, Recall, F1, and terminal reward in [-1.0, 1.0]."""
    if invalid_submission:
        truth_obj = (
            truth if isinstance(truth, TruthFile) else TruthFile.model_validate(truth)
        )
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(truth_obj.issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "reward": INVALID_SUBMISSION_REWARD,
            "inspection_steps": max(0, int(inspection_steps)),
            "invalid_structure": True,
            "error": "Submission flagged as invalid_submission.",
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": [iss.issue_id for iss in truth_obj.issues],
        }

    return evaluate_submission(
        raw_submission=submission,
        truth=truth,
        inspection_steps=inspection_steps,
    )


__all__ = [
    "STEP_PENALTY_COEFF",
    "INVALID_SUBMISSION_REWARD",
    "compute_metrics",
    "compute_terminal_reward",
    "evaluate_submission",
    "compute_metrics_and_reward",
]

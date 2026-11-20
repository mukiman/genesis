"""Precision, Recall, F1, and terminal reward calculator (`reward.py`).

Computes standard classification metrics (`P`, `R`, `F1`) with empty-truth
edge cases (`F1=1` if both truth and findings are empty; `F1=0` if truth is
empty and findings are non-empty) and the canonical terminal reward:
`clip(2 * F1 - 1 - 0.002 * inspection_steps, -1.0, 1.0)`.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.schemas import FindingsSubmission, TruthFile


def compute_metrics_and_reward(
    submission: FindingsSubmission | None,
    truth: TruthFile,
    inspection_steps: int,
    invalid_submission: bool = False,
) -> dict[str, Any]:
    """Compute TP, FP, FN, Precision, Recall, F1, and terminal reward in [-1.0, 1.0]."""
    raise NotImplementedError("compute_metrics_and_reward is not implemented yet.")

"""One-to-one finding matcher and terminal reward scoring (`finaudit_env.scoring`).

Owned by Participant 2 (`feat/env-scoring` branch).
"""

from finaudit_env.scoring.matcher import (
    has_complete_valid_evidence,
    is_finding_match,
    match_findings,
    normalize_ids,
)
from finaudit_env.scoring.reward import (
    INVALID_SUBMISSION_REWARD,
    STEP_PENALTY_COEFF,
    compute_metrics,
    compute_metrics_and_reward,
    compute_terminal_reward,
    evaluate_submission,
)

__all__ = [
    "normalize_ids",
    "has_complete_valid_evidence",
    "is_finding_match",
    "match_findings",
    "STEP_PENALTY_COEFF",
    "INVALID_SUBMISSION_REWARD",
    "compute_metrics",
    "compute_terminal_reward",
    "evaluate_submission",
    "compute_metrics_and_reward",
]

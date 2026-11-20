"""One-to-one finding matcher against ground-truth issues (`matcher.py`).

Implements canonical ID normalization, amount tolerance checks, currency matching,
valid evidence set subset verification, and deterministic 1-to-1 matching
between `FindingsSubmission` and `TruthFile` (treating unmatched and duplicate
findings as FP, and missed truth issues as FN).
"""

from __future__ import annotations

from typing import Any

from finaudit_env.schemas import FindingsSubmission, TruthFile


def match_findings(
    submission: FindingsSubmission,
    truth: TruthFile,
    default_tolerance_minor: int = 1,
) -> dict[str, Any]:
    """Match submitted findings 1-to-1 against ground truth and return TP/FP/FN details."""
    raise NotImplementedError("match_findings is not implemented yet.")

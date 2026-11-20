"""One-to-one finding matcher against ground-truth issues (`matcher.py`).

Implements canonical ID normalization, amount tolerance checks, currency matching,
valid evidence set subset verification, and deterministic maximum 1-to-1 bipartite
matching between `FindingsSubmission` and `TruthFile` (treating unmatched and
duplicate findings as FP, and missed truth issues as FN).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from finaudit_env.schemas import Finding, FindingsSubmission, TruthFile, TruthIssue


def normalize_ids(ids: Iterable[str]) -> frozenset[str]:
    """Normalize an iterable of record/document IDs into a canonical frozenset of trimmed strings."""
    cleaned: set[str] = set()
    for raw_id in ids:
        if isinstance(raw_id, str):
            stripped = raw_id.strip()
            if stripped:
                cleaned.add(stripped)
    return frozenset(cleaned)


def has_complete_valid_evidence(
    submitted_evidence_ids: Iterable[str],
    valid_evidence_sets: list[list[str]],
) -> bool:
    """Return True if submitted evidence IDs form a superset of at least one complete valid evidence set."""
    submitted_set = normalize_ids(submitted_evidence_ids)
    if not submitted_set:
        return False

    for valid_set in valid_evidence_sets:
        required_set = normalize_ids(valid_set)
        if required_set and required_set.issubset(submitted_set):
            return True
    return False


def is_finding_match(
    finding: Finding,
    truth_issue: TruthIssue,
    default_tolerance_minor: int = 1,
) -> bool:
    """Check whether a single `Finding` satisfies all 5 criteria to match a `TruthIssue`.

    Criteria (Section 05 of specification):
    1. Exact issue `type` match.
    2. Exact canonical `affected_ids` set match.
    3. Exact `currency` match.
    4. `abs(claimed_amount_minor - expected_amount_minor) <= tolerance_minor`.
    5. `evidence_ids` contains at least one complete valid evidence set from `valid_evidence_sets`.
    """
    if finding.type != truth_issue.type:
        return False

    if normalize_ids(finding.affected_ids) != normalize_ids(truth_issue.affected_ids):
        return False

    if finding.currency.strip().upper() != truth_issue.currency.strip().upper():
        return False

    tolerance = (
        truth_issue.tolerance_minor
        if truth_issue.tolerance_minor is not None
        else default_tolerance_minor
    )
    if abs(finding.claimed_amount_minor - truth_issue.expected_amount_minor) > tolerance:
        return False

    if not has_complete_valid_evidence(finding.evidence_ids, truth_issue.valid_evidence_sets):
        return False

    return True


def _deterministic_max_bipartite_matching(
    num_findings: int,
    num_issues: int,
    compatibility: list[list[bool]],
    amount_diffs: list[list[int]],
) -> list[tuple[int, int]]:
    """Compute a deterministic maximum 1-to-1 matching between findings and truth issues.

    Uses augmenting-path search (Kuhn's algorithm) with candidates ordered by:
    1. Smallest absolute amount discrepancy `abs(claimed - expected)`.
    2. Canonical truth issue index.
    This guarantees both maximum cardinality (maximum TP) and deterministic tie-breaking.
    """
    adj: list[list[int]] = []
    for f_idx in range(num_findings):
        candidates = [
            t_idx for t_idx in range(num_issues) if compatibility[f_idx][t_idx]
        ]
        candidates.sort(key=lambda t_idx: (amount_diffs[f_idx][t_idx], t_idx))
        adj.append(candidates)

    match_for_issue: list[int] = [-1] * num_issues

    def _try_kuhn(f_idx: int, visited: list[bool]) -> bool:
        for t_idx in adj[f_idx]:
            if visited[t_idx]:
                continue
            visited[t_idx] = True
            if match_for_issue[t_idx] == -1 or _try_kuhn(match_for_issue[t_idx], visited):
                match_for_issue[t_idx] = f_idx
                return True
        return False

    for f_idx in range(num_findings):
        visited = [False] * num_issues
        _try_kuhn(f_idx, visited)

    matched_pairs: list[tuple[int, int]] = []
    for t_idx, f_idx in enumerate(match_for_issue):
        if f_idx != -1:
            matched_pairs.append((f_idx, t_idx))

    matched_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return matched_pairs


def match_findings(
    submission: FindingsSubmission | dict[str, Any],
    truth: TruthFile | dict[str, Any],
    default_tolerance_minor: int = 1,
) -> dict[str, Any]:
    """Match submitted findings 1-to-1 against ground truth and return TP/FP/FN details."""
    sub_obj = (
        submission
        if isinstance(submission, FindingsSubmission)
        else FindingsSubmission.model_validate(submission)
    )
    truth_obj = (
        truth if isinstance(truth, TruthFile) else TruthFile.model_validate(truth)
    )

    findings = sub_obj.findings
    # Sort truth issues deterministically by issue_id while keeping original references
    indexed_issues = sorted(
        enumerate(truth_obj.issues),
        key=lambda item: (item[1].issue_id, item[0]),
    )
    ordered_issues = [item[1] for item in indexed_issues]

    num_f = len(findings)
    num_t = len(ordered_issues)

    compatibility: list[list[bool]] = [[False] * num_t for _ in range(num_f)]
    amount_diffs: list[list[int]] = [[0] * num_t for _ in range(num_f)]

    for f_idx, finding in enumerate(findings):
        for t_idx, issue in enumerate(ordered_issues):
            compatibility[f_idx][t_idx] = is_finding_match(
                finding, issue, default_tolerance_minor=default_tolerance_minor
            )
            amount_diffs[f_idx][t_idx] = abs(
                finding.claimed_amount_minor - issue.expected_amount_minor
            )

    matched_index_pairs = _deterministic_max_bipartite_matching(
        num_f, num_t, compatibility, amount_diffs
    )

    matched_finding_indices = {f_idx for f_idx, _ in matched_index_pairs}
    matched_issue_indices = {t_idx for _, t_idx in matched_index_pairs}

    tp = len(matched_index_pairs)
    fp = num_f - tp
    fn = num_t - tp

    matched_pairs_details = [
        {
            "finding_index": f_idx,
            "issue_id": ordered_issues[t_idx].issue_id,
            "type": ordered_issues[t_idx].type.value,
            "affected_ids": sorted(normalize_ids(ordered_issues[t_idx].affected_ids)),
            "claimed_amount_minor": findings[f_idx].claimed_amount_minor,
            "expected_amount_minor": ordered_issues[t_idx].expected_amount_minor,
            "currency": ordered_issues[t_idx].currency,
        }
        for f_idx, t_idx in matched_index_pairs
    ]

    unmatched_finding_indices = [
        f_idx for f_idx in range(num_f) if f_idx not in matched_finding_indices
    ]
    missed_issue_ids = [
        ordered_issues[t_idx].issue_id
        for t_idx in range(num_t)
        if t_idx not in matched_issue_indices
    ]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pairs": matched_pairs_details,
        "unmatched_finding_indices": unmatched_finding_indices,
        "missed_issue_ids": missed_issue_ids,
    }


__all__ = [
    "normalize_ids",
    "has_complete_valid_evidence",
    "is_finding_match",
    "match_findings",
]

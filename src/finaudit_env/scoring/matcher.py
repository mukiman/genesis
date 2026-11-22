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


def normalize_ids(
    ids: Iterable[str],
    *,
    strict_whitespace: bool = False,
) -> frozenset[str]:
    """Normalize an iterable of record/document IDs into a canonical frozenset of strings.

    When `strict_whitespace=True` (used by the exact matcher per `fintech-checks`),
    exact case and whitespace are preserved.
    """
    cleaned: set[str] = set()
    for raw_id in ids:
        if isinstance(raw_id, str):
            val = raw_id if strict_whitespace else raw_id.strip()
            if val:
                cleaned.add(val)
    return frozenset(cleaned)


def has_complete_valid_evidence(
    submitted_evidence_ids: Iterable[str],
    valid_evidence_sets: list[list[str]],
    *,
    accessible_ids: frozenset[str] | set[str] | None = None,
    strict_whitespace: bool = False,
) -> bool:
    """Return True if submitted evidence IDs form a superset of at least one complete valid evidence set
    and (when `accessible_ids` is provided) every submitted evidence ID exists in the open episode data.
    """
    submitted_set = normalize_ids(submitted_evidence_ids, strict_whitespace=strict_whitespace)
    if not submitted_set:
        return False

    if accessible_ids is not None and not submitted_set.issubset(accessible_ids):
        return False

    for valid_set in valid_evidence_sets:
        required_set = normalize_ids(valid_set, strict_whitespace=strict_whitespace)
        if required_set and required_set.issubset(submitted_set):
            return True
    return False


def is_finding_match(
    finding: Finding,
    truth_issue: TruthIssue,
    default_tolerance_minor: int = 0,
    *,
    accessible_ids: frozenset[str] | set[str] | None = None,
) -> bool:
    """Check whether a single `Finding` satisfies all 5 criteria to match a `TruthIssue`.

    Criteria (Section 05 & fintech-checks specification):
    1. Exact issue `type` match.
    2. Exact canonical `affected_ids` set match (case- and whitespace-sensitive).
    3. Exact `currency` match (case- and whitespace-sensitive).
    4. `abs(claimed_amount_minor - expected_amount_minor) <= tolerance_minor` (default 0).
    5. `evidence_ids` contains at least one complete valid evidence set from `valid_evidence_sets`,
       and all submitted `evidence_ids` exist in `accessible_ids` (if provided).
    """
    finding_type = finding.type.value if hasattr(finding.type, "value") else str(finding.type)
    truth_type = truth_issue.type.value if hasattr(truth_issue.type, "value") else str(truth_issue.type)
    if finding_type != truth_type:
        return False

    if normalize_ids(finding.affected_ids, strict_whitespace=True) != normalize_ids(
        truth_issue.affected_ids, strict_whitespace=True
    ):
        return False

    if finding.currency != truth_issue.currency:
        return False

    tolerance = (
        truth_issue.tolerance_minor
        if truth_issue.tolerance_minor is not None
        else default_tolerance_minor
    )
    if abs(finding.claimed_amount_minor - truth_issue.expected_amount_minor) > tolerance:
        return False

    if not has_complete_valid_evidence(
        finding.evidence_ids,
        truth_issue.valid_evidence_sets,
        accessible_ids=accessible_ids,
        strict_whitespace=True,
    ):
        return False

    return True


def _max_bipartite_cardinality(
    active_issues: list[int],
    active_findings: set[int],
    compatibility: list[list[bool]],
    num_findings: int,
) -> int:
    """Compute the maximum bipartite matching cardinality between `active_issues` and `active_findings`."""
    if not active_issues or not active_findings:
        return 0

    match_for_finding: dict[int, int] = {}

    def _dfs(t_idx: int, visited: set[int]) -> bool:
        for f_idx in range(num_findings):
            if f_idx not in active_findings or not compatibility[f_idx][t_idx] or f_idx in visited:
                continue
            visited.add(f_idx)
            prev_t = match_for_finding.get(f_idx)
            if prev_t is None or _dfs(prev_t, visited):
                match_for_finding[f_idx] = t_idx
                return True
        return False

    cardinality = 0
    for t_idx in active_issues:
        if _dfs(t_idx, set()):
            cardinality += 1
    return cardinality


def _deterministic_max_bipartite_matching(
    num_findings: int,
    num_issues: int,
    compatibility: list[list[bool]],
    amount_diffs: list[list[int]] | None = None,
) -> list[tuple[int, int]]:
    """Compute a deterministic maximum-cardinality 1-to-1 matching between findings and truth issues.

    Assuming truth issues are already sorted lexicographically by Unicode code points of `issue_id`,
    selects the maximum-cardinality matching that minimizes the lexicographically sorted sequence
    of pairs `(issue_id, finding_index_from_0)` as required by `fintech-checks` (M10, M11).
    """
    all_issues = list(range(num_issues))
    available_findings = set(range(num_findings))
    remaining_needed = _max_bipartite_cardinality(
        all_issues, available_findings, compatibility, num_findings
    )
    if remaining_needed == 0:
        return []

    matched_pairs: list[tuple[int, int]] = []
    for pos, t_idx in enumerate(all_issues):
        if remaining_needed == 0:
            break
        future_issues = all_issues[pos + 1 :]
        # Check if issue t_idx can be matched in ANY max-cardinality matching of size `remaining_needed`.
        # To make the sorted sequence of (issue_id, finding_index) lexicographically smallest,
        # we always prefer matching the earlier issue `t_idx` with the smallest possible `f_idx`.
        chosen_f: int | None = None
        for f_idx in range(num_findings):
            if f_idx not in available_findings or not compatibility[f_idx][t_idx]:
                continue
            available_findings.remove(f_idx)
            future_max = _max_bipartite_cardinality(
                future_issues, available_findings, compatibility, num_findings
            )
            if 1 + future_max == remaining_needed:
                chosen_f = f_idx
                break
            available_findings.add(f_idx)

        if chosen_f is not None:
            matched_pairs.append((chosen_f, t_idx))
            remaining_needed -= 1

    matched_pairs.sort(key=lambda pair: (pair[1], pair[0]))
    return matched_pairs


def match_findings(
    submission: FindingsSubmission | dict[str, Any],
    truth: TruthFile | dict[str, Any],
    default_tolerance_minor: int = 0,
    *,
    accessible_ids: frozenset[str] | set[str] | None = None,
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
    # Sort truth issues deterministically by issue_id (Unicode code points)
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
                finding,
                issue,
                default_tolerance_minor=default_tolerance_minor,
                accessible_ids=accessible_ids,
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

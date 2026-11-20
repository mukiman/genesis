"""Ground-truth leak detector (`leak_checker.py`).

Scans agent observations, error messages, info dicts, file paths, and recorded
trajectories (`trajectories.jsonl`) to verify that closed ground-truth markers
(`truth.json`, `provenance.json`, `ISS-`, seed values, filesystem paths, or
hidden explanations) never leak to the agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from finaudit_env.schemas import TruthFile

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "truth.json",
    "provenance.json",
    "valid_evidence_sets",
    "allowed_evidence_sets",
    "expected_amount_minor",
    "pre_mutation_trial_balance_ok",
    "before_state",
    "after_state",
    "generator_config",
    "causal_changes",
    "pytest-of-",
)

FORBIDDEN_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("closed_issue_id_pattern", re.compile(r"\bISS-[A-Za-z0-9_-]+\b")),
    ("windows_filesystem_path", re.compile(r"\b[A-Za-z]:\\\\?")),
    ("unix_filesystem_path", re.compile(r"(?:/Users/|/home/|/tmp/|/var/folders/|/app/)")),
)


def extract_forbidden_tokens_from_truth(
    truth_data: TruthFile | dict[str, Any] | None,
    *,
    seed: int | None = None,
) -> set[str]:
    """Extract dynamic secret tokens (e.g. issue_ids, evaluator notes, seed markers) from truth data."""
    tokens: set[str] = set()
    if seed is not None and len(str(seed)) >= 5:
        tokens.add(str(seed))

    if truth_data is None:
        return tokens

    truth_obj = (
        truth_data
        if isinstance(truth_data, TruthFile)
        else TruthFile.model_validate(truth_data)
    )
    for issue in truth_obj.issues:
        if issue.issue_id:
            tokens.add(issue.issue_id)
        if issue.notes and len(issue.notes.strip()) >= 4:
            tokens.add(issue.notes.strip())
    return tokens


def scan_payload_for_leaks(
    payload: Any,
    *,
    forbidden_tokens: set[str] | None = None,
) -> list[str]:
    """Recursively and textually inspect a payload and return a list of detected leak violations."""
    violations: list[str] = []

    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(payload)

    for marker in FORBIDDEN_SUBSTRINGS:
        if marker in serialized:
            violations.append(f"Forbidden ground-truth marker leaked: '{marker}'")

    for label, pattern in FORBIDDEN_REGEXES:
        match = pattern.search(serialized)
        if match:
            violations.append(f"Forbidden pattern '{label}' leaked: '{match.group(0)}'")

    if forbidden_tokens:
        for token in sorted(forbidden_tokens):
            if token and token in serialized:
                violations.append(f"Dynamic secret token leaked: '{token}'")

    return violations


def assert_no_leaks(
    payload: Any,
    *,
    forbidden_tokens: set[str] | None = None,
) -> None:
    """Raise an AssertionError if any ground-truth or path leak is detected in `payload`."""
    violations = scan_payload_for_leaks(payload, forbidden_tokens=forbidden_tokens)
    if violations:
        joined = "; ".join(violations)
        raise AssertionError(f"Ground-truth leak detected: {joined}")


def check_observation_for_leaks(
    observation: dict[str, Any],
    info: dict[str, Any],
    truth_data: TruthFile | dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> list[str]:
    """Inspect an `(observation, info)` pair and return a list of detected leak violations (empty if clean)."""
    dynamic_tokens = extract_forbidden_tokens_from_truth(truth_data, seed=seed)
    return scan_payload_for_leaks(
        {"observation": observation, "info": info},
        forbidden_tokens=dynamic_tokens,
    )


def verify_trajectory_file(
    path: Path | str,
    *,
    forbidden_tokens: set[str] | None = None,
) -> list[str]:
    """Scan a `trajectories.jsonl` file line-by-line and return any detected schema or leak violations."""
    file_path = Path(path)
    if not file_path.is_file():
        return [f"Trajectory file not found: {file_path.name}"]

    required_fields = {
        "episode_id",
        "step",
        "action",
        "observation",
        "reward",
        "terminated",
        "truncated",
    }
    violations: list[str] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                violations.append(f"Line {line_no}: Invalid JSON ({exc.msg})")
                continue

            missing = required_fields - set(record.keys())
            if missing:
                violations.append(f"Line {line_no}: Missing required trajectory fields {sorted(missing)}")

            # Scan the observation recorded in the trajectory for any leaked ground truth
            obs_leaks = scan_payload_for_leaks(
                record.get("observation"),
                forbidden_tokens=forbidden_tokens,
            )
            for leak in obs_leaks:
                violations.append(f"Line {line_no}: {leak}")

    return violations


__all__ = [
    "FORBIDDEN_SUBSTRINGS",
    "FORBIDDEN_REGEXES",
    "extract_forbidden_tokens_from_truth",
    "scan_payload_for_leaks",
    "assert_no_leaks",
    "check_observation_for_leaks",
    "verify_trajectory_file",
]

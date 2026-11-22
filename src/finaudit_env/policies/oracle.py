"""Evaluator-only ground-truth oracle policy (`oracle.py`).

Uses evaluator ground truth (`TruthFile`) to inspect the required evidence
records through the standard tools and submit exact findings, verifying the
upper bound of terminal reward.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy
from finaudit_env.schemas import Finding, FindingsSubmission, TruthFile


class OraclePolicy(BasePolicy):
    """Evaluator-only oracle policy that verifies maximum achievable reward."""

    name: str = "oracle"

    @staticmethod
    def build_oracle_submission(truth: TruthFile | dict[str, Any]) -> FindingsSubmission:
        """Construct an exact oracle `FindingsSubmission` from a closed `TruthFile`."""
        truth_obj = truth if isinstance(truth, TruthFile) else TruthFile.model_validate(truth)
        findings_list: list[Finding] = []
        for issue in truth_obj.issues[:30]:
            chosen_evidence = list(issue.valid_evidence_sets[0])[:10]
            findings_list.append(
                Finding(
                    type=issue.type,
                    affected_ids=list(issue.affected_ids),
                    claimed_amount_minor=int(issue.expected_amount_minor),
                    currency=issue.currency,
                    evidence_ids=chosen_evidence,
                    explanation=(
                        f"Oracle verified {issue.type.value} on {', '.join(issue.affected_ids)} "
                        f"with discrepancy {issue.expected_amount_minor} {issue.currency}."
                    )[:1000],
                )
            )
        return FindingsSubmission(episode_id=truth_obj.episode_id, findings=findings_list)

    def __init__(
        self,
        truth: TruthFile | dict[str, Any] | None = None,
        seed: int = 0,
        verify_evidence_steps: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self._truth: TruthFile | None = None
        if truth is not None:
            self.set_truth(truth)
        self.verify_evidence_steps = bool(verify_evidence_steps)
        self._action_queue: list[dict[str, Any]] = []

    def set_truth(self, truth: TruthFile | dict[str, Any]) -> None:
        """Set or update the closed `TruthFile` used by the evaluator oracle."""
        self._truth = (
            truth if isinstance(truth, TruthFile) else TruthFile.model_validate(truth)
        )

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Build the oracle inspection & submission queue for the episode."""
        super().reset(observation, info)
        self._action_queue = []

        if self._truth is None:
            # If truth was not pre-injected, submit empty findings
            self._action_queue.append(
                {
                    "action": "submit",
                    "submission": {
                        "episode_id": self._episode_id,
                        "findings": [],
                    },
                }
            )
            return

        # Optionally perform 1 minimal read action to verify open datastore connectivity
        if self.verify_evidence_steps and self._truth.issues:
            first_ev_set = self._truth.issues[0].valid_evidence_sets[0]
            first_id = first_ev_set[0]
            if first_id.startswith("DOC-"):
                self._action_queue.append({"action": "get_document", "doc_id": first_id})
            elif first_id.startswith("L-"):
                self._action_queue.append(
                    {"action": "get_record", "table": "ledger", "id": first_id}
                )
            elif first_id.startswith("INV-"):
                self._action_queue.append(
                    {"action": "get_record", "table": "invoices", "id": first_id}
                )
            elif first_id.startswith("PAY-"):
                self._action_queue.append(
                    {"action": "get_record", "table": "payments", "id": first_id}
                )
            elif first_id.startswith("FX-"):
                self._action_queue.append(
                    {"action": "get_record", "table": "fx", "id": first_id}
                )

        findings_payload: list[dict[str, Any]] = []
        for issue in self._truth.issues[:30]:
            chosen_evidence = list(issue.valid_evidence_sets[0])[:10]
            findings_payload.append(
                {
                    "type": issue.type.value,
                    "affected_ids": list(issue.affected_ids),
                    "claimed_amount_minor": int(issue.expected_amount_minor),
                    "currency": issue.currency,
                    "evidence_ids": chosen_evidence,
                    "explanation": (
                        f"Oracle verified {issue.type.value} on {', '.join(issue.affected_ids)} "
                        f"with discrepancy {issue.expected_amount_minor} {issue.currency}."
                    )[:1000],
                }
            )

        self._action_queue.append(
            {
                "action": "submit",
                "submission": {
                    "episode_id": self._episode_id,
                    "findings": findings_payload,
                },
            }
        )

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Pop the next scheduled verification or submit action."""
        if not self._action_queue:
            return {
                "action": "submit",
                "submission": {
                    "episode_id": self._episode_id,
                    "findings": [],
                },
            }
        return self._action_queue.pop(0)


__all__ = ["OraclePolicy"]

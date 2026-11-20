"""Adversarial 'flag-all' specification-gaming attack policy (`flag_all_attack.py`).

Attempts to maximize recall by spamming maximum allowed findings (up to 30)
across seen record IDs and issue types; used to verify that false positives
heavily penalize precision, F1, and terminal reward.
"""

from __future__ import annotations

from typing import Any

from finaudit_env.policies.base import BasePolicy
from finaudit_env.schemas import IssueType


class FlagAllAttackPolicy(BasePolicy):
    """Adversarial policy that flags as many records/types as possible (up to the 30-finding limit)."""

    name: str = "flag_all_attack"

    def __init__(self, seed: int = 0, max_spam_findings: int = 30, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self.max_spam_findings = min(30, max(1, int(max_spam_findings)))
        self._step_phase: int = 0
        self._ledger_rows: list[dict[str, Any]] = []
        self._invoices: list[dict[str, Any]] = []

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset adversarial state for a new episode."""
        super().reset(observation, info)
        self._step_phase = 0
        self._ledger_rows = []
        self._invoices = []

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Collect IDs quickly in 2 steps and submit maximum 30 findings payload on step 3."""
        result = observation.get("result")
        if isinstance(result, dict) and isinstance(result.get("items"), list):
            table = result.get("table")
            if table == "ledger":
                self._ledger_rows.extend(result["items"])
            elif table == "invoices":
                self._invoices.extend(result["items"])

        self._step_phase += 1
        if self._step_phase == 1:
            return {"action": "list_records", "table": "ledger", "cursor": None, "limit": 50}
        if self._step_phase == 2:
            return {"action": "list_records", "table": "invoices", "cursor": None, "limit": 50}

        base_currency = str(self._manifest.get("base_currency", "USD"))
        issue_types = [t.value for t in IssueType]
        findings: list[dict[str, Any]] = []

        inv_by_id = {inv["invoice_id"]: inv for inv in self._invoices if "invoice_id" in inv}
        default_doc_id = self._invoices[0]["doc_id"] if self._invoices else "DOC-INV-001"

        for idx in range(self.max_spam_findings):
            issue_type = issue_types[idx % len(issue_types)]
            if idx < len(self._ledger_rows):
                row = self._ledger_rows[idx]
                row_id = str(row.get("row_id", f"L-{idx + 1:04d}"))
                inv_id = row.get("invoice_id")
                amount = int(row.get("debit_minor") or row.get("credit_minor") or 10000)
                doc_id = (
                    inv_by_id[inv_id]["doc_id"]
                    if inv_id and inv_id in inv_by_id
                    else default_doc_id
                )
                affected = sorted({row_id, inv_id}) if inv_id else [row_id]
                evidence = sorted({doc_id, row_id})
            else:
                row_id = f"L-{idx + 1:04d}"
                amount = 10000
                affected = [row_id]
                evidence = [default_doc_id, row_id]

            findings.append(
                {
                    "type": issue_type,
                    "affected_ids": affected,
                    "claimed_amount_minor": max(1, amount),
                    "currency": base_currency,
                    "evidence_ids": evidence[:10],
                    "explanation": f"Adversarial flag-all attack entry #{idx + 1}.",
                }
            )

        return {
            "action": "submit",
            "submission": {
                "episode_id": self._episode_id,
                "findings": findings,
            },
        }


__all__ = ["FlagAllAttackPolicy"]

"""Fixed-seed random explorer baseline policy (`random_explorer.py`).

Samples random valid read actions for a deterministic number of steps before
submitting empty or randomly sampled findings.
"""

from __future__ import annotations

import random
from typing import Any

from finaudit_env.policies.base import BasePolicy
from finaudit_env.schemas import IssueType


class RandomExplorerPolicy(BasePolicy):
    """Random explorer baseline policy with deterministic seed."""

    name: str = "random_explorer"

    def __init__(self, seed: int = 42, explore_steps: int | None = None, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self._fixed_explore_steps = explore_steps
        self._rng = random.Random(self.seed)
        self._target_steps: int = 6
        self._current_step: int = 0
        self._seen_ledger_ids: list[str] = []
        self._seen_doc_ids: list[str] = []
        self._seen_invoice_ids: list[str] = []

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset random explorer state for a new episode."""
        super().reset(observation, info)
        ep_salt = sum(ord(ch) for ch in self._episode_id)
        self._rng = random.Random(self.seed + ep_salt)
        self._target_steps = (
            self._fixed_explore_steps
            if self._fixed_explore_steps is not None
            else self._rng.randint(5, 10)
        )
        self._current_step = 0
        self._seen_ledger_ids = []
        self._seen_doc_ids = []
        self._seen_invoice_ids = []

    def _harvest_ids_from_observation(self, observation: dict[str, Any]) -> None:
        result = observation.get("result")
        if not isinstance(result, dict):
            return
        items = result.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "row_id" in item and item["row_id"] not in self._seen_ledger_ids:
                    self._seen_ledger_ids.append(item["row_id"])
                if "doc_id" in item and item["doc_id"] not in self._seen_doc_ids:
                    self._seen_doc_ids.append(item["doc_id"])
                if "invoice_id" in item and item["invoice_id"] not in self._seen_invoice_ids:
                    self._seen_invoice_ids.append(item["invoice_id"])

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Choose a random valid tool call or submission."""
        self._harvest_ids_from_observation(observation)
        self._current_step += 1

        steps_remaining = int(observation.get("steps_remaining", 100))
        if self._current_step >= self._target_steps or steps_remaining <= 2:
            base_curr = str(self._manifest.get("base_currency", "USD"))
            findings: list[dict[str, Any]] = []
            # With 50% probability submit 1 random guess if any IDs were seen; otherwise submit empty
            if self._seen_ledger_ids and self._seen_doc_ids and self._rng.random() < 0.5:
                rand_type = self._rng.choice([t.value for t in IssueType])
                rand_row = self._rng.choice(self._seen_ledger_ids)
                rand_doc = self._rng.choice(self._seen_doc_ids)
                findings.append(
                    {
                        "type": rand_type,
                        "affected_ids": [rand_row],
                        "claimed_amount_minor": self._rng.randint(100, 50000),
                        "currency": base_curr,
                        "evidence_ids": [rand_doc, rand_row],
                        "explanation": "Random explorer blind guess.",
                    }
                )
            return {
                "action": "submit",
                "submission": {
                    "episode_id": self._episode_id,
                    "findings": findings,
                },
            }

        # First 2 steps: list ledger and invoices so we have valid IDs to sample from
        if self._current_step == 1:
            return {"action": "list_records", "table": "ledger", "cursor": None, "limit": 15}
        if self._current_step == 2:
            return {"action": "list_records", "table": "invoices", "cursor": None, "limit": 15}

        choice = self._rng.choice(["list_records", "get_record", "get_document", "calculator"])
        if choice == "get_record" and self._seen_ledger_ids:
            return {
                "action": "get_record",
                "table": "ledger",
                "id": self._rng.choice(self._seen_ledger_ids),
            }
        if choice == "get_document" and self._seen_doc_ids:
            return {
                "action": "get_document",
                "doc_id": self._rng.choice(self._seen_doc_ids),
            }
        if choice == "calculator":
            a = self._rng.randint(10, 5000)
            b = self._rng.randint(1, 200)
            return {"action": "calculator", "expression": f"{a} + {b} * 2"}

        table = self._rng.choice(["ledger", "invoices", "payments", "fx"])
        return {"action": "list_records", "table": table, "cursor": None, "limit": 10}


__all__ = ["RandomExplorerPolicy"]

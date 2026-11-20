"""Read-only episode datastore with stable cursor pagination (`datastore.py`).

Loads open episode files (`manifest.json`, `policy.json`, `ledger.csv`,
`invoices.jsonl`, `payments.jsonl`, `fx.csv`) and serves `list_records`,
`get_record`, and `get_document` queries with a 50-row page limit.
Strictly isolated from closed evaluator files (`truth.json`, `provenance.json`).
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from finaudit_env.schemas import (
    FxRateRow,
    InvoiceDoc,
    LedgerRow,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ValidTable,
)

OPEN_FILENAMES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "policy.json",
        "ledger.csv",
        "invoices.jsonl",
        "payments.jsonl",
        "fx.csv",
    }
)

VALID_TABLES: frozenset[str] = frozenset({"ledger", "invoices", "payments", "fx"})
MAX_PAGE_LIMIT: int = 50


class EpisodeDataStore:
    """Read-only access layer for open episode tables and documents."""

    def __init__(
        self,
        episode_dir: Path | str | None = None,
        episode_data: dict[str, Any] | None = None,
        *,
        manifest: Manifest | dict[str, Any] | None = None,
        policy: PolicyConfig | dict[str, Any] | None = None,
        ledger: list[LedgerRow | dict[str, Any]] | None = None,
        invoices: list[InvoiceDoc | dict[str, Any]] | None = None,
        payments: list[PaymentDoc | dict[str, Any]] | None = None,
        fx: list[FxRateRow | dict[str, Any]] | None = None,
    ) -> None:
        if episode_dir is not None:
            resolved_dir = self._resolve_public_dir(Path(episode_dir))
            (
                self._manifest,
                self._policy,
                self._ledger,
                self._invoices,
                self._payments,
                self._fx,
            ) = self._load_from_directory(resolved_dir)
        elif episode_data is not None:
            (
                self._manifest,
                self._policy,
                self._ledger,
                self._invoices,
                self._payments,
                self._fx,
            ) = self._load_from_models_or_dicts(
                manifest=episode_data.get("manifest"),
                policy=episode_data.get("policy"),
                ledger=episode_data.get("ledger", []),
                invoices=episode_data.get("invoices", []),
                payments=episode_data.get("payments", []),
                fx=episode_data.get("fx", episode_data.get("fx_rates", [])),
            )
        elif manifest is not None and policy is not None:
            (
                self._manifest,
                self._policy,
                self._ledger,
                self._invoices,
                self._payments,
                self._fx,
            ) = self._load_from_models_or_dicts(
                manifest=manifest,
                policy=policy,
                ledger=ledger or [],
                invoices=invoices or [],
                payments=payments or [],
                fx=fx or [],
            )
        else:
            raise ValueError(
                "EpisodeDataStore requires either `episode_dir`, `episode_data`, "
                "or explicit `manifest` and `policy` arguments."
            )

        # Pre-serialize clean dictionaries and build O(1) lookup indices.
        self._manifest_dict: dict[str, Any] = self._manifest.model_dump(mode="json")
        self._policy_dict: dict[str, Any] = self._policy.model_dump(mode="json")

        self._table_rows: dict[str, list[dict[str, Any]]] = {
            "ledger": [row.model_dump(mode="json") for row in self._ledger],
            "invoices": [doc.model_dump(mode="json") for doc in self._invoices],
            "payments": [doc.model_dump(mode="json") for doc in self._payments],
            "fx": [row.model_dump(mode="json") for row in self._fx],
        }

        self._record_index: dict[str, dict[str, dict[str, Any]]] = {
            "ledger": {row["row_id"]: row for row in self._table_rows["ledger"]},
            "invoices": {doc["invoice_id"]: doc for doc in self._table_rows["invoices"]},
            "payments": {doc["payment_id"]: doc for doc in self._table_rows["payments"]},
            "fx": {row["fx_id"]: row for row in self._table_rows["fx"] if row.get("fx_id")},
        }

        self._document_index: dict[str, dict[str, Any]] = {}
        for doc in self._table_rows["invoices"]:
            self._document_index[doc["doc_id"]] = doc
        for doc in self._table_rows["payments"]:
            self._document_index[doc["doc_id"]] = doc

        self._known_ids: frozenset[str] = self._build_known_ids()

    @staticmethod
    def _resolve_public_dir(base_dir: Path) -> Path:
        """Resolve episode directory (supports flat layout or `public/` subfolder)."""
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Episode directory does not exist: {base_dir.name}")
        public_sub = base_dir / "public"
        if (public_sub / "manifest.json").is_file():
            return public_sub
        return base_dir

    @classmethod
    def _load_from_directory(
        cls, directory: Path
    ) -> tuple[
        Manifest,
        PolicyConfig,
        list[LedgerRow],
        list[InvoiceDoc],
        list[PaymentDoc],
        list[FxRateRow],
    ]:
        """Strictly load ONLY the 6 open files; never access truth.json or provenance.json."""
        for fname in OPEN_FILENAMES:
            fpath = directory / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing required open episode file: {fname}")

        manifest_raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = Manifest.model_validate(manifest_raw)

        policy_raw = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
        policy = PolicyConfig.model_validate(policy_raw)

        ledger_rows: list[LedgerRow] = []
        with (directory / "ledger.csv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                cleaned_row: dict[str, Any] = {
                    "row_id": raw_row["row_id"].strip(),
                    "timestamp": raw_row["timestamp"].strip(),
                    "account": raw_row["account"].strip(),
                    "debit_minor": int(raw_row["debit_minor"]),
                    "credit_minor": int(raw_row["credit_minor"]),
                    "currency": raw_row["currency"].strip(),
                    "entity_id": raw_row["entity_id"].strip(),
                    "invoice_id": raw_row.get("invoice_id", "").strip() or None,
                    "payment_id": raw_row.get("payment_id", "").strip() or None,
                }
                ledger_rows.append(LedgerRow.model_validate(cleaned_row))

        invoices: list[InvoiceDoc] = []
        with (directory / "invoices.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    invoices.append(InvoiceDoc.model_validate(json.loads(stripped)))

        payments: list[PaymentDoc] = []
        with (directory / "payments.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    payments.append(PaymentDoc.model_validate(json.loads(stripped)))

        fx_rows: list[FxRateRow] = []
        with (directory / "fx.csv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                fx_dict: dict[str, Any] = {
                    "date": raw_row["date"].strip(),
                    "source_currency": raw_row["source_currency"].strip(),
                    "target_currency": raw_row["target_currency"].strip(),
                    "rate": raw_row["rate"].strip(),
                }
                if raw_row.get("fx_id") and raw_row["fx_id"].strip():
                    fx_dict["fx_id"] = raw_row["fx_id"].strip()
                fx_rows.append(FxRateRow.model_validate(fx_dict))

        return manifest, policy, ledger_rows, invoices, payments, fx_rows

    @staticmethod
    def _load_from_models_or_dicts(
        *,
        manifest: Manifest | dict[str, Any] | None,
        policy: PolicyConfig | dict[str, Any] | None,
        ledger: list[LedgerRow | dict[str, Any]],
        invoices: list[InvoiceDoc | dict[str, Any]],
        payments: list[PaymentDoc | dict[str, Any]],
        fx: list[FxRateRow | dict[str, Any]],
    ) -> tuple[
        Manifest,
        PolicyConfig,
        list[LedgerRow],
        list[InvoiceDoc],
        list[PaymentDoc],
        list[FxRateRow],
    ]:
        if manifest is None or policy is None:
            raise ValueError("Both `manifest` and `policy` must be provided.")

        manifest_obj = manifest if isinstance(manifest, Manifest) else Manifest.model_validate(manifest)
        policy_obj = policy if isinstance(policy, PolicyConfig) else PolicyConfig.model_validate(policy)
        ledger_objs = [r if isinstance(r, LedgerRow) else LedgerRow.model_validate(r) for r in ledger]
        invoice_objs = [d if isinstance(d, InvoiceDoc) else InvoiceDoc.model_validate(d) for d in invoices]
        payment_objs = [d if isinstance(d, PaymentDoc) else PaymentDoc.model_validate(d) for d in payments]
        fx_objs = [r if isinstance(r, FxRateRow) else FxRateRow.model_validate(r) for r in fx]
        return manifest_obj, policy_obj, ledger_objs, invoice_objs, payment_objs, fx_objs

    def _build_known_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for table_map in self._record_index.values():
            ids.update(table_map.keys())
        ids.update(self._document_index.keys())
        for row in self._table_rows["ledger"]:
            if row.get("entity_id"):
                ids.add(row["entity_id"])
        for doc in self._table_rows["invoices"]:
            if doc.get("entity_id"):
                ids.add(doc["entity_id"])
        for doc in self._table_rows["payments"]:
            if doc.get("entity_id"):
                ids.add(doc["entity_id"])
        return frozenset(ids)

    def get_manifest(self) -> dict[str, Any]:
        """Return a deep copy of the episode manifest."""
        return copy.deepcopy(self._manifest_dict)

    def get_policy(self) -> dict[str, Any]:
        """Return a deep copy of the episode accounting policy."""
        return copy.deepcopy(self._policy_dict)

    def list_records(
        self,
        table: ValidTable | str,
        cursor: str | None = None,
        limit: int = MAX_PAGE_LIMIT,
    ) -> dict[str, Any]:
        """Return a paginated slice of records and the next stable cursor."""
        if table not in VALID_TABLES:
            raise ValueError(f"Unknown table '{table}'. Allowed tables: {sorted(VALID_TABLES)}")

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > MAX_PAGE_LIMIT:
            raise ValueError(f"Invalid limit '{limit}'. Must be an integer between 1 and {MAX_PAGE_LIMIT}.")

        if cursor is None or cursor == "":
            offset = 0
        else:
            if not isinstance(cursor, str) or not cursor.isdigit():
                raise ValueError(f"Invalid pagination cursor '{cursor}'. Must be a non-negative integer string.")
            offset = int(cursor)

        rows = self._table_rows[table]
        total_count = len(rows)

        if offset > total_count and total_count > 0:
            raise ValueError(f"Cursor offset {offset} exceeds table size {total_count} for '{table}'.")

        page_slice = rows[offset : offset + limit]
        next_offset = offset + len(page_slice)
        next_cursor: str | None = str(next_offset) if next_offset < total_count else None

        return {
            "table": table,
            "items": copy.deepcopy(page_slice),
            "next_cursor": next_cursor,
            "total_count": total_count,
        }

    def get_record(self, table: ValidTable | str, record_id: str) -> dict[str, Any] | None:
        """Return a single record copy by canonical ID from the specified open table, or None."""
        if table not in VALID_TABLES:
            raise ValueError(f"Unknown table '{table}'. Allowed tables: {sorted(VALID_TABLES)}")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Record ID must be a non-empty string.")

        record = self._record_index[table].get(record_id)
        if record is None:
            return None
        return copy.deepcopy(record)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Return an invoice or payment document copy by `doc_id`, or None."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        doc = self._document_index.get(doc_id)
        if doc is None:
            return None
        return copy.deepcopy(doc)

    def all_known_ids(self) -> set[str]:
        """Return a set of all canonical open IDs in the episode."""
        return set(self._known_ids)


# Alias for backward compatibility
EpisodeDatastore = EpisodeDataStore

__all__ = [
    "OPEN_FILENAMES",
    "VALID_TABLES",
    "MAX_PAGE_LIMIT",
    "EpisodeDataStore",
    "EpisodeDatastore",
]

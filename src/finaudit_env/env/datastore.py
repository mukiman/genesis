"""Read-only episode datastore with stable cursor pagination (`datastore.py`).

Loads open episode files (`manifest.json`, `policy.json`, `ledger.csv`,
`invoices.jsonl`, `payments.jsonl`, `fx.csv`) and serves `list_records`,
`get_record`, and `get_document` queries with a 50-row page limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EpisodeDatastore:
    """Read-only access layer for open episode tables and documents."""

    def __init__(self, episode_dir: Path | str | None = None, episode_data: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("EpisodeDatastore.__init__ is not implemented yet.")

    def list_records(self, table: str, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Return a paginated slice of records and the next stable cursor."""
        raise NotImplementedError("EpisodeDatastore.list_records is not implemented yet.")

    def get_record(self, table: str, record_id: str) -> dict[str, Any]:
        """Return a single record by canonical ID from the specified open table."""
        raise NotImplementedError("EpisodeDatastore.get_record is not implemented yet.")

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """Return an invoice or payment document by `doc_id`."""
        raise NotImplementedError("EpisodeDatastore.get_document is not implemented yet.")

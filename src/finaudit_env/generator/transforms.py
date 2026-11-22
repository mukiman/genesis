"""Isomorphic episode transformations (`transforms.py`).

Applies benign record reordering and canonical ID remapping while preserving
underlying financial semantics and updating `truth.json` / `provenance.json`
for paired generalization checks.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from finaudit_env.schemas import (
    FindingsSubmission,
    InvoiceDoc,
    LedgerRow,
    PaymentDoc,
    ProvenanceEvent,
)


def _build_deranged_bijection(
    old_ids: list[str],
    prefix: str,
    width: int,
    rng: random.Random,
    offset: int = 100,
) -> dict[str, str]:
    """Create a deterministic 1-to-1 mapping from `old_ids` to freshly numbered canonical IDs."""
    unique_old = sorted(set(old_ids))
    n = len(unique_old)
    pool_indices = list(range(offset + 1, offset + n + 1))
    rng.shuffle(pool_indices)
    return {old_id: f"{prefix}{idx:0{width}d}" for old_id, idx in zip(unique_old, pool_indices)}


def permute_and_remap_episode(
    episode_data: dict[str, Any],
    seed: int,
    *,
    permute_order: bool = True,
    remap_ids: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Permute record order and/or remap IDs, returning `(transformed_episode, id_map)`.

    Guarantees:
    - Bijective 1-to-1 remapping of all canonical IDs (`L-XXXX`, `DOC-INV-XXX`,
      `INV-YYYY-XXX`, `DOC-PAY-XXX`, `PAY-YYYY-XXX`, `ENT-XXX`) when `remap_ids=True`.
    - Deterministic row order shuffling across `ledger`, `invoices`, `payments`, and `fx`
      when `permute_order=True` (without shuffling `line_items`).
    - Synchronous remapping of `truth.json` (`affected_ids` and `valid_evidence_sets`)
      and `provenance.json` so that Oracle and structural policies remain invariant.
    """
    ep = copy.deepcopy(episode_data)
    rng = random.Random(seed)

    ledger: list[LedgerRow] = ep["ledger"]
    invoices: list[InvoiceDoc] = ep["invoices"]
    payments: list[PaymentDoc] = ep["payments"]
    fx_rows = ep["fx"]
    manifest = ep["manifest"]
    year_str = manifest.period_start[:4]

    id_map: dict[str, str] = {}

    if remap_ids:
        # 1. Ledger row IDs (L-XXXX)
        id_map.update(
            _build_deranged_bijection(
                old_ids=[r.row_id for r in ledger],
                prefix="L-",
                width=4,
                rng=rng,
                offset=500,
            )
        )

        # 2. Invoice document IDs (DOC-INV-XXX) and business IDs (INV-YYYY-XXX)
        id_map.update(
            _build_deranged_bijection(
                old_ids=[i.doc_id for i in invoices],
                prefix="DOC-INV-",
                width=3,
                rng=rng,
                offset=200,
            )
        )
        id_map.update(
            _build_deranged_bijection(
                old_ids=[i.invoice_id for i in invoices],
                prefix=f"INV-{year_str}-",
                width=3,
                rng=rng,
                offset=200,
            )
        )

        # 3. Payment document IDs (DOC-PAY-XXX) and business IDs (PAY-YYYY-XXX)
        id_map.update(
            _build_deranged_bijection(
                old_ids=[p.doc_id for p in payments],
                prefix="DOC-PAY-",
                width=3,
                rng=rng,
                offset=300,
            )
        )
        id_map.update(
            _build_deranged_bijection(
                old_ids=[p.payment_id for p in payments],
                prefix=f"PAY-{year_str}-",
                width=3,
                rng=rng,
                offset=300,
            )
        )

        # 4. Counterparty entity IDs (ENT-XXX)
        all_entities: list[str] = []
        all_entities.extend(r.entity_id for r in ledger if r.entity_id)
        all_entities.extend(i.entity_id for i in invoices if i.entity_id)
        all_entities.extend(p.entity_id for p in payments if p.entity_id)
        id_map.update(
            _build_deranged_bijection(
                old_ids=all_entities,
                prefix="ENT-",
                width=3,
                rng=rng,
                offset=100,
            )
        )
    else:
        for r in ledger:
            id_map[r.row_id] = r.row_id
            if r.entity_id:
                id_map[r.entity_id] = r.entity_id
        for i in invoices:
            id_map[i.doc_id] = i.doc_id
            id_map[i.invoice_id] = i.invoice_id
            if i.entity_id:
                id_map[i.entity_id] = i.entity_id
        for p in payments:
            id_map[p.doc_id] = p.doc_id
            id_map[p.payment_id] = p.payment_id
            if p.entity_id:
                id_map[p.entity_id] = p.entity_id

    # 5. FX composite IDs remain identity-mapped (since they encode date and currency pair)
    for fx in fx_rows:
        if fx.fx_id:
            id_map[fx.fx_id] = fx.fx_id

    if remap_ids:
        # Apply remapping to open records
        for row in ledger:
            row.row_id = id_map[row.row_id]
            row.entity_id = id_map.get(row.entity_id, row.entity_id)
            if row.invoice_id is not None:
                row.invoice_id = id_map.get(row.invoice_id, row.invoice_id)
            if row.payment_id is not None:
                row.payment_id = id_map.get(row.payment_id, row.payment_id)

        for inv in invoices:
            inv.doc_id = id_map[inv.doc_id]
            inv.invoice_id = id_map[inv.invoice_id]
            inv.entity_id = id_map.get(inv.entity_id, inv.entity_id)

        for pay in payments:
            pay.doc_id = id_map[pay.doc_id]
            pay.payment_id = id_map[pay.payment_id]
            if pay.entity_id is not None:
                pay.entity_id = id_map.get(pay.entity_id, pay.entity_id)
            pay.invoice_refs = [id_map.get(ref, ref) for ref in pay.invoice_refs]

    if permute_order:
        # Deterministically permute table row orders
        rng.shuffle(ledger)
        rng.shuffle(invoices)
        rng.shuffle(payments)
        rng.shuffle(fx_rows)

    # Synchronously update ground-truth issues
    for issue in ep["truth"].issues:
        issue.affected_ids = sorted(id_map.get(aid, aid) for aid in issue.affected_ids)
        issue.valid_evidence_sets = [
            sorted(id_map.get(eid, eid) for eid in ev_set) for ev_set in issue.valid_evidence_sets
        ]

    # Update provenance events and record the transformation event
    for evt in ep["provenance"].events:
        evt.affected_ids = [id_map.get(aid, aid) for aid in evt.affected_ids]

    next_evt_num = len(ep["provenance"].events) + 1
    ep["provenance"].events.append(
        ProvenanceEvent(
            event_id=f"EVT-{next_evt_num:03d}",
            stage="transform",
            description=f"Applied isomorphic ID remapping ({len(id_map)} IDs) and row permutation with seed={seed}.",
            after_state={"transform_seed": seed, "remapped_ids_count": len(id_map)},
        )
    )

    return ep, id_map


def remap_submission_ids(submission: FindingsSubmission, id_map: dict[str, str]) -> FindingsSubmission:
    """Remap all `affected_ids` and `evidence_ids` in a `FindingsSubmission` using `id_map`."""
    remapped = copy.deepcopy(submission)
    for finding in remapped.findings:
        finding.affected_ids = sorted(id_map.get(aid, aid) for aid in finding.affected_ids)
        finding.evidence_ids = sorted(id_map.get(eid, eid) for eid in finding.evidence_ids)
    return remapped

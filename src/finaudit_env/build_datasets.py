"""Dataset builder entrypoint (`build_datasets.py`).

Generates reproducible `train` and `dev` episode directories under `data/`
from configuration files in `configs/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from finaudit_env.generator import generate_episode, verify_episode
from finaudit_env.schemas import (
    FxRateRow,
    InvoiceDoc,
    LedgerRow,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    ProvenanceFile,
    TruthFile,
)

LEDGER_CSV_COLUMNS: list[str] = [
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
]

FX_CSV_COLUMNS: list[str] = [
    "date",
    "source_currency",
    "target_currency",
    "rate",
]


def _default_split_configs(split: str, count: int = 10) -> list[dict[str, Any]]:
    """Fallback deterministic split configurations if config JSON files are absent."""
    base_seed = 1100 if split == "train" else 2200
    currencies = ["USD", "KZT", "EUR", "GBP", "CHF"]
    rules = ["transaction_date", "invoice_issue_date", "previous_business_day"]
    configs: list[dict[str, Any]] = []

    for i in range(1, count + 1):
        ep_id = f"ep_{split}_{i:03d}"
        seed = base_seed + i
        base_curr = currencies[(i - 1) % len(currencies)]
        foreign = [c for c in ["USD", "EUR", "GBP", "KZT", "JPY"] if c != base_curr][:2]
        if i <= 2:
            err_cfg: dict[str, Any] = {"clean_only": True}
        elif i >= 8:
            scenarios = ["partial_fx_cutoff"] if i == 8 else (["duplicate_fx_cutoff"] if i == 9 else ["partial_fx_cutoff", "duplicate_fx_cutoff"])
            err_cfg = {
                "duplicate_invoice": 1,
                "unallocated_or_partial_payment": 1,
                "fx_conversion_error": 1,
                "period_cutoff_error": 1,
                "interacting_scenarios": scenarios,
            }
        else:
            err_cfg = {
                "duplicate_invoice": 1,
                "unallocated_or_partial_payment": 1,
                "fx_conversion_error": 1,
                "period_cutoff_error": 1,
                "interacting_scenarios": [],
            }
        configs.append(
            {
                "episode_id": ep_id,
                "seed": seed,
                "base_currency": base_curr,
                "foreign_currencies": foreign,
                "period_start": f"2026-{min(12, i):02d}-01",
                "period_end": f"2026-{min(12, i):02d}-28",
                "fx_rate_date_rule": rules[(i - 1) % len(rules)],
                "rounding_mode": "HALF_EVEN",
                "error_config": err_cfg,
            }
        )
    return configs


def save_episode_to_dir(episode_data: dict[str, Any], episode_dir: Path | str) -> Path:
    """Serialize all open and closed episode files into `episode_dir`."""
    ep_path = Path(episode_dir)
    ep_path.mkdir(parents=True, exist_ok=True)

    manifest: Manifest = episode_data["manifest"]
    policy: PolicyConfig = episode_data["policy"]
    ledger: list[LedgerRow] = episode_data["ledger"]
    invoices: list[InvoiceDoc] = episode_data["invoices"]
    payments: list[PaymentDoc] = episode_data["payments"]
    fx_rows: list[FxRateRow] = episode_data["fx"]
    truth: TruthFile = episode_data["truth"]
    provenance: ProvenanceFile = episode_data["provenance"]

    # 1. manifest.json
    (ep_path / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 2. policy.json
    (ep_path / "policy.json").write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 3. ledger.csv (explicit nulls serialized as empty strings in CSV)
    with (ep_path / "ledger.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ledger:
            d = row.model_dump(mode="json")
            writer.writerow(
                {
                    "row_id": d["row_id"],
                    "timestamp": d["timestamp"],
                    "account": d["account"],
                    "debit_minor": d["debit_minor"],
                    "credit_minor": d["credit_minor"],
                    "currency": d["currency"],
                    "entity_id": d["entity_id"],
                    "invoice_id": "" if d["invoice_id"] is None else d["invoice_id"],
                    "payment_id": "" if d["payment_id"] is None else d["payment_id"],
                }
            )

    # 4. invoices.jsonl
    with (ep_path / "invoices.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for inv in invoices:
            f.write(json.dumps(inv.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # 5. payments.jsonl
    with (ep_path / "payments.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for pay in payments:
            f.write(json.dumps(pay.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # 6. fx.csv
    with (ep_path / "fx.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FX_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for fx in fx_rows:
            d = fx.model_dump(mode="json")
            writer.writerow(
                {
                    "date": d["date"],
                    "source_currency": d["source_currency"],
                    "target_currency": d["target_currency"],
                    "rate": d["rate"],
                }
            )

    # 7. truth.json (CLOSED evaluator file)
    (ep_path / "truth.json").write_text(
        json.dumps(truth.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 8. provenance.json (CLOSED evaluator file)
    (ep_path / "provenance.json").write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return ep_path


def load_episode_from_dir(episode_dir: Path | str) -> dict[str, Any]:
    """Load an episode directory back into typed Pydantic models for verification."""
    ep_path = Path(episode_dir)

    manifest = Manifest.model_validate_json((ep_path / "manifest.json").read_text(encoding="utf-8"))
    policy = PolicyConfig.model_validate_json((ep_path / "policy.json").read_text(encoding="utf-8"))

    ledger: list[LedgerRow] = []
    with (ep_path / "ledger.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            ledger.append(
                LedgerRow(
                    row_id=raw["row_id"],
                    timestamp=raw["timestamp"],
                    account=raw["account"],
                    debit_minor=int(raw["debit_minor"]),
                    credit_minor=int(raw["credit_minor"]),
                    currency=raw["currency"],
                    entity_id=raw["entity_id"],
                    invoice_id=raw["invoice_id"] or None,
                    payment_id=raw["payment_id"] or None,
                )
            )

    invoices: list[InvoiceDoc] = []
    for line in (ep_path / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            invoices.append(InvoiceDoc.model_validate_json(line))

    payments: list[PaymentDoc] = []
    for line in (ep_path / "payments.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            payments.append(PaymentDoc.model_validate_json(line))

    fx_rows: list[FxRateRow] = []
    with (ep_path / "fx.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            fx_rows.append(
                FxRateRow(
                    date=raw["date"],
                    source_currency=raw["source_currency"],
                    target_currency=raw["target_currency"],
                    rate=raw["rate"],
                )
            )

    truth = TruthFile.model_validate_json((ep_path / "truth.json").read_text(encoding="utf-8"))
    provenance = ProvenanceFile.model_validate_json((ep_path / "provenance.json").read_text(encoding="utf-8"))

    return {
        "episode_id": manifest.episode_id,
        "seed": provenance.seed,
        "config": provenance.generator_config,
        "manifest": manifest,
        "policy": policy,
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx": fx_rows,
        "truth": truth,
        "provenance": provenance,
    }


def build_datasets(config_dir: Path | str = "configs", output_dir: Path | str = "data") -> dict[str, Any]:
    """Generate open and closed episode files for train and dev splits."""
    cfg_root = Path(config_dir)
    out_root = Path(output_dir)

    summary: dict[str, Any] = {
        "train_count": 0,
        "dev_count": 0,
        "verified_count": 0,
        "episodes": {"train": [], "dev": []},
    }

    for split in ("train", "dev"):
        cfg_file = cfg_root / f"{split}_episodes.json"
        if cfg_file.exists():
            split_configs = json.loads(cfg_file.read_text(encoding="utf-8"))
        else:
            split_configs = _default_split_configs(split, count=10)

        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for ep_cfg in split_configs:
            seed = int(ep_cfg["seed"])
            ep_data = generate_episode(seed=seed, config=ep_cfg)
            ver = verify_episode(ep_data)
            if not ver["valid"]:
                raise RuntimeError(f"Generated episode {ep_data['episode_id']} failed verification: {ver['errors']}")

            ep_out_dir = split_dir / ep_data["episode_id"]
            save_episode_to_dir(ep_data, ep_out_dir)

            summary[f"{split}_count"] += 1
            summary["verified_count"] += 1
            summary["episodes"][split].append(
                {
                    "episode_id": ep_data["episode_id"],
                    "seed": seed,
                    "base_currency": ep_data["manifest"].base_currency,
                    "ledger_rows": ep_data["manifest"].ledger_rows_count,
                    "invoices": ep_data["manifest"].invoices_count,
                    "payments": ep_data["manifest"].payments_count,
                    "issues_count": len(ep_data["truth"].issues),
                    "path": str(ep_out_dir),
                }
            )

    return summary


def main() -> None:
    """CLI entrypoint for dataset generation."""
    result = build_datasets()
    print(
        f"Successfully generated and verified {result['verified_count']} episodes "
        f"(train={result['train_count']}, dev={result['dev_count']})."
    )


if __name__ == "__main__":
    main()

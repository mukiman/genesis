"""Interactive / scripted demonstration entrypoint (`demo.py`).

Demonstrates a step-by-step audit episode walkthrough using the RL environment
interface (`reset` / `step`) across all 4 baseline policies (`simple_reconciler`,
`random_explorer`, `flag_all_attack`, `oracle`) and prints observations, actions,
and final evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finaudit_env.env.core import FinAuditEnv
from finaudit_env.policies.flag_all_attack import FlagAllAttackPolicy
from finaudit_env.policies.oracle import OraclePolicy
from finaudit_env.policies.random_explorer import RandomExplorerPolicy
from finaudit_env.policies.simple_reconciler import SimpleReconcilerPolicy
from finaudit_env.schemas import (
    CURRENCY_EXPONENTS,
    FxRateRow,
    InvoiceDoc,
    IssueType,
    LedgerRow,
    LineItem,
    Manifest,
    PaymentDoc,
    PolicyConfig,
    TruthFile,
    TruthIssue,
)


def build_demo_episode_data(seed: int = 42) -> dict[str, Any]:
    """Build a self-contained synthetic episode with 4 financial errors + clean entries for demo runs."""
    # Try dynamic generator if already merged, else fall back to built-in demo episode
    try:
        from finaudit_env.generator import company as gen_company
        from finaudit_env.generator import errors as gen_errors

        if hasattr(gen_company, "generate_clean_company") and hasattr(gen_errors, "inject_errors"):
            clean_ep = gen_company.generate_clean_company(
                seed=seed,
                config={"episode_id": f"ep_demo_{seed:03d}", "num_standard_invoices": 12},
            )
            return gen_errors.inject_errors(
                clean_ep,
                seed=seed,
                error_config={
                    "duplicate_invoice": 1,
                    "unallocated_or_partial_payment": 1,
                    "fx_conversion_error": 1,
                    "period_cutoff_error": 1,
                },
            )
    except Exception:
        pass

    ep_id = f"ep_demo_{seed:03d}"
    manifest = Manifest(
        episode_id=ep_id,
        company_name="Genesis FinTech Demo Corp",
        base_currency="USD",
        period_start="2026-03-01",
        period_end="2026-03-31",
        timezone="UTC",
        rounding_mode="HALF_EVEN",
        currency_exponents=dict(CURRENCY_EXPONENTS),
        ledger_rows_count=8,
        invoices_count=4,
        payments_count=1,
        fx_rates_count=1,
    )
    policy = PolicyConfig(
        fx_rate_date_rule="transaction_date",
        period_start="2026-03-01",
        period_end="2026-03-31",
        timezone="UTC",
        rounding_mode="HALF_EVEN",
        amount_tolerance_minor=1,
    )
    fx = [
        FxRateRow(
            date="2026-03-10",
            source_currency="EUR",
            target_currency="USD",
            rate="1.1000",
        )
    ]
    item_std = [
        LineItem(
            description="Enterprise Audit Cloud Node",
            quantity=1,
            unit_price_minor=10000,
            amount_minor=10000,
        )
    ]
    invoices = [
        InvoiceDoc(
            doc_id="DOC-INV-001",
            invoice_id="INV-2026-001",
            entity_id="ENT-001",
            issued_at="2026-03-05T10:00:00+00:00",
            due_at="2026-03-20T17:00:00+00:00",
            currency="USD",
            total_minor=10000,
            line_items=item_std,
        ),
        # Duplicate of INV-2026-001
        InvoiceDoc(
            doc_id="DOC-INV-002",
            invoice_id="INV-2026-002",
            entity_id="ENT-001",
            issued_at="2026-03-06T16:45:00+00:00",
            due_at="2026-03-20T17:00:00+00:00",
            currency="USD",
            total_minor=10000,
            line_items=item_std,
        ),
        # Foreign currency invoice (10000 EUR @ 1.1000 = 11000 USD, but posted as 12650 USD -> FX error of 1650)
        InvoiceDoc(
            doc_id="DOC-INV-003",
            invoice_id="INV-2026-003",
            entity_id="ENT-002",
            issued_at="2026-03-10T11:00:00+00:00",
            due_at="2026-03-25T17:00:00+00:00",
            currency="EUR",
            total_minor=10000,
            line_items=[
                LineItem(
                    description="FX Advisory Service",
                    quantity=1,
                    unit_price_minor=10000,
                    amount_minor=10000,
                )
            ],
        ),
        # Out-of-period cutoff invoice (issued 2026-04-03 outside [2026-03-01, 2026-03-31])
        InvoiceDoc(
            doc_id="DOC-INV-004",
            invoice_id="INV-2026-004",
            entity_id="ENT-003",
            issued_at="2026-04-03T11:20:00+00:00",
            due_at="2026-04-18T17:00:00+00:00",
            currency="USD",
            total_minor=25000,
            line_items=[
                LineItem(
                    description="Next Period License",
                    quantity=1,
                    unit_price_minor=25000,
                    amount_minor=25000,
                )
            ],
        ),
    ]
    payments = [
        # Payment for INV-2026-001 of 10000 USD, but only 6000 USD allocated in ledger -> diff 4000
        PaymentDoc(
            doc_id="DOC-PAY-001",
            payment_id="PAY-2026-001",
            entity_id="ENT-001",
            timestamp="2026-03-07T14:00:00+00:00",
            currency="USD",
            amount_minor=10000,
            invoice_refs=["INV-2026-001"],
        )
    ]
    ledger = [
        LedgerRow(
            row_id="L-0001",
            timestamp="2026-03-05T10:00:00+00:00",
            account="1100_Accounts_Receivable",
            debit_minor=10000,
            credit_minor=0,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-001",
            payment_id=None,
        ),
        LedgerRow(
            row_id="L-0002",
            timestamp="2026-03-05T10:00:00+00:00",
            account="4000_Sales_Revenue",
            debit_minor=0,
            credit_minor=10000,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-001",
            payment_id=None,
        ),
        # Partial payment settlement for PAY-2026-001 (6000 instead of 10000)
        LedgerRow(
            row_id="L-0003",
            timestamp="2026-03-07T14:00:00+00:00",
            account="1000_Cash_and_Bank",
            debit_minor=6000,
            credit_minor=0,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-001",
            payment_id="PAY-2026-001",
        ),
        LedgerRow(
            row_id="L-0004",
            timestamp="2026-03-07T14:00:00+00:00",
            account="1100_Accounts_Receivable",
            debit_minor=0,
            credit_minor=6000,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-001",
            payment_id="PAY-2026-001",
        ),
        # Duplicate invoice accrual L-0005 / L-0006
        LedgerRow(
            row_id="L-0005",
            timestamp="2026-03-06T16:45:00+00:00",
            account="1100_Accounts_Receivable",
            debit_minor=10000,
            credit_minor=0,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-002",
            payment_id=None,
        ),
        LedgerRow(
            row_id="L-0006",
            timestamp="2026-03-06T16:45:00+00:00",
            account="4000_Sales_Revenue",
            debit_minor=0,
            credit_minor=10000,
            currency="USD",
            entity_id="ENT-001",
            invoice_id="INV-2026-002",
            payment_id=None,
        ),
        # FX conversion error accrual L-0007 (12650 instead of 11000)
        LedgerRow(
            row_id="L-0007",
            timestamp="2026-03-10T11:00:00+00:00",
            account="1100_Accounts_Receivable",
            debit_minor=12650,
            credit_minor=0,
            currency="USD",
            entity_id="ENT-002",
            invoice_id="INV-2026-003",
            payment_id=None,
        ),
        # Period cutoff error accrual L-0008 (issued 2026-04-03)
        LedgerRow(
            row_id="L-0008",
            timestamp="2026-04-03T11:20:00+00:00",
            account="1100_Accounts_Receivable",
            debit_minor=25000,
            credit_minor=0,
            currency="USD",
            entity_id="ENT-003",
            invoice_id="INV-2026-004",
            payment_id=None,
        ),
    ]
    truth = TruthFile(
        episode_id=ep_id,
        issues=[
            TruthIssue(
                issue_id="ISS-001",
                type=IssueType.DUPLICATE_INVOICE,
                affected_ids=["INV-2026-001", "INV-2026-002", "L-0005"],
                expected_amount_minor=10000,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-001", "DOC-INV-002", "L-0005"],
                    ["DOC-INV-001", "DOC-INV-002", "L-0001", "L-0005"],
                ],
                tolerance_minor=1,
            ),
            TruthIssue(
                issue_id="ISS-002",
                type=IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT,
                affected_ids=["INV-2026-001", "L-0004", "PAY-2026-001"],
                expected_amount_minor=4000,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-001", "DOC-PAY-001", "L-0004"],
                ],
                tolerance_minor=1,
            ),
            TruthIssue(
                issue_id="ISS-003",
                type=IssueType.FX_CONVERSION_ERROR,
                affected_ids=["FX-2026-03-10-EUR-USD", "INV-2026-003", "L-0007"],
                expected_amount_minor=1650,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-003", "FX-2026-03-10-EUR-USD", "L-0007"],
                ],
                tolerance_minor=1,
            ),
            TruthIssue(
                issue_id="ISS-004",
                type=IssueType.PERIOD_CUTOFF_ERROR,
                affected_ids=["INV-2026-004", "L-0008"],
                expected_amount_minor=25000,
                currency="USD",
                valid_evidence_sets=[
                    ["DOC-INV-004", "L-0008"],
                ],
                tolerance_minor=1,
            ),
        ],
    )
    return {
        "manifest": manifest,
        "policy": policy,
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx": fx,
        "truth": truth,
    }


def run_demo(seed: int = 42, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a demonstration episode across the 4 baseline policies and return the summary."""
    cfg = dict(config or {})
    if "episode_dir" not in cfg and "episode_data" not in cfg:
        train_ep = Path("data/train/ep_train_001")
        if (train_ep / "manifest.json").is_file() or (train_ep / "public" / "manifest.json").is_file():
            cfg["episode_dir"] = train_ep
        else:
            cfg["episode_data"] = build_demo_episode_data(seed=seed)

    env = FinAuditEnv(mode="train")
    obs0, _ = env.reset(seed=seed, config=cfg)
    truth_obj = env._evaluator_truth

    policies = [
        SimpleReconcilerPolicy(seed=seed),
        RandomExplorerPolicy(seed=seed),
        FlagAllAttackPolicy(seed=seed),
        OraclePolicy(truth=truth_obj, seed=seed),
    ]

    results: dict[str, Any] = {}
    reconciler_trajectory: list[dict[str, Any]] = []

    for pol in policies:
        summary = pol.run_episode(env, seed=seed, config=cfg)
        results[pol.name] = summary
        if pol.name == "simple_reconciler":
            reconciler_trajectory = env.get_trajectory()

    return {
        "episode_id": obs0["manifest"]["episode_id"],
        "company_name": obs0["manifest"].get("company_name", "Synthetic Corp"),
        "policies": results,
        "reconciler_trajectory": reconciler_trajectory,
    }


def main() -> None:
    """CLI entrypoint for demo run."""
    demo_res = run_demo(seed=42)
    print("=" * 78)
    print(f"FINAUDIT RL ENVIRONMENT DEMO // Episode: {demo_res['episode_id']} ({demo_res['company_name']})")
    print("=" * 78)

    print("\n--- Step-by-Step Walkthrough (SimpleReconcilerPolicy) ---")
    for step_entry in demo_res["reconciler_trajectory"]:
        step_no = step_entry["step"]
        act = step_entry["action"]
        act_name = act.get("action") if isinstance(act, dict) else str(act)
        if act_name == "list_records":
            detail = f"table={act.get('table')}, cursor={act.get('cursor')}, limit={act.get('limit')}"
        elif act_name == "calculator":
            detail = f"expr='{act.get('expression')}'"
        elif act_name == "submit":
            f_cnt = len(act.get("submission", {}).get("findings", []))
            detail = f"submitted {f_cnt} findings"
        else:
            detail = str(act)
        print(
            f"  [Step {step_no:02d}] {act_name:<14} ({detail}) -> "
            f"reward={step_entry['reward']:+.4f}, terminated={step_entry['terminated']}"
        )

    print("\n--- Baseline Policies Comparison ---")
    header = f"{'Policy':<20} | {'TP':>3} | {'FP':>3} | {'FN':>3} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Reward':>8} | {'Steps':>5} | {'Time(s)':>7}"
    print(header)
    print("-" * len(header))
    for pol_name, summary in demo_res["policies"].items():
        ev = summary["evaluation"] or {}
        print(
            f"{pol_name:<20} | "
            f"{ev.get('tp', 0):>3} | "
            f"{ev.get('fp', 0):>3} | "
            f"{ev.get('fn', 0):>3} | "
            f"{ev.get('precision', 0.0):>6.3f} | "
            f"{ev.get('recall', 0.0):>6.3f} | "
            f"{ev.get('f1', 0.0):>6.3f} | "
            f"{ev.get('reward', -1.0):>+8.4f} | "
            f"{summary['steps']:>5} | "
            f"{summary['elapsed_seconds']:>7.4f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()


__all__ = [
    "build_demo_episode_data",
    "run_demo",
    "main",
]

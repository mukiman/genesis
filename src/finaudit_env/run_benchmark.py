"""Benchmark runner entrypoint (`run_benchmark.py`).

Runs the 4 baseline policies (`random_explorer`, `simple_reconciler`,
`flag_all_attack`, `oracle`) across evaluation episodes, records trajectories
to `artifacts/trajectories.jsonl` (verifying zero ground-truth leaks), and
outputs diagnostic metrics (`P`, `R`, `F1`, `terminal_reward`, `steps`, `elapsed_time`)
to `artifacts/eval_report.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finaudit_env.demo import build_demo_episode_data
from finaudit_env.env.core import FinAuditEnv
from finaudit_env.env.leak_checker import verify_trajectory_file
from finaudit_env.policies.flag_all_attack import FlagAllAttackPolicy
from finaudit_env.policies.oracle import OraclePolicy
from finaudit_env.policies.random_explorer import RandomExplorerPolicy
from finaudit_env.policies.simple_reconciler import SimpleReconcilerPolicy


def _discover_episode_configs(data_dir: Path | str) -> list[dict[str, Any]]:
    """Discover episode directories on disk or fall back to synthetic benchmark episodes."""
    base = Path(data_dir)
    discovered: list[Path] = []

    if base.is_dir():
        if (base / "manifest.json").is_file() or (base / "public" / "manifest.json").is_file():
            discovered.append(base)
        else:
            for child in sorted(base.iterdir()):
                if child.is_dir() and (
                    (child / "manifest.json").is_file()
                    or (child / "public" / "manifest.json").is_file()
                ):
                    discovered.append(child)

    # Also check sibling train/dev folders if base had fewer than 2 episodes
    if not discovered and base.name in ("dev", "train") and base.parent.is_dir():
        for split in ("train", "dev"):
            split_dir = base.parent / split
            if split_dir.is_dir():
                for child in sorted(split_dir.iterdir()):
                    if child.is_dir() and (
                        (child / "manifest.json").is_file()
                        or (child / "public" / "manifest.json").is_file()
                    ):
                        discovered.append(child)

    if discovered:
        return [{"episode_dir": ep_path} for ep_path in discovered]

    # Fallback: build 4 deterministic synthetic episodes (including 1 clean episode)
    configs: list[dict[str, Any]] = []
    for idx, seed in enumerate([101, 102, 103, 104], start=1):
        ep_bundle = build_demo_episode_data(seed=seed)
        if idx == 4:
            # Make the 4th episode a clean episode (0 issues)
            ep_bundle = build_demo_episode_data(seed=seed)
            ep_bundle["truth"].issues = []
            # Remove errors from the built-in fallback if needed
            try:
                from finaudit_env.generator import company as gen_company

                if hasattr(gen_company, "generate_clean_company"):
                    ep_bundle = gen_company.generate_clean_company(
                        seed=seed,
                        config={"episode_id": f"ep_clean_{seed:03d}", "num_standard_invoices": 10},
                    )
            except Exception:
                pass
        configs.append({"episode_data": ep_bundle})
    return configs


def run_benchmark(
    data_dir: Path | str = "data/dev",
    artifacts_dir: Path | str = "artifacts",
) -> dict[str, Any]:
    """Execute the 4 baseline policies on episodes, record trajectories, and compute diagnostic metrics."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = out_dir / "trajectories.jsonl"
    eval_report_path = out_dir / "eval_report.json"

    # Reset trajectories file
    if trajectories_path.exists():
        trajectories_path.unlink()

    episode_configs = _discover_episode_configs(data_dir)
    policy_names = ["random_explorer", "simple_reconciler", "flag_all_attack", "oracle"]

    per_policy_episodes: dict[str, list[dict[str, Any]]] = {name: [] for name in policy_names}
    all_forbidden_tokens: set[str] = set()

    for ep_idx, cfg in enumerate(episode_configs, start=1):
        seed = 1000 + ep_idx
        env_probe = FinAuditEnv(mode="train")
        env_probe.reset(seed=seed, config=cfg)
        truth_obj = env_probe._evaluator_truth
        all_forbidden_tokens.update(env_probe._forbidden_tokens)

        policies = [
            RandomExplorerPolicy(seed=seed),
            SimpleReconcilerPolicy(seed=seed),
            FlagAllAttackPolicy(seed=seed),
            OraclePolicy(truth=truth_obj, seed=seed),
        ]

        for pol in policies:
            # Record non-oracle trajectories in train mode without ground truth in observations
            env = FinAuditEnv(mode="train")
            summary = pol.run_episode(env, seed=seed, config=cfg)
            ev = summary["evaluation"] or {}

            if pol.name != "oracle":
                env.dump_trajectory_jsonl(trajectories_path, append=True)

            per_policy_episodes[pol.name].append(
                {
                    "episode_id": summary["episode_id"],
                    "tp": int(ev.get("tp", 0)),
                    "fp": int(ev.get("fp", 0)),
                    "fn": int(ev.get("fn", 0)),
                    "precision": float(ev.get("precision", 0.0)),
                    "recall": float(ev.get("recall", 0.0)),
                    "f1": float(ev.get("f1", 0.0)),
                    "reward": float(ev.get("reward", -1.0)),
                    "steps": int(summary["steps"]),
                    "elapsed_seconds": float(summary["elapsed_seconds"]),
                }
            )

    leak_violations = verify_trajectory_file(
        trajectories_path,
        forbidden_tokens=all_forbidden_tokens,
    )
    if leak_violations:
        raise AssertionError(f"Trajectory leak check failed: {leak_violations}")

    aggregated_summary: dict[str, dict[str, Any]] = {}
    for pol_name, runs in per_policy_episodes.items():
        n = max(1, len(runs))
        aggregated_summary[pol_name] = {
            "episodes_count": len(runs),
            "mean_precision": round(sum(r["precision"] for r in runs) / n, 4),
            "mean_recall": round(sum(r["recall"] for r in runs) / n, 4),
            "mean_f1": round(sum(r["f1"] for r in runs) / n, 4),
            "mean_reward": round(sum(r["reward"] for r in runs) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in runs) / n, 2),
            "mean_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in runs) / n, 6),
            "total_tp": sum(r["tp"] for r in runs),
            "total_fp": sum(r["fp"] for r in runs),
            "total_fn": sum(r["fn"] for r in runs),
            "episodes": runs,
        }

    report = {
        "schema_version": "1.0.0",
        "episodes_evaluated": len(episode_configs),
        "trajectories_file": str(trajectories_path.name),
        "trajectory_leak_violations": len(leak_violations),
        "policies": aggregated_summary,
    }
    eval_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for benchmark execution."""
    report = run_benchmark()
    print("=" * 78)
    print(f"FINAUDIT BENCHMARK REPORT // Episodes evaluated: {report['episodes_evaluated']}")
    print("=" * 78)
    header = f"{'Policy':<20} | {'Mean P':>7} | {'Mean R':>7} | {'Mean F1':>7} | {'Mean Reward':>11} | {'Steps':>6} | {'Time(s)':>8}"
    print(header)
    print("-" * len(header))
    for pol_name, stats in report["policies"].items():
        print(
            f"{pol_name:<20} | "
            f"{stats['mean_precision']:>7.4f} | "
            f"{stats['mean_recall']:>7.4f} | "
            f"{stats['mean_f1']:>7.4f} | "
            f"{stats['mean_reward']:>+11.4f} | "
            f"{stats['mean_steps']:>6.2f} | "
            f"{stats['mean_elapsed_seconds']:>8.4f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()


__all__ = [
    "run_benchmark",
    "main",
]

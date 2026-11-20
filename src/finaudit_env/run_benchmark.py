"""Benchmark runner entrypoint (`run_benchmark.py`).

Runs the 4 baseline policies (`random_explorer`, `simple_reconciler`,
`flag_all_attack`, `oracle`) across evaluation episodes (defaulting to all 20
open `train` + `dev` episodes), performs 20 paired reproducibility & isomorphic
ID-remapping checks (`permute_and_remap_episode`), records leak-free trajectories
to `artifacts/trajectories.jsonl`, and outputs diagnostic metrics (`P`, `R`, `F1`,
`terminal_reward`, `steps`, `elapsed_time`) to `artifacts/eval_report.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finaudit_env.build_datasets import load_episode_from_dir
from finaudit_env.demo import build_demo_episode_data
from finaudit_env.env.core import FinAuditEnv
from finaudit_env.env.leak_checker import verify_trajectory_file
from finaudit_env.generator.transforms import permute_and_remap_episode, remap_submission_ids
from finaudit_env.policies.flag_all_attack import FlagAllAttackPolicy
from finaudit_env.policies.oracle import OraclePolicy
from finaudit_env.policies.random_explorer import RandomExplorerPolicy
from finaudit_env.policies.simple_reconciler import SimpleReconcilerPolicy
from finaudit_env.schemas import FindingsSubmission
from finaudit_env.scoring.reward import evaluate_submission


def _discover_episode_configs(data_dir: Path | str) -> list[dict[str, Any]]:
    """Discover episode directories on disk or fall back to synthetic benchmark episodes."""
    base = Path(data_dir)
    discovered: list[Path] = []

    def _scan_folder(folder: Path) -> None:
        if not folder.is_dir():
            return
        if (folder / "manifest.json").is_file() or (folder / "public" / "manifest.json").is_file():
            if folder not in discovered:
                discovered.append(folder)
            return
        for child in sorted(folder.iterdir()):
            if child.is_dir():
                if (child / "manifest.json").is_file() or (child / "public" / "manifest.json").is_file():
                    if child not in discovered:
                        discovered.append(child)

    if base.is_dir():
        if base.name in ("data", "train", "dev"):
            parent = base if base.name == "data" else base.parent
            if (parent / "train").is_dir() and (parent / "dev").is_dir() and base.name in ("data", "dev"):
                _scan_folder(parent / "train")
                _scan_folder(parent / "dev")
            else:
                _scan_folder(base)
        else:
            _scan_folder(base)

    if discovered:
        return [{"episode_dir": ep_path} for ep_path in discovered]

    # Fallback: build 4 deterministic synthetic episodes (including 1 clean episode)
    configs: list[dict[str, Any]] = []
    for idx, seed in enumerate([101, 102, 103, 104], start=1):
        ep_bundle = build_demo_episode_data(seed=seed)
        if idx == 4:
            ep_bundle["truth"].issues = []
        configs.append({"episode_data": ep_bundle})
    return configs


def _run_paired_generalization_check(
    ep_cfg: dict[str, Any],
    pair_index: int,
) -> dict[str, Any]:
    """Run paired reproducibility and isomorphic ID-permutation check on one episode."""
    seed = 7000 + pair_index
    transform_seed = 9000 + pair_index

    if "episode_dir" in ep_cfg:
        orig_ep_data = load_episode_from_dir(ep_cfg["episode_dir"])
    else:
        orig_ep_data = ep_cfg["episode_data"]

    # 1. Exact seed & action sequence replay check
    env_run1 = FinAuditEnv(mode="train")
    res_rec_1 = SimpleReconcilerPolicy(seed=seed).run_episode(
        env_run1, seed=seed, config={"episode_data": orig_ep_data}
    )
    traj_1 = env_run1.get_trajectory()

    env_run2 = FinAuditEnv(mode="train")
    res_rec_2 = SimpleReconcilerPolicy(seed=seed).run_episode(
        env_run2, seed=seed, config={"episode_data": orig_ep_data}
    )
    traj_2 = env_run2.get_trajectory()

    exact_replay_match = (
        traj_1 == traj_2
        and res_rec_1["evaluation"]["reward"] == res_rec_2["evaluation"]["reward"]
    )

    # 2. Isomorphic row permutation and canonical ID remapping check
    permuted_ep_data, id_map = permute_and_remap_episode(orig_ep_data, seed=transform_seed)

    env_orig_oracle = FinAuditEnv(mode="train")
    oracle_orig = OraclePolicy(truth=orig_ep_data["truth"], seed=seed).run_episode(
        env_orig_oracle, seed=seed, config={"episode_data": orig_ep_data}
    )

    env_perm_oracle = FinAuditEnv(mode="train")
    oracle_perm = OraclePolicy(truth=permuted_ep_data["truth"], seed=seed).run_episode(
        env_perm_oracle, seed=seed, config={"episode_data": permuted_ep_data}
    )

    env_perm_rec = FinAuditEnv(mode="train")
    res_rec_perm = SimpleReconcilerPolicy(seed=seed).run_episode(
        env_perm_rec, seed=seed, config={"episode_data": permuted_ep_data}
    )

    oracle_invariant = (
        oracle_orig["evaluation"]["f1"] == oracle_perm["evaluation"]["f1"] == 1.0
        and abs(oracle_orig["evaluation"]["reward"] - oracle_perm["evaluation"]["reward"]) < 1e-9
    )

    # Also verify that remapping the original Oracle submission through id_map matches permuted truth
    orig_oracle_sub = FindingsSubmission.model_validate(oracle_orig["last_action"]["submission"])
    remapped_oracle_sub = remap_submission_ids(orig_oracle_sub, id_map)
    remapped_eval = evaluate_submission(
        remapped_oracle_sub,
        permuted_ep_data["truth"],
        inspection_steps=oracle_orig["steps"] - 1,
    )
    oracle_remapped_answer_match = (
        remapped_eval["f1"] == 1.0
        and abs(remapped_eval["reward"] - oracle_orig["evaluation"]["reward"]) < 1e-9
    )

    reconciler_invariant = (
        res_rec_1["evaluation"]["f1"] == res_rec_perm["evaluation"]["f1"]
        and abs(res_rec_1["evaluation"]["reward"] - res_rec_perm["evaluation"]["reward"]) < 1e-9
    )

    passed = bool(
        exact_replay_match
        and oracle_invariant
        and oracle_remapped_answer_match
        and reconciler_invariant
    )
    return {
        "pair_index": pair_index,
        "episode_id": str(orig_ep_data["manifest"].episode_id),
        "remapped_ids_count": len(id_map),
        "exact_replay_match": exact_replay_match,
        "oracle_invariant": oracle_invariant,
        "oracle_remapped_answer_match": oracle_remapped_answer_match,
        "reconciler_invariant": reconciler_invariant,
        "oracle_f1": float(oracle_perm["evaluation"]["f1"]),
        "oracle_reward": float(oracle_perm["evaluation"]["reward"]),
        "reconciler_f1": float(res_rec_perm["evaluation"]["f1"]),
        "reconciler_reward": float(res_rec_perm["evaluation"]["reward"]),
        "passed": passed,
    }


def run_benchmark(
    data_dir: Path | str = "data",
    artifacts_dir: Path | str = "artifacts",
) -> dict[str, Any]:
    """Execute the 4 baseline policies on episodes, run paired checks, record trajectories, and save report."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = out_dir / "trajectories.jsonl"
    eval_report_path = out_dir / "eval_report.json"

    if trajectories_path.exists():
        trajectories_path.unlink()

    episode_configs = _discover_episode_configs(data_dir)
    policy_names = ["random_explorer", "simple_reconciler", "flag_all_attack", "oracle"]

    per_policy_episodes: dict[str, list[dict[str, Any]]] = {name: [] for name in policy_names}
    all_forbidden_tokens: set[str] = set()
    paired_checks: list[dict[str, Any]] = []

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

        paired_checks.append(_run_paired_generalization_check(cfg, pair_index=ep_idx))

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

    paired_passed = sum(1 for c in paired_checks if c["passed"])
    report = {
        "schema_version": "1.0.0",
        "episodes_evaluated": len(episode_configs),
        "trajectories_file": str(trajectories_path.name),
        "trajectory_leak_violations": len(leak_violations),
        "paired_checks_total": len(paired_checks),
        "paired_checks_passed": paired_passed,
        "paired_checks": paired_checks,
        "policies": aggregated_summary,
    }
    eval_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for benchmark execution."""
    report = run_benchmark()
    print("=" * 84)
    print(
        f"FINAUDIT BENCHMARK REPORT // Episodes: {report['episodes_evaluated']} | "
        f"Paired Checks: {report['paired_checks_passed']}/{report['paired_checks_total']} PASS | "
        f"Trajectory Leaks: {report['trajectory_leak_violations']}"
    )
    print("=" * 84)
    header = (
        f"{'Policy':<20} | {'Mean P':>7} | {'Mean R':>7} | {'Mean F1':>7} | "
        f"{'Mean Reward':>11} | {'Steps':>6} | {'Time(s)':>8}"
    )
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
    print("=" * 84)


if __name__ == "__main__":
    main()


__all__ = [
    "run_benchmark",
    "main",
]

"""Official Judge Harness Adapter (`judge_adapter.py`).

Implements the unified FinTech verification entrypoints specified in
`https://genesis-exe.com/fintech-checks.html`:
- `generate_episode(seed: int, config: dict, output_dir: str) -> None`
- `make_env()` -> environment object with `reset(seed, config)` and `step(action)`
- `score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict`
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from finaudit_env.build_datasets import save_episode_to_dir
from finaudit_env.env.core import FinAuditEnv
from finaudit_env.env.datastore import EpisodeDataStore
from finaudit_env.env.leak_checker import assert_no_leaks
from finaudit_env.generator import generate_episode as _internal_generate_episode
from finaudit_env.schemas import FindingsSubmission, TruthFile
from finaudit_env.scoring.reward import evaluate_submission

# In-memory evaluator error log for invalid submissions scored via score_episode
EVALUATOR_ERROR_LOG: list[dict[str, Any]] = []

_DIRECT_ERROR_KEYS: frozenset[str] = frozenset(
    {
        "clean_only",
        "clean",
        "duplicate_invoice",
        "unallocated_or_partial_payment",
        "unallocated_payment",
        "partial_payment",
        "payment_error_mode",
        "fx_conversion_error",
        "period_cutoff_error",
        "interacting_scenarios",
    }
)


def _normalize_generator_config(seed: int, config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize judge/team config dictionary into the canonical generator configuration."""
    raw = copy.deepcopy(dict(config)) if isinstance(config, dict) else {}
    if isinstance(raw.get("team_config"), dict):
        team_cfg = copy.deepcopy(raw.pop("team_config"))
        team_cfg.update(raw)
        raw = team_cfg

    if "error_config" not in raw:
        extracted_err: dict[str, Any] = {}
        for k in _DIRECT_ERROR_KEYS:
            if k in raw:
                extracted_err[k] = raw[k]
        if extracted_err:
            raw["error_config"] = extracted_err

    raw.setdefault("seed", int(seed))
    return raw


def generate_episode(seed: int, config: dict[str, Any] | None, output_dir: str | Path) -> None:
    """Generate a complete episode (6 open files + closed truth.json & provenance.json) into `output_dir`."""
    norm_cfg = _normalize_generator_config(seed=int(seed), config=config)
    ep_data = _internal_generate_episode(seed=int(seed), config=norm_cfg)
    save_episode_to_dir(ep_data, Path(output_dir))


def _classify_error_code(error_msg: str, *, is_submit: bool = False) -> str:
    """Map an internal bounded error message to a public error code without leaking truth."""
    if is_submit:
        return "INVALID_SUBMISSION_STRUCTURE"
    lowered = error_msg.lower()
    if "time limit" in lowered or "timeout" in lowered:
        return "EPISODE_TIMEOUT"
    if "already finished" in lowered:
        return "EPISODE_ALREADY_FINISHED"
    if "not found" in lowered:
        return "NOT_FOUND"
    if "calculator" in lowered or "arithmetic" in lowered or "division" in lowered or "syntax" in lowered or "expression" in lowered:
        return "CALCULATOR_ERROR"
    if "cursor" in lowered or "limit" in lowered or "table" in lowered:
        return "INVALID_PAGINATION_OR_TABLE"
    return "INVALID_ACTION"


class JudgeEnvAdapter:
    """Thin adapter around `FinAuditEnv` enforcing the exact `fintech-checks` envelope contract."""

    def __init__(self, env: FinAuditEnv | None = None) -> None:
        self._env: FinAuditEnv = env if env is not None else FinAuditEnv(mode="closed")

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment for a closed (`episode_dir`) or open (`train`) episode."""
        cfg = copy.deepcopy(dict(config)) if isinstance(config, dict) else {}
        if "mode" not in cfg:
            cfg["mode"] = "closed" if "episode_dir" in cfg else "train"

        if "episode_dir" not in cfg and "episode_data" not in cfg:
            cfg = _normalize_generator_config(seed=0 if seed is None else int(seed), config=cfg)

        raw_obs, raw_info = self._env.reset(seed=seed, config=cfg)
        rem = int(raw_obs.get("remaining_steps", raw_obs.get("steps_remaining", 100)))
        observation: dict[str, Any] = {
            "task": raw_obs["task"],
            "manifest": raw_obs["manifest"],
            "policy": raw_obs["policy"],
            "remaining_steps": rem,
            "steps_remaining": rem,
        }
        return observation, raw_info

    def step(
        self,
        action: dict[str, Any] | Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one step using `{"name": "<action>", "args": {...}}` or internal action dict."""
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        rem = int(raw_obs.get("remaining_steps", raw_obs.get("steps_remaining", 0)))

        is_submit = isinstance(action, dict) and (
            action.get("name") == "submit"
            or action.get("action") == "submit"
            or action.get("tool") == "submit"
            or action.get("type") == "submit"
        )
        is_calculator = isinstance(action, dict) and (
            action.get("name") == "calculator"
            or action.get("action") == "calculator"
            or action.get("tool") == "calculator"
            or action.get("type") == "calculator"
        )

        if "error" in raw_obs and raw_obs["error"] is not None:
            err_val = raw_obs["error"]
            if isinstance(err_val, dict):
                err_obj = err_val
            else:
                err_str = str(err_val)
                err_obj = {
                    "code": _classify_error_code(err_str, is_submit=is_submit),
                    "message": err_str,
                }
            obs: dict[str, Any] = {
                "result": None,
                "error": err_obj,
                "remaining_steps": rem,
                "steps_remaining": rem,
            }
        else:
            raw_res = raw_obs.get("result")
            if is_submit and isinstance(raw_res, dict) and raw_res.get("accepted") is True:
                res_payload: Any = {"accepted": True}
            elif is_calculator and isinstance(raw_res, dict) and "result" in raw_res:
                res_payload = str(raw_res["result"])
            else:
                res_payload = raw_res
            obs = {
                "result": res_payload,
                "error": None,
                "remaining_steps": rem,
                "steps_remaining": rem,
            }

        assert_no_leaks(
            {"observation": obs, "info": info},
            forbidden_tokens=self._env._forbidden_tokens,
        )
        out_reward: float = 0.0 if self._env.mode != "train" else float(reward)
        return obs, out_reward, bool(terminated), bool(truncated), info

    def get_last_evaluation(self) -> dict[str, Any] | None:
        return self._env.get_last_evaluation()

    def get_evaluator_log(self) -> list[str]:
        return self._env.get_evaluator_log()

    def get_trajectory(self) -> list[dict[str, Any]]:
        return self._env.get_trajectory()


def make_env() -> JudgeEnvAdapter:
    """Create and return an environment instance implementing `reset(seed, config)` and `step(action)`."""
    return JudgeEnvAdapter()


def _resolve_accessible_ids_for_scoring(ep_path: Path, raw_truth_dict: dict[str, Any]) -> frozenset[str] | None:
    """Load accessible open IDs from `episode_dir` for verifying `evidence_ids` existence (M04 / Test 11)."""
    if isinstance(raw_truth_dict.get("accessible_ids"), list):
        return frozenset(str(x) for x in raw_truth_dict["accessible_ids"])
    try:
        store = EpisodeDataStore(episode_dir=ep_path)
        return store.all_known_ids()
    except Exception:
        return None


def _prepare_submission_for_scoring(
    findings: Any,
    expected_episode_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate outer envelope of `findings` passed to `score_episode`."""
    if not isinstance(findings, dict):
        return None, "Submission payload must be a JSON object (dict)."

    # If the caller passed {"findings": <findings.json dict>} wrapper, unwrap once
    payload = dict(findings)
    if (
        set(payload.keys()) == {"findings"}
        and isinstance(payload.get("findings"), dict)
        and "findings" in payload["findings"]
    ):
        payload = dict(payload["findings"])

    if "findings" not in payload or not isinstance(payload.get("findings"), list):
        return None, "Submission object must contain a 'findings' list."

    if "episode_id" not in payload:
        payload["episode_id"] = expected_episode_id
    elif not isinstance(payload["episode_id"], str) or not payload["episode_id"]:
        return None, "Field 'episode_id' in findings must be a non-empty string."

    return payload, None


def score_episode(
    episode_dir: str | Path,
    findings: dict[str, Any] | Any,
    inspection_steps: int,
) -> dict[str, Any]:
    """Evaluator-only scoring entrypoint (`score_episode` from `fintech-checks`).

    Returns:
    - On valid submission structure:
      `{"tp": int, "fp": int, "fn": int, "precision": float, "recall": float, "f1": float, "return": float}`
    - On invalid submission structure (>30 findings, >10 evidence_ids before dedup,
      >1000 chars explanation, non-int amount, etc.):
      `{"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1}`
      and records the error in `EVALUATOR_ERROR_LOG`.
    """
    ep_path = Path(episode_dir)
    truth_candidates = [ep_path / "truth.json", ep_path / "private" / "truth.json"]
    raw_truth_dict: dict[str, Any] = {}
    truth_obj: TruthFile | None = None

    for candidate in truth_candidates:
        if candidate.is_file():
            raw_truth_dict = json.loads(candidate.read_text(encoding="utf-8"))
            clean_truth_dict = {
                k: v for k, v in raw_truth_dict.items() if k in ("schema_version", "episode_id", "issues")
            }
            truth_obj = TruthFile.model_validate(clean_truth_dict)
            break

    if truth_obj is None:
        truth_obj = FinAuditEnv._load_truth_from_dir(ep_path)

    accessible_ids = _resolve_accessible_ids_for_scoring(ep_path, raw_truth_dict)
    prepared_sub, prep_err = _prepare_submission_for_scoring(findings, truth_obj.episode_id)

    if prep_err is not None or prepared_sub is None:
        EVALUATOR_ERROR_LOG.append(
            {
                "episode_dir": str(ep_path.name),
                "error": prep_err or "Invalid submission payload.",
            }
        )
        return {
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "return": -1,
        }

    eval_res = evaluate_submission(
        prepared_sub,
        truth_obj,
        inspection_steps=int(inspection_steps),
        default_tolerance_minor=0,
        accessible_ids=accessible_ids,
        check_episode_id=False,
    )

    if eval_res.get("invalid_structure"):
        EVALUATOR_ERROR_LOG.append(
            {
                "episode_dir": str(ep_path.name),
                "error": str(eval_res.get("error") or "Invalid findings submission structure."),
            }
        )
        return {
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "return": -1,
        }

    return {
        "tp": int(eval_res["tp"]),
        "fp": int(eval_res["fp"]),
        "fn": int(eval_res["fn"]),
        "precision": float(eval_res["precision"]),
        "recall": float(eval_res["recall"]),
        "f1": float(eval_res["f1"]),
        "return": float(eval_res["reward"]),
    }


__all__ = [
    "EVALUATOR_ERROR_LOG",
    "JudgeEnvAdapter",
    "generate_episode",
    "make_env",
    "score_episode",
]

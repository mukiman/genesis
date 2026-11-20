"""Core RL environment implementation (`core.py`).

Implements the Gymnasium-style JSON-compatible `reset(seed, config)` and
`step(action)` loop with a 100-step budget, 60-second timeout, bounded error
handling, trajectory logging, and strict separation from closed ground-truth data.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from finaudit_env.env.calculator import CalculatorError, safe_calculate
from finaudit_env.env.datastore import EpisodeDataStore
from finaudit_env.env.leak_checker import (
    assert_no_leaks,
    extract_forbidden_tokens_from_truth,
    scan_payload_for_leaks,
)
from finaudit_env.schemas import (
    Action,
    CalculatorAction,
    FindingsSubmission,
    GetDocumentAction,
    GetRecordAction,
    ListRecordsAction,
    SubmitAction,
    TruthFile,
    parse_action,
)
from finaudit_env.scoring.reward import evaluate_submission

TASK_DESCRIPTION: str = (
    "Inspect the open financial records (ledger, invoices, payments, fx) using "
    "list_records, get_record, get_document, and calculator. Identify all financial "
    "audit discrepancies and submit findings with canonical affected_ids, discrepancy "
    "amounts in minor units, currency, and verifiable evidence_ids before the step budget expires."
)


class FinAuditEnv:
    """Reproducible financial audit RL environment (`reset` / `step` contract)."""

    MAX_STEPS: int = 100
    DEFAULT_TIMEOUT_SECONDS: float = 60.0

    def __init__(
        self,
        mode: Literal["train", "eval", "closed"] = "train",
        *,
        max_steps: int = MAX_STEPS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        episode_dir: Path | str | None = None,
        episode_data: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("train", "eval", "closed"):
            raise ValueError(f"Unsupported environment mode '{mode}'. Use 'train' or 'eval'.")
        if max_steps < 1 or max_steps > self.MAX_STEPS:
            raise ValueError(f"max_steps must be between 1 and {self.MAX_STEPS}, got {max_steps}.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self._default_mode: str = "eval" if mode == "closed" else mode
        self._default_max_steps: int = int(max_steps)
        self._default_timeout_seconds: float = float(timeout_seconds)

        self.mode: str = self._default_mode
        self.max_steps: int = self._default_max_steps
        self.timeout_seconds: float = self._default_timeout_seconds
        self._default_episode_dir: Path | None = Path(episode_dir) if episode_dir is not None else None
        self._default_episode_data: dict[str, Any] | None = (
            copy.deepcopy(episode_data) if episode_data is not None else None
        )

        # Per-episode mutable state (isolated on every reset)
        self._datastore: EpisodeDataStore | None = None
        self._evaluator_truth: TruthFile | None = None
        self._forbidden_tokens: set[str] = set()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._inspection_steps: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._start_time: float = 0.0
        self._last_evaluation: dict[str, Any] | None = None
        self._evaluator_log: list[str] = []
        self._trajectory: list[dict[str, Any]] = []
        self._is_reset: bool = False

    @staticmethod
    def _load_truth_from_dir(episode_dir: Path) -> TruthFile:
        """Load closed `truth.json` for evaluator scoring only (never exposed to datastore or agent)."""
        candidates = [
            episode_dir / "truth.json",
            episode_dir / "private" / "truth.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                return TruthFile.model_validate(raw)

        # Fallback: if no truth.json is present, inspect manifest for episode_id and default to empty clean truth
        manifest_candidates = [
            episode_dir / "manifest.json",
            episode_dir / "public" / "manifest.json",
        ]
        ep_id = "ep_unknown"
        for m_path in manifest_candidates:
            if m_path.is_file():
                ep_id = json.loads(m_path.read_text(encoding="utf-8")).get("episode_id", ep_id)
                break
        return TruthFile(episode_id=ep_id, issues=[])

    @staticmethod
    def _generate_episode_dynamically(seed: int | None, config: dict[str, Any]) -> dict[str, Any]:
        """Attempt to invoke the synthetic company & error generator when no pre-built directory is supplied."""
        effective_seed = 0 if seed is None else int(seed)
        try:
            from finaudit_env.build_datasets import generate_episode

            return generate_episode(seed=effective_seed, config=config)
        except Exception:
            pass

        try:
            from finaudit_env.generator import company as gen_company
            from finaudit_env.generator import errors as gen_errors
        except Exception as exc:
            raise RuntimeError("Generator module could not be imported for dynamic episode generation.") from exc

        for builder_name in ("generate_episode", "build_episode"):
            if hasattr(gen_errors, builder_name):
                return getattr(gen_errors, builder_name)(seed=effective_seed, config=config)  # type: ignore[no-any-return]
            if hasattr(gen_company, builder_name):
                return getattr(gen_company, builder_name)(seed=effective_seed, config=config)  # type: ignore[no-any-return]

        if hasattr(gen_company, "generate_clean_company"):
            clean_bundle = gen_company.generate_clean_company(seed=effective_seed, config=config)
            if hasattr(gen_errors, "inject_errors"):
                try:
                    return gen_errors.inject_errors(clean_bundle, seed=effective_seed, error_config=config)  # type: ignore[no-any-return]
                except NotImplementedError:
                    pass
            if isinstance(clean_bundle, dict):
                return clean_bundle

        raise ValueError(
            "No `episode_dir` or `episode_data` provided in reset(config=...), "
            "and dynamic episode generator is not fully configured."
        )

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment state and return initial `(observation, info)`."""
        cfg = dict(config) if config is not None else {}

        self.mode = self._default_mode
        self.max_steps = self._default_max_steps
        self.timeout_seconds = self._default_timeout_seconds

        if "mode" in cfg and cfg["mode"] in ("train", "eval", "closed"):
            self.mode = "eval" if cfg["mode"] == "closed" else cfg["mode"]
        if "max_steps" in cfg:
            new_max = int(cfg["max_steps"])
            if 1 <= new_max <= self.MAX_STEPS:
                self.max_steps = new_max
        if "timeout_seconds" in cfg:
            new_timeout = float(cfg["timeout_seconds"])
            if new_timeout > 0:
                self.timeout_seconds = new_timeout

        ep_dir = cfg.get("episode_dir") or self._default_episode_dir
        ep_data = cfg.get("episode_data") or self._default_episode_data

        if ep_dir is not None:
            resolved_dir = Path(ep_dir)
            self._datastore = EpisodeDataStore(episode_dir=resolved_dir)
            if "truth" in cfg and cfg["truth"] is not None:
                self._evaluator_truth = (
                    cfg["truth"]
                    if isinstance(cfg["truth"], TruthFile)
                    else TruthFile.model_validate(cfg["truth"])
                )
            else:
                self._evaluator_truth = self._load_truth_from_dir(resolved_dir)
        elif ep_data is not None:
            self._datastore = EpisodeDataStore(episode_data=ep_data)
            raw_truth = cfg.get("truth") or ep_data.get("truth")
            if raw_truth is not None:
                self._evaluator_truth = (
                    raw_truth
                    if isinstance(raw_truth, TruthFile)
                    else TruthFile.model_validate(raw_truth)
                )
            else:
                ep_id = self._datastore.get_manifest()["episode_id"]
                self._evaluator_truth = TruthFile(episode_id=ep_id, issues=[])
        else:
            generated = self._generate_episode_dynamically(seed=seed, config=cfg)
            self._datastore = EpisodeDataStore(episode_data=generated)
            raw_truth = generated.get("truth")
            if raw_truth is not None:
                self._evaluator_truth = (
                    raw_truth
                    if isinstance(raw_truth, TruthFile)
                    else TruthFile.model_validate(raw_truth)
                )
            else:
                ep_id = self._datastore.get_manifest()["episode_id"]
                self._evaluator_truth = TruthFile(episode_id=ep_id, issues=[])

        manifest_dict = self._datastore.get_manifest()
        policy_dict = self._datastore.get_policy()
        self._episode_id = str(manifest_dict["episode_id"])
        self._forbidden_tokens = extract_forbidden_tokens_from_truth(
            self._evaluator_truth, seed=seed
        )

        # Isolate all counters and logs for the new episode
        self._step_count = 0
        self._inspection_steps = 0
        self._terminated = False
        self._truncated = False
        self._last_evaluation = None
        self._evaluator_log = []
        self._trajectory = []
        self._start_time = time.monotonic()
        self._is_reset = True

        observation: dict[str, Any] = {
            "task": TASK_DESCRIPTION,
            "manifest": manifest_dict,
            "policy": policy_dict,
            "steps_remaining": self.max_steps,
        }
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "step": 0,
            "steps_remaining": self.max_steps,
        }

        assert_no_leaks(
            {"observation": observation, "info": info},
            forbidden_tokens=self._forbidden_tokens,
        )
        return observation, info

    def _safe_id_for_error(self, raw_id: str) -> str:
        """Sanitize a queried ID before embedding it in an error string so adversarial IDs cannot trigger leak failures."""
        if not isinstance(raw_id, str) or len(raw_id) > 64:
            return "<redacted_id>"
        if scan_payload_for_leaks(raw_id, forbidden_tokens=self._forbidden_tokens):
            return "<redacted_id>"
        return raw_id

    def _build_step_info(self) -> dict[str, Any]:
        """Construct the public `info` dictionary without leaking closed truth or eval scores."""
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "inspection_steps": self._inspection_steps,
            "steps_remaining": max(0, self.max_steps - self._step_count),
        }
        if self.mode == "train" and (self._terminated or self._truncated) and self._last_evaluation is not None:
            info["evaluation"] = {
                "tp": self._last_evaluation["tp"],
                "fp": self._last_evaluation["fp"],
                "fn": self._last_evaluation["fn"],
                "precision": self._last_evaluation["precision"],
                "recall": self._last_evaluation["recall"],
                "f1": self._last_evaluation["f1"],
                "reward": self._last_evaluation["reward"],
                "invalid_structure": self._last_evaluation["invalid_structure"],
                "error": self._last_evaluation.get("error"),
            }
        return info

    def _record_trajectory_step(
        self,
        action_payload: Any,
        observation: dict[str, Any],
        returned_reward: float,
    ) -> None:
        try:
            serialized_action = (
                action_payload.model_dump(mode="json")
                if hasattr(action_payload, "model_dump")
                else copy.deepcopy(action_payload)
            )
        except Exception:
            serialized_action = str(action_payload)

        assert_no_leaks(observation, forbidden_tokens=self._forbidden_tokens)

        self._trajectory.append(
            {
                "episode_id": self._episode_id,
                "step": self._step_count,
                "action": serialized_action,
                "observation": copy.deepcopy(observation),
                "reward": float(returned_reward),
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
            }
        )

    def _score_empty_truncation(self, reason: str) -> float:
        """Evaluate an empty submission when step limit or time limit truncates the episode."""
        assert self._evaluator_truth is not None
        empty_sub = FindingsSubmission(episode_id=self._episode_id, findings=[])
        eval_res = evaluate_submission(
            empty_sub,
            self._evaluator_truth,
            inspection_steps=self._inspection_steps,
        )
        eval_res["truncation_reason"] = reason
        self._last_evaluation = eval_res
        return float(eval_res["reward"]) if self.mode == "train" else 0.0

    def step(
        self,
        action: dict[str, Any] | Action | Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one agent action and return `(observation, reward, terminated, truncated, info)`."""
        if not self._is_reset or self._datastore is None or self._evaluator_truth is None:
            raise RuntimeError("Environment must be initialized via reset() before calling step().")

        # 1. Prevent post-termination / duplicate submission tampering
        if self._terminated or self._truncated:
            obs = {
                "error": "Episode has already finished. Call reset() to start a new episode.",
                "steps_remaining": max(0, self.max_steps - self._step_count),
            }
            info = self._build_step_info()
            assert_no_leaks({"observation": obs, "info": info}, forbidden_tokens=self._forbidden_tokens)
            return obs, 0.0, self._terminated, self._truncated, info

        # 2. Check wall-clock episode timeout before executing the action
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.timeout_seconds:
            self._step_count += 1
            self._truncated = True
            self._terminated = False
            returned_reward = self._score_empty_truncation("timeout")
            obs = {
                "error": "Episode time limit exceeded. Terminating with empty submission.",
                "steps_remaining": max(0, self.max_steps - self._step_count),
            }
            info = self._build_step_info()
            self._record_trajectory_step(action, obs, returned_reward)
            assert_no_leaks({"observation": obs, "info": info}, forbidden_tokens=self._forbidden_tokens)
            return obs, returned_reward, self._terminated, self._truncated, info

        # 3. Special handling for malformed `submit` payloads:
        # Section 05: "Неверная структура. Завершает эпизод с наградой -1 и ошибкой в журнале оценщика."
        is_submit_attempt = isinstance(action, SubmitAction) or (
            isinstance(action, dict)
            and (
                action.get("action") == "submit"
                or action.get("tool") == "submit"
                or action.get("name") == "submit"
                or action.get("type") == "submit"
            )
        )

        try:
            parsed_action = parse_action(action, default_episode_id=self._episode_id)
        except (ValidationError, TypeError, ValueError) as exc:
            self._step_count += 1
            if is_submit_attempt:
                self._terminated = True
                self._truncated = False
                raw_sub = action.get("submission") if isinstance(action, dict) else None
                eval_res = evaluate_submission(
                    raw_sub,
                    self._evaluator_truth,
                    inspection_steps=self._inspection_steps,
                )
                self._last_evaluation = eval_res
                if eval_res.get("error"):
                    self._evaluator_log.append(str(eval_res["error"]))
                returned_reward = float(eval_res["reward"]) if self.mode == "train" else 0.0
                obs = {
                    "error": "Invalid findings submission structure. Episode terminated with penalty.",
                    "steps_remaining": max(0, self.max_steps - self._step_count),
                }
                info = self._build_step_info()
                self._record_trajectory_step(action, obs, returned_reward)
                assert_no_leaks({"observation": obs, "info": info}, forbidden_tokens=self._forbidden_tokens)
                return obs, returned_reward, self._terminated, self._truncated, info

            # Non-submit invalid action consumes 1 inspection step and returns bounded error
            self._inspection_steps += 1
            returned_reward = 0.0
            if self._step_count >= self.max_steps:
                self._truncated = True
                returned_reward = self._score_empty_truncation("step_limit_reached")
            obs = {
                "error": f"Invalid action format or arguments: {exc.__class__.__name__}",
                "steps_remaining": max(0, self.max_steps - self._step_count),
            }
            info = self._build_step_info()
            self._record_trajectory_step(action, obs, returned_reward)
            assert_no_leaks({"observation": obs, "info": info}, forbidden_tokens=self._forbidden_tokens)
            return obs, returned_reward, self._terminated, self._truncated, info

        # 4. Execute valid parsed action
        self._step_count += 1

        if isinstance(parsed_action, SubmitAction):
            self._terminated = True
            self._truncated = False
            eval_res = evaluate_submission(
                parsed_action.submission,
                self._evaluator_truth,
                inspection_steps=self._inspection_steps,
            )
            self._last_evaluation = eval_res
            returned_reward = float(eval_res["reward"]) if self.mode == "train" else 0.0

            if eval_res["invalid_structure"]:
                if eval_res.get("error"):
                    self._evaluator_log.append(str(eval_res["error"]))
                obs = {
                    "error": "Invalid findings submission (e.g. episode_id mismatch). Episode terminated.",
                    "steps_remaining": max(0, self.max_steps - self._step_count),
                }
            else:
                obs = {
                    "result": {
                        "status": "submitted",
                        "episode_id": self._episode_id,
                        "findings_count": len(parsed_action.submission.findings),
                    },
                    "steps_remaining": max(0, self.max_steps - self._step_count),
                }
            info = self._build_step_info()
            self._record_trajectory_step(parsed_action, obs, returned_reward)
            assert_no_leaks({"observation": obs, "info": info}, forbidden_tokens=self._forbidden_tokens)
            return obs, returned_reward, self._terminated, self._truncated, info

        # All non-submit actions count as inspection steps
        self._inspection_steps += 1
        obs_payload: dict[str, Any] = {}

        if isinstance(parsed_action, ListRecordsAction):
            try:
                page = self._datastore.list_records(
                    table=parsed_action.table,
                    cursor=parsed_action.cursor,
                    limit=parsed_action.limit,
                )
                obs_payload["result"] = page
            except ValueError as exc:
                obs_payload["error"] = str(exc)

        elif isinstance(parsed_action, GetRecordAction):
            try:
                record = self._datastore.get_record(
                    table=parsed_action.table,
                    record_id=parsed_action.id,
                )
                if record is None:
                    safe_id = self._safe_id_for_error(parsed_action.id)
                    obs_payload["error"] = (
                        f"Record '{safe_id}' not found in table '{parsed_action.table}'."
                    )
                else:
                    obs_payload["result"] = record
            except ValueError as exc:
                obs_payload["error"] = str(exc)

        elif isinstance(parsed_action, GetDocumentAction):
            try:
                doc = self._datastore.get_document(doc_id=parsed_action.doc_id)
                if doc is None:
                    safe_doc_id = self._safe_id_for_error(parsed_action.doc_id)
                    obs_payload["error"] = f"Document '{safe_doc_id}' not found."
                else:
                    obs_payload["result"] = doc
            except ValueError as exc:
                obs_payload["error"] = str(exc)

        elif isinstance(parsed_action, CalculatorAction):
            try:
                calc_res = safe_calculate(parsed_action.expression)
                obs_payload["result"] = calc_res
            except CalculatorError as exc:
                obs_payload["error"] = str(exc)

        obs_payload["steps_remaining"] = max(0, self.max_steps - self._step_count)

        returned_reward = 0.0
        if self._step_count >= self.max_steps and not self._terminated:
            self._truncated = True
            returned_reward = self._score_empty_truncation("step_limit_reached")

        info = self._build_step_info()
        self._record_trajectory_step(parsed_action, obs_payload, returned_reward)
        assert_no_leaks({"observation": obs_payload, "info": info}, forbidden_tokens=self._forbidden_tokens)
        return obs_payload, returned_reward, self._terminated, self._truncated, info

    def get_last_evaluation(self) -> dict[str, Any] | None:
        """Return the internal evaluator scoring dictionary for the current/finished episode."""
        return copy.deepcopy(self._last_evaluation)

    def get_evaluator_log(self) -> list[str]:
        """Return a copy of the evaluator error log for the current episode."""
        return list(self._evaluator_log)

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return a deep copy of recorded trajectory steps for the current episode."""
        return copy.deepcopy(self._trajectory)

    def dump_trajectory_jsonl(self, path: Path | str, *, append: bool = True) -> None:
        """Write or append the recorded episode trajectory to a JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if append else "w"
        with out_path.open(file_mode, encoding="utf-8") as f:
            for entry in self._trajectory:
                assert_no_leaks(entry.get("observation"), forbidden_tokens=self._forbidden_tokens)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Alias for convenience
AuditEnv = FinAuditEnv

__all__ = [
    "TASK_DESCRIPTION",
    "FinAuditEnv",
    "AuditEnv",
]

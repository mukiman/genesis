"""Abstract base policy interface (`base.py`).

Defines the uniform contract (`reset`, `act`, `run_episode`) for policies
interacting with `FinAuditEnv` or `IsolatedFinAuditEnv`.
"""

from __future__ import annotations

import time
from typing import Any


class BasePolicy:
    """Base interface for all financial audit RL and heuristic policies."""

    name: str = "base_policy"

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = int(seed)
        self._episode_id: str = ""
        self._manifest: dict[str, Any] = {}
        self._policy: dict[str, Any] = {}

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Reset internal policy state at the start of an episode."""
        self._manifest = dict(observation.get("manifest") or {})
        self._policy = dict(observation.get("policy") or {})
        self._episode_id = str(
            self._manifest.get("episode_id") or info.get("episode_id") or "ep_unknown"
        )

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Select the next JSON-serializable action given the latest observation."""
        raise NotImplementedError("BasePolicy.act must be implemented by subclasses.")

    def run_episode(
        self,
        env: Any,
        *,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        already_reset: bool = False,
        initial_obs: dict[str, Any] | None = None,
        initial_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a full episode loop on `env` until `terminated` or `truncated`."""
        start_time = time.monotonic()

        if already_reset and initial_obs is not None and initial_info is not None:
            obs, info = initial_obs, initial_info
        else:
            obs, info = env.reset(seed=seed if seed is not None else self.seed, config=config)

        self.reset(obs, info)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_action: dict[str, Any] | None = None

        while not (terminated or truncated):
            action = self.act(obs, info)
            last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps > getattr(env, "MAX_STEPS", 100) + 2:
                break

        elapsed_seconds = time.monotonic() - start_time
        eval_res = env.get_last_evaluation() if hasattr(env, "get_last_evaluation") else None

        return {
            "policy": self.name,
            "episode_id": self._episode_id,
            "steps": steps,
            "returned_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "last_action": last_action,
            "evaluation": eval_res,
        }


__all__ = ["BasePolicy"]

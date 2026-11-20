"""Isolated client wrapper for agent interaction (`isolated_client.py`).

Provides the standard `reset(seed, config)` and `step(action)` interface to an
agent while communicating with an out-of-process `EvaluatorServer` over
`subprocess` stdin/stdout JSON-RPC.
Ensures zero ground-truth objects (`TruthFile`, `TruthIssue`, `provenance.json`)
exist in the agent's Python process memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from finaudit_env.env.core import FinAuditEnv


class IsolatedFinAuditEnv:
    """Client proxy that runs `evaluator_server.py` in a separate subprocess and forwards `reset`/`step`."""

    MAX_STEPS: int = FinAuditEnv.MAX_STEPS

    def __init__(
        self,
        mode: Literal["train", "eval", "closed"] = "train",
        *,
        max_steps: int = FinAuditEnv.MAX_STEPS,
        timeout_seconds: float = FinAuditEnv.DEFAULT_TIMEOUT_SECONDS,
        episode_dir: Path | str | None = None,
    ) -> None:
        self.mode: str = "eval" if mode == "closed" else mode
        self.max_steps: int = int(max_steps)
        self.timeout_seconds: float = float(timeout_seconds)
        self._default_episode_dir: str | None = str(Path(episode_dir)) if episode_dir is not None else None
        self._req_counter: int = 0
        self._proc: subprocess.Popen[str] | None = None

        self._start_subprocess()
        self._rpc(
            "init",
            {
                "mode": self.mode,
                "max_steps": self.max_steps,
                "timeout_seconds": self.timeout_seconds,
                "episode_dir": self._default_episode_dir,
            },
        )

    def _start_subprocess(self) -> None:
        env_vars = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = env_vars.get("PYTHONPATH", "")
        env_vars["PYTHONPATH"] = (
            f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_root
        )

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "finaudit_env.env.evaluator_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env_vars,
        )

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Isolated evaluator subprocess is not running.")
        if self._proc.poll() is not None:
            raise RuntimeError("Isolated evaluator subprocess terminated unexpectedly.")

        self._req_counter += 1
        req_id = self._req_counter
        payload = {
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        raw_response = self._proc.stdout.readline()
        if not raw_response:
            stderr_output = ""
            if self._proc.stderr is not None:
                try:
                    stderr_output = self._proc.stderr.read()
                except Exception:
                    stderr_output = ""
            raise RuntimeError(f"Empty response from evaluator subprocess. Stderr: {stderr_output[:200]}")

        resp = json.loads(raw_response)
        if not resp.get("ok", False):
            err_msg = resp.get("error", "Unknown IPC error from evaluator server.")
            raise RuntimeError(err_msg)

        result = resp.get("result")
        return result if isinstance(result, dict) else {}

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward `reset` to the isolated evaluator server."""
        cfg: dict[str, Any] = {}
        if config is not None:
            for k, v in config.items():
                # Convert Path objects to strings for JSON serialization over IPC
                cfg[k] = str(v) if isinstance(v, Path) else v

        res = self._rpc("reset", {"seed": seed, "config": cfg})
        return res["observation"], res["info"]

    def step(
        self,
        action: dict[str, Any] | Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Forward `step` to the isolated evaluator server."""
        serialized_action = (
            action.model_dump(mode="json") if hasattr(action, "model_dump") else action
        )
        res = self._rpc("step", {"action": serialized_action})
        return (
            res["observation"],
            float(res["reward"]),
            bool(res["terminated"]),
            bool(res["truncated"]),
            res["info"],
        )

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Fetch the recorded trajectory from the isolated evaluator server."""
        res = self._rpc("get_trajectory")
        traj = res.get("trajectory", [])
        return list(traj) if isinstance(traj, list) else []

    def get_last_evaluation(self) -> dict[str, Any] | None:
        """Fetch the last evaluation result from the evaluator server (for benchmark runner)."""
        res = self._rpc("get_last_evaluation")
        evaluation = res.get("evaluation")
        return evaluation if isinstance(evaluation, dict) else None

    def dump_trajectory_jsonl(self, path: Path | str, *, append: bool = True) -> None:
        """Write or append the recorded trajectory from the isolated server to a JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        traj = self.get_trajectory()
        file_mode = "a" if append else "w"
        with out_path.open(file_mode, encoding="utf-8") as f:
            for entry in traj:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Shut down the isolated evaluator subprocess cleanly."""
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._rpc("shutdown")
        except Exception:
            pass
        finally:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    self._proc.wait(timeout=3.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def __enter__(self) -> IsolatedFinAuditEnv:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Backward-compatible aliases
IsolatedFinAuditClient = IsolatedFinAuditEnv
IsolatedEnvClient = IsolatedFinAuditEnv

__all__ = [
    "IsolatedFinAuditEnv",
    "IsolatedFinAuditClient",
    "IsolatedEnvClient",
]

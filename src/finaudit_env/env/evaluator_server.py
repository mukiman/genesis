"""Out-of-process evaluator server (`evaluator_server.py`).

Hosts the environment and closed ground-truth evaluator (`truth.json`,
`provenance.json`) in an isolated process so that the agent process cannot
access truth files, seeds, or internal evaluator memory.
Communicates via line-delimited JSON-RPC over stdin/stdout.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from finaudit_env.env.core import FinAuditEnv
from finaudit_env.env.leak_checker import assert_no_leaks

_PATH_SANITIZE_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\"']+|/(?:Users|home|tmp|var|app)/[^\s\"']+)")


def _sanitize_error_message(message: str) -> str:
    """Remove any accidental filesystem paths or closed file references from error strings."""
    cleaned = _PATH_SANITIZE_RE.sub("<path>", message)
    for secret_word in ("truth.json", "provenance.json", "valid_evidence_sets", "expected_amount_minor"):
        cleaned = cleaned.replace(secret_word, "<redacted>")
    return cleaned


class EvaluatorServer:
    """Isolated evaluator server handling `init`, `reset`, `step`, and inspection requests over JSON IPC."""

    def __init__(
        self,
        mode: Literal["train", "eval", "closed"] = "train",
        *,
        max_steps: int = FinAuditEnv.MAX_STEPS,
        timeout_seconds: float = FinAuditEnv.DEFAULT_TIMEOUT_SECONDS,
        episode_dir: Path | str | None = None,
        episode_data: dict[str, Any] | None = None,
    ) -> None:
        self._env = FinAuditEnv(
            mode=mode,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            episode_dir=episode_dir,
            episode_data=episode_data,
        )
        self._should_shutdown: bool = False

    @property
    def should_shutdown(self) -> bool:
        return self._should_shutdown

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a single JSON IPC request without leaking truth or filesystem paths."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if not isinstance(method, str):
            return {"id": req_id, "ok": False, "error": "Request 'method' must be a string."}

        try:
            if method == "ping":
                return {"id": req_id, "ok": True, "result": {"status": "pong"}}

            if method == "shutdown":
                self._should_shutdown = True
                return {"id": req_id, "ok": True, "result": {"status": "shutting_down"}}

            if method == "init":
                mode = params.get("mode", "train")
                max_steps = int(params.get("max_steps", FinAuditEnv.MAX_STEPS))
                timeout_seconds = float(
                    params.get("timeout_seconds", FinAuditEnv.DEFAULT_TIMEOUT_SECONDS)
                )
                ep_dir = params.get("episode_dir")
                ep_data = params.get("episode_data")
                self._env = FinAuditEnv(
                    mode=mode,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    episode_dir=ep_dir,
                    episode_data=ep_data,
                )
                return {"id": req_id, "ok": True, "result": {"status": "initialized"}}

            if method == "reset":
                seed = params.get("seed")
                config = params.get("config")
                obs, info = self._env.reset(seed=seed, config=config)
                payload = {"observation": obs, "info": info}
                assert_no_leaks(payload, forbidden_tokens=self._env._forbidden_tokens)
                return {"id": req_id, "ok": True, "result": payload}

            if method == "step":
                action = params.get("action")
                obs, reward, terminated, truncated, info = self._env.step(action)
                payload = {
                    "observation": obs,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
                assert_no_leaks(
                    {"observation": obs, "info": info},
                    forbidden_tokens=self._env._forbidden_tokens,
                )
                return {"id": req_id, "ok": True, "result": payload}

            if method == "get_trajectory":
                traj = self._env.get_trajectory()
                for entry in traj:
                    assert_no_leaks(
                        entry.get("observation"),
                        forbidden_tokens=self._env._forbidden_tokens,
                    )
                return {"id": req_id, "ok": True, "result": {"trajectory": traj}}

            if method == "get_last_evaluation":
                # Only returns evaluation summary (with closed issue IDs stripped if needed)
                eval_res = self._env.get_last_evaluation()
                return {"id": req_id, "ok": True, "result": {"evaluation": eval_res}}

            return {"id": req_id, "ok": False, "error": f"Unknown IPC method: '{method}'"}

        except Exception as exc:
            safe_msg = _sanitize_error_message(f"{exc.__class__.__name__}: {exc}")
            return {"id": req_id, "ok": False, "error": safe_msg}


def run_stdio_server() -> None:
    """Run the line-delimited JSON-RPC evaluator server loop over stdin/stdout."""
    server = EvaluatorServer()
    for raw_line in sys.stdin:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
            if not isinstance(request, dict):
                response: dict[str, Any] = {
                    "id": None,
                    "ok": False,
                    "error": "JSON-RPC request must be a JSON object.",
                }
            else:
                response = server.handle_request(request)
        except json.JSONDecodeError as exc:
            response = {"id": None, "ok": False, "error": f"Invalid JSON request: {exc.msg}"}

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

        if server.should_shutdown:
            break


if __name__ == "__main__":
    run_stdio_server()


__all__ = [
    "EvaluatorServer",
    "run_stdio_server",
]

"""Interactive / scripted demonstration entrypoint (`demo.py`).

Demonstrates a step-by-step audit episode walkthrough using the RL environment
interface (`reset` / `step`) and prints observations, actions, and final reward.
"""

from __future__ import annotations

from typing import Any


def run_demo(seed: int = 42, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a single demonstration episode and return trajectory summary."""
    raise NotImplementedError("run_demo is not implemented yet.")


def main() -> None:
    """CLI entrypoint for demo run."""
    raise NotImplementedError("demo CLI is not implemented yet.")


if __name__ == "__main__":
    main()

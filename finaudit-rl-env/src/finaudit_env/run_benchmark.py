"""Benchmark runner entrypoint (`run_benchmark.py`).

Runs the 4 baseline policies (`random_explorer`, `simple_reconciler`,
`flag_all_attack`, `oracle`) across evaluation episodes, records trajectories
to `artifacts/trajectories.jsonl`, and outputs diagnostic metrics (P, R, F1,
terminal_reward, steps, elapsed_time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_benchmark(data_dir: Path | str = "data/dev", artifacts_dir: Path | str = "artifacts") -> dict[str, Any]:
    """Execute baseline policies on episodes and compute diagnostic metrics."""
    raise NotImplementedError("run_benchmark is not implemented yet (Participant 2: feat/env-scoring).")


def main() -> None:
    """CLI entrypoint for benchmark execution."""
    raise NotImplementedError("run_benchmark CLI is not implemented yet.")


if __name__ == "__main__":
    main()

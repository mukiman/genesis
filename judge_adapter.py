"""Top-level `judge_adapter.py` entrypoint for the Genesis FinTech jury harness.

Re-exports `generate_episode`, `make_env`, and `score_episode` from
`finaudit_env.judge_adapter` so the judge container can import `judge_adapter`
from the repository root or installed site-packages.
"""

from __future__ import annotations

from finaudit_env.judge_adapter import (
    EVALUATOR_ERROR_LOG,
    JudgeEnvAdapter,
    generate_episode,
    make_env,
    score_episode,
)

__all__ = [
    "EVALUATOR_ERROR_LOG",
    "JudgeEnvAdapter",
    "generate_episode",
    "make_env",
    "score_episode",
]

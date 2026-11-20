"""RL environment, read-only datastore, safe calculator, and evaluator isolation (`finaudit_env.env`).

Owned by Participant 2 (`feat/env-scoring` branch).
"""

from finaudit_env.env.calculator import CalculatorError, evaluate_expression, safe_calculate
from finaudit_env.env.core import AuditEnv, FinAuditEnv, TASK_DESCRIPTION
from finaudit_env.env.datastore import EpisodeDataStore, EpisodeDatastore, MAX_PAGE_LIMIT, OPEN_FILENAMES
from finaudit_env.env.evaluator_server import EvaluatorServer, run_stdio_server
from finaudit_env.env.isolated_client import (
    IsolatedEnvClient,
    IsolatedFinAuditClient,
    IsolatedFinAuditEnv,
)
from finaudit_env.env.leak_checker import (
    assert_no_leaks,
    check_observation_for_leaks,
    scan_payload_for_leaks,
    verify_trajectory_file,
)

__all__ = [
    "TASK_DESCRIPTION",
    "FinAuditEnv",
    "AuditEnv",
    "EpisodeDataStore",
    "EpisodeDatastore",
    "MAX_PAGE_LIMIT",
    "OPEN_FILENAMES",
    "CalculatorError",
    "safe_calculate",
    "evaluate_expression",
    "EvaluatorServer",
    "run_stdio_server",
    "IsolatedFinAuditEnv",
    "IsolatedFinAuditClient",
    "IsolatedEnvClient",
    "assert_no_leaks",
    "check_observation_for_leaks",
    "scan_payload_for_leaks",
    "verify_trajectory_file",
]

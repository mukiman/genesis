"""FinAudit RL Environment package (`finaudit_env`).

Reproducible financial audit simulator, causal error generator, read-only
investigation environment, 1-to-1 finding matcher, and reward evaluator.
"""

from finaudit_env.env.core import AuditEnv, FinAuditEnv
from finaudit_env.env.isolated_client import IsolatedFinAuditClient, IsolatedFinAuditEnv
from finaudit_env.schemas import SCHEMA_VERSION

__version__ = SCHEMA_VERSION

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "FinAuditEnv",
    "AuditEnv",
    "IsolatedFinAuditEnv",
    "IsolatedFinAuditClient",
]

"""FinAudit RL Environment package (`finaudit_env`).

Provides shared Pydantic v2 data contracts, synthetic company generator,
isolated RL audit environment, 1-to-1 finding matcher, terminal reward engine,
and baseline policies.
"""

from finaudit_env.schemas import CURRENCY_EXPONENTS, SCHEMA_VERSION, IssueType

__all__ = ["SCHEMA_VERSION", "CURRENCY_EXPONENTS", "IssueType"]

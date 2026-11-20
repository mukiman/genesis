"""Baseline policy adapters subpackage (`finaudit_env.policies`).

Provides the 4 mandatory baseline policies for evaluator benchmarking:
- `random_explorer`: Fixed-seed random investigation policy.
- `simple_reconciler`: Rule-based baseline reconciliation policy.
- `flag_all_attack`: Specification-gaming adversarial policy ("flag everything").
- `oracle`: Evaluator-only ground-truth oracle policy verifying maximum reward.
"""

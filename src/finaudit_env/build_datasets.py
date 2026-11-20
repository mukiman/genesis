"""Dataset builder entrypoint (`build_datasets.py`).

Generates reproducible `train` and `dev` episode directories under `data/`
from configuration files in `configs/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_datasets(config_dir: Path | str = "configs", output_dir: Path | str = "data") -> dict[str, Any]:
    """Generate open and closed episode files for train and dev splits."""
    raise NotImplementedError("build_datasets is not implemented yet (Participant 1: feat/generator).")


def main() -> None:
    """CLI entrypoint for dataset generation."""
    raise NotImplementedError("build_datasets CLI is not implemented yet.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML experiment config from disk.

    Parameters
    ----------
    config_path:
        Path to a YAML file with paths, experiment settings, model params, and outputs.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
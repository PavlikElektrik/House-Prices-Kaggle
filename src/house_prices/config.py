from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
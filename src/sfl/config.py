"""Load the YAML configs in ``configs/``."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or CONFIG_DIR) / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def paper_config() -> dict:
    return load_yaml("paper.yaml")


def impl_defaults() -> dict:
    return load_yaml("impl_defaults.yaml")


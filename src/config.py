from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
ALGORITHM_CONFIG_DIR = CONFIG_DIR / "algorithms"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def load_env_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "env.yaml")


def load_algorithm_config(algo: str) -> dict[str, Any]:
    path = ALGORITHM_CONFIG_DIR / f"{algo}.yaml"
    if not path.exists():
        available = ", ".join(available_algorithms())
        raise FileNotFoundError(f"Unknown algorithm '{algo}'. Available: {available}")
    return load_yaml(path)


def available_algorithms() -> list[str]:
    return sorted(path.stem for path in ALGORITHM_CONFIG_DIR.glob("*.yaml"))

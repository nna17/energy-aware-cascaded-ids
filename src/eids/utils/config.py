"""YAML configs as attribute-accessible dicts. Paths inside configs are relative to the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Config(dict):
    """dict with attribute access; nested dicts become Config too."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str) -> Path | None:
        value = self.get(key)
        return None if value is None else resolve_path(value)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path, **overrides: Any) -> Config:
    with open(resolve_path(path), encoding="utf-8") as f:
        cfg = _wrap(yaml.safe_load(f) or {})
    cfg.update(_wrap(overrides))
    return cfg

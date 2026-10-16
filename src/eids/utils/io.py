import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eids.utils.config import resolve_path


def make_run_dir(experiment: str, base: str = "results/runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(base) / experiment / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

"""Training time and inference latency of representative detectors (one seed, this machine's CPU).

Training uses all cores (n_jobs=-1, as in the pool); inference is timed on one thread over a batch of test flows
and reported per flow. Single-flow calls are not timed: in Python they measure interpreter overhead, not the model.

    python experiments/timing.py --data configs/data/ciciiot2025.yaml --out results/runs/timing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eids.cascade.pool import make_detector
from eids.data.datamodule import load_datamodule
from eids.utils.config import load_config, resolve_path
from eids.utils.logging import get_logger
from eids.utils.seed import set_seed

log = get_logger()

SPECS = [
    {"kind": "decision_tree", "max_leaves": 256, "n_features": 20},
    {"kind": "decision_tree", "max_leaves": 1024},
    {"kind": "random_forest", "n_estimators": 10, "max_leaves": 512},
    {"kind": "extra_trees", "n_estimators": 10, "max_leaves": 512},
    {"kind": "random_forest", "n_estimators": 100, "max_leaves": None, "balanced": False, "tag": "unw"},
    {"kind": "lightgbm", "n_estimators": 200, "max_leaves": 32},
    {"kind": "mlp", "hidden": [32], "n_features": 20, "epochs": 10},
]
BATCH = 20000


def single_thread(model) -> None:
    """Inference on one thread: per-flow time then reflects the model, not parallel dispatch overhead."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "n_jobs"):
        inner.set_params(n_jobs=1)
    torch = __import__("torch")
    torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--extra", help="pool_strong yaml whose tuned_f1 LightGBM is also timed")
    parser.add_argument("--out", default="results/runs/timing")
    args = parser.parse_args()
    set_seed(0)
    data_cfg = load_config(args.data)
    dm = load_datamodule(data_cfg)
    specs = list(SPECS)
    if args.extra:
        specs += [{k: (v[0] if isinstance(v, list) else v) for k, v in m.items()}
                  for m in load_config(args.extra).models if m.get("tag") == "tuned_f1"]
    X = dm.X["test"][:BATCH]
    rows = []
    for spec in specs:
        model = make_detector(spec, 0)
        t0 = time.perf_counter()
        model.fit(dm.X["train"], dm.y["train"], balanced=spec.get("balanced", True))
        fit_s = time.perf_counter() - t0
        single_thread(model)
        model.predict_proba(X[:100])  # warm-up
        t0 = time.perf_counter()
        model.predict_proba(X)
        us = (time.perf_counter() - t0) / len(X) * 1e6
        rows.append({"dataset": Path(args.data).stem, "model": model.name, "train_rows": len(dm.y["train"]),
                     "fit_s": fit_s, "us_per_flow": us, "ops": model.n_comparisons(), "size_kb": model.size_kb()})
        log.info(f"{model.name}: fit {fit_s:.1f}s, {us:.2f} us/flow (one thread)")
        del model
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"timing_{Path(args.data).stem}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()

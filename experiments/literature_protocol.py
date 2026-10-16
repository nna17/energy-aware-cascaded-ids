"""Replicate the evaluation protocol common in the literature and compare it with ours, same models.

Published protocol: raw rows (duplicates and contradictory labels kept, no class capping), random
stratified 80/20 split, default unweighted Random Forest (100 unrestricted trees) and LightGBM.
Our protocol: the cleaned, deduplicated, per-class split data used in the paper.
Reported at fine, category and binary label levels: accuracy, macro-F1 and weighted-F1.

    python experiments/literature_protocol.py --data configs/data/ciciot2023.yaml --raw-files 3 --sample 500000
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eids.data.adapters import get_adapter
from eids.data.adapters.base import FINE_LABEL
from eids.data.datamodule import load_datamodule
from eids.data.labels import add_label_levels
from eids.utils.config import load_config
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()


def models(seed: int) -> dict:
    return {
        "random_forest_default": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        # min_child_weight=1: the LightGBM default (1e-3) diverges on these many-class problems
        "lightgbm_default": lgb.LGBMClassifier(min_child_weight=1.0, random_state=seed, n_jobs=-1, verbose=-1),
    }


def score(y_true, y_pred, levels: dict[str, np.ndarray], fine_classes: list[str]) -> dict:
    """levels maps level name -> array mapping fine-class index to that level's label."""
    out = {}
    for level, mapping in levels.items():
        t, p = mapping[y_true], mapping[y_pred]
        out |= {f"{level}_accuracy": accuracy_score(t, p), f"{level}_macro_f1": f1_score(t, p, average="macro"),
                f"{level}_weighted_f1": f1_score(t, p, average="weighted")}
    return out


def level_maps(classes: list[str], adapter) -> dict[str, np.ndarray]:
    cat = np.array([adapter.category_of(c) for c in classes])
    return {"fine": np.array(classes), "category": cat, "binary": np.where(cat == "Benign", "Benign", "Attack")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--raw-files", type=int, help="limit raw CSV files read (CIC-IoT2023)")
    parser.add_argument("--sample", type=int, default=500_000, help="random raw rows used (with duplicates)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    cfg = load_config(args.data, **({"max_files": args.raw_files} if args.raw_files else {}))
    adapter = get_adapter(cfg)
    run_dir = make_run_dir(f"literature_protocol_{adapter.name}")
    get_logger(log_file=run_dir / "log.txt")

    raw = adapter.load_raw()
    raw = raw.drop(columns=[c for c in adapter.leaky_columns() + list(cfg.get("extra_leaky_columns") or [])
                            if c in raw.columns])
    raw = add_label_levels(raw, adapter)
    feats = [c for c in raw.columns if not c.startswith("label_") and pd.api.types.is_numeric_dtype(raw[c])]
    stats = {"raw_rows": len(raw), "raw_duplicate_share": float(raw.duplicated().mean()), "features": len(feats)}
    if len(raw) > args.sample:
        raw = raw.sample(n=args.sample, random_state=0)
    classes = sorted(raw["label_fine"].unique())
    index = {c: i for i, c in enumerate(classes)}
    X_raw = raw[feats].replace([np.inf, -np.inf], np.nan).fillna(-1).to_numpy(np.float32)
    y_raw = raw["label_fine"].map(index).to_numpy()
    maps_raw = level_maps(classes, adapter)

    dm = load_datamodule(cfg)  # our protocol (same cleaned data as the paper)
    maps_ours = level_maps(dm.labels.classes, adapter)
    log.info(stats)

    rows = []
    for seed in args.seeds:
        Xtr, Xte, ytr, yte = train_test_split(X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=seed)
        for name, model in models(seed).items():
            model.fit(Xtr, ytr)
            rows.append({"protocol": "published (duplicates kept, random split)", "model": name, "seed": seed,
                         **score(yte, np.asarray(model.predict(Xte)).astype(int), maps_raw, classes)})
            log.info(rows[-1])
        for name, model in models(seed).items():
            model.fit(dm.X["train"], dm.y["train"])
            rows.append({"protocol": "ours (deduplicated, cleaned)", "model": name, "seed": seed,
                         **score(dm.y["test"], np.asarray(model.predict(dm.X["test"])).astype(int), maps_ours,
                                 dm.labels.classes)})
            log.info(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "results.csv", index=False)
    save_json({**stats, "sample": args.sample, "raw_files": args.raw_files}, run_dir / "stats.json")
    cols = [c for c in df.columns if c.endswith(("accuracy", "macro_f1"))]
    with pd.option_context("display.width", 250):
        log.info("\n" + df.groupby(["protocol", "model"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()

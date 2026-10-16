"""Raw dataset -> cleaned, labelled, split, transformed arrays on disk.

load chunks (dedupe + cap per class) -> clean -> label levels -> split -> fit transform on train
-> rank features -> save
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eids.data.adapters import get_adapter
from eids.data.adapters.base import FINE_LABEL, META_DEVICE, META_PREFIX, META_TIME
from eids.data.cleaning import clean
from eids.data.labels import add_label_levels, benign_name, build_label_space, label_column
from eids.data.preprocess import Preprocessor, rank_features
from eids.data.splits import collect_chunks, device_split, stratified_split, temporal_split
from eids.utils.config import Config
from eids.utils.io import save_json
from eids.utils.logging import get_logger

log = get_logger()
SPLITS = ("train", "val", "test")


def processed_path(cfg: Config) -> Path:
    return cfg.get_path("processed_dir") / cfg.label_level


def prepare_dataset(cfg: Config) -> Path:
    adapter = get_adapter(cfg)
    out_dir = processed_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cap on fine labels so every attack keeps its share inside its category.
    df, collect_report = collect_chunks(adapter.iter_chunks(), FINE_LABEL, cfg.get("max_rows_per_class"), cfg.seed,
                                        uncapped=cfg.get("uncapped_classes") or [])
    log.info(f"[{adapter.name}] loading: {collect_report.as_dict()}")
    leaky = adapter.leaky_columns() + list(cfg.get("extra_leaky_columns") or [])
    df, report = clean(df, FINE_LABEL, leaky, cfg.get("drop_conflicting_duplicates", True))
    log.info(f"[{adapter.name}] cleaning: {report.as_dict()}")

    df = add_label_levels(df, adapter)
    label_col = label_column(cfg.label_level)

    split_info = {}
    mode = cfg.get("split_mode", "random")
    fine = df["label_fine"].to_numpy()
    if mode == "random":
        split_idx = stratified_split(fine, dict(cfg.split), cfg.seed)
    elif mode == "temporal":
        split_idx = temporal_split(fine, df[META_TIME].to_numpy(), dict(cfg.split))
    elif mode == "device":
        split_idx, held_out = device_split(fine, df[META_DEVICE].to_numpy(), dict(cfg.split), cfg.seed,
                                           cfg.get("test_device_share", 0.2))
        split_info = {"held_out_devices": held_out,
                      "test_classes": sorted(set(fine[split_idx["test"]])),
                      "dropped_test_records": int(np.isin(df[META_DEVICE], held_out).sum() - len(split_idx["test"]))}
    else:
        raise ValueError(f"unknown split_mode {mode!r}")
    parts = {name: df.iloc[idx] for name, idx in split_idx.items()}

    labels = build_label_space(parts["train"][label_col], cfg.label_level,
                               benign_name(cfg.label_level, adapter), cfg.minority_threshold)
    feature_cols = [c for c in df.columns if not c.startswith("label_") and not c.startswith(META_PREFIX)]
    pre = Preprocessor(scaler=cfg.get("scaler", "quantile")).fit(parts["train"], feature_cols)

    arrays = {}
    for name in SPLITS:
        arrays[f"X_{name}"] = pre.transform(parts[name])
        arrays[f"y_{name}"] = labels.encode(parts[name][label_col])

    order = rank_features(arrays["X_train"], arrays["y_train"], seed=cfg.seed)
    for name in SPLITS:
        arrays[f"X_{name}"] = arrays[f"X_{name}"][:, order]
    np.savez_compressed(out_dir / "arrays.npz", **arrays)

    meta = {
        "dataset": adapter.name,
        "label_level": cfg.label_level,
        "classes": labels.classes,
        "benign_index": labels.benign_index,
        "minority": labels.minority,
        "feature_names": [pre.feature_names[i] for i in order],  # ranked, most important first
        "loading": collect_report.as_dict(),
        "cleaning": report.as_dict(),
        "split_mode": mode,
        **split_info,
        "class_counts": {
            name: dict(zip(*np.unique(parts[name][label_col], return_counts=True))) for name in SPLITS
        },
    }
    save_json(meta, out_dir / "meta.json")
    log.info(f"[{adapter.name}] saved {out_dir} | features={len(order)} classes={labels.n_classes} "
             f"minority={[labels.classes[i] for i in labels.minority]}")
    return out_dir

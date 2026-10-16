"""Candidate pool: tier models trained once, stored as per-flow (predicted class, confidence, margin).

That is all a cascade needs, so the search evaluates thousands of deployments without retraining
or re-running any model, and the pool stays small (~5 bytes per flow per model). Confidence is the
maximum class probability; margin is top-1 minus top-2 probability (an alternative uncertainty score).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eids.data.adapters import get_adapter
from eids.data.datamodule import DataModule
from eids.models.baselines.mlp_detector import MLPDetector
from eids.models.baselines.trees import TreeModel
from eids.utils.config import Config
from eids.utils.io import load_json, save_json
from eids.utils.logging import get_logger

log = get_logger()
SPLITS = ("val", "test")


@dataclass
class Pool:
    names: list[str]
    cost: np.ndarray        # operations per flow: worst-case comparisons (trees) or multiply-accumulates (MLP)
    size_kb: np.ndarray
    n_features: np.ndarray  # flow features the model needs (feature-extraction cost)
    pred: dict[str, np.ndarray]  # split -> (n_models, n_flows) int16
    conf: dict[str, np.ndarray]  # split -> (n_models, n_flows) float16, max class probability
    margin: dict[str, np.ndarray] | None  # split -> (n_models, n_flows) float16, top-1 minus top-2
    y: dict[str, np.ndarray]
    classes: list[str]
    class_group: dict[str, np.ndarray]  # granularity -> group id per class ("global", "category", "class")
    minority: list[int]
    benign_index: int

    @property
    def n_models(self) -> int:
        return len(self.names)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def model_grid(cfg: Config, n_features_total: int) -> list[dict]:
    """Expand the pool config into model specs."""
    specs = []
    for family in cfg:
        feats = family.get("n_features") or [None]
        if family.kind == "mlp":
            for hidden in family["hidden"]:
                for k in feats:
                    spec = {"kind": "mlp", "hidden": list(hidden), "epochs": family.get("epochs", 10)}
                    if k and k < n_features_total:
                        spec["n_features"] = k
                    specs.append(spec)
            continue
        extra = {k: v for k, v in family.items() if k not in ("kind", "n_estimators", "max_leaves", "n_features")}
        for n in family.get("n_estimators", [1]):
            for leaves in family["max_leaves"]:
                for k in feats:
                    spec = {"kind": family.kind, "n_estimators": n, "max_leaves": leaves, **extra}
                    if k and k < n_features_total:
                        spec["n_features"] = k
                    specs.append(spec)
    return specs


def make_detector(spec: dict, seed: int):
    params = {k: v for k, v in spec.items() if k not in ("kind", "balanced")}
    return (MLPDetector if spec["kind"] == "mlp" else TreeModel)(spec["kind"], params, seed)


def build_pool(specs: list[dict], dm: DataModule, data_cfg: Config, seed: int, out_dir: Path,
               balanced: bool = True) -> Path:
    """Train every spec and store its per-flow predictions. If a pool already exists in `out_dir`, its
    models are kept and only specs not in it are trained (so adding a model family is cheap)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names, cost, size, feats = [], [], [], []
    pred = {s: [] for s in SPLITS}
    conf = {s: [] for s in SPLITS}
    margin = {s: [] for s in SPLITS}
    if (out_dir / "pool.npz").exists():
        old = load_pool(out_dir)
        if old.margin is None:
            raise ValueError(f"{out_dir} predates margin storage; rebuild it into a new directory")
        names, cost, size, feats = list(old.names), list(old.cost), list(old.size_kb), list(old.n_features)
        for s in SPLITS:
            pred[s], conf[s], margin[s] = list(old.pred[s]), list(old.conf[s]), list(old.margin[s])
        log.info(f"extending existing pool of {len(names)} models in {out_dir}")
    todo = [s for s in specs if make_detector(s, seed).name not in names]
    for i, spec in enumerate(todo, 1):
        model = make_detector(spec, seed).fit(dm.X["train"], dm.y["train"], balanced=spec.get("balanced", balanced))
        names.append(model.name)
        cost.append(model.n_comparisons())
        size.append(model.size_kb())
        feats.append(spec.get("n_features") or dm.n_features)
        for s in SPLITS:
            p = model.predict_proba(dm.X[s])
            top2 = np.sort(p, axis=1)[:, -2:]
            pred[s].append(p.argmax(axis=1).astype(np.int16))
            conf[s].append(top2[:, 1].astype(np.float16))
            margin[s].append((top2[:, 1] - top2[:, 0]).astype(np.float16))
        log.info(f"[pool {i}/{len(todo)}] {model.name}: {cost[-1]} comparisons, {size[-1]:.1f} KB")
        del model

    adapter = get_adapter(data_cfg)
    categories = [adapter.category_of(c) for c in dm.labels.classes]
    cat_ids = {c: j for j, c in enumerate(sorted(set(categories)))}
    np.savez_compressed(
        out_dir / "pool.npz",
        cost=np.array(cost, dtype=np.float64), size_kb=np.array(size), n_features=np.array(feats),
        **{f"pred_{s}": np.stack(pred[s]) for s in SPLITS},
        **{f"conf_{s}": np.stack(conf[s]) for s in SPLITS},
        **{f"margin_{s}": np.stack(margin[s]) for s in SPLITS},
        **{f"y_{s}": dm.y[s] for s in SPLITS},
        group_category=np.array([cat_ids[c] for c in categories]),
    )
    save_json({"names": names, "classes": dm.labels.classes, "categories": categories,
               "minority": dm.labels.minority, "benign_index": dm.labels.benign_index, "seed": seed},
              out_dir / "pool.json")
    return out_dir


def load_pool(path: Path) -> Pool:
    a = np.load(path / "pool.npz")
    meta = load_json(path / "pool.json")
    n_classes = len(meta["classes"])
    return Pool(
        names=meta["names"], cost=a["cost"], size_kb=a["size_kb"], n_features=a["n_features"],
        pred={s: a[f"pred_{s}"] for s in SPLITS}, conf={s: a[f"conf_{s}"] for s in SPLITS},
        margin={s: a[f"margin_{s}"] for s in SPLITS} if "margin_val" in a.files else None,
        y={s: a[f"y_{s}"] for s in SPLITS}, classes=meta["classes"],
        class_group={"global": np.zeros(n_classes, dtype=np.int64),
                     "category": a["group_category"].astype(np.int64),
                     "class": np.arange(n_classes)},
        minority=meta["minority"], benign_index=meta["benign_index"],
    )

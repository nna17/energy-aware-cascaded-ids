"""Tree models: single decision tree, LightGBM, XGBoost, Random Forest, with size and cost estimates.

`params["n_features"]` (optional) makes a model read only the first k ranked columns, which models a
device that extracts fewer flow features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Optional tuned hyper-parameters passed through to the libraries (absent = library defaults used so far).
LGB_EXTRA = ("learning_rate", "min_child_samples", "colsample_bytree", "subsample", "subsample_freq", "reg_lambda")
FOREST_EXTRA = ("max_features", "min_samples_leaf")

# Rough bytes per node for an embedded tree (feature id, threshold, child/leaf index).
# Only an estimate for comparing sizes; real flash use is measured in Phase 1.
BYTES_PER_NODE = 12


@dataclass
class TreeModel:
    kind: str
    params: dict[str, Any]
    seed: int = 0
    model: Any = field(default=None, repr=False)

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        k = self.params.get("n_features")
        return X[:, :k] if k else X

    def fit(self, X: np.ndarray, y: np.ndarray, balanced: bool = True) -> "TreeModel":
        p = self.params
        X = self._inputs(X)
        if self.kind == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_leaf_nodes=p["max_leaves"], class_weight="balanced" if balanced else None,
                random_state=self.seed,
            )
            self.model.fit(X, y)
        elif self.kind == "lightgbm":
            # min_child_weight=1: with balanced weights the default (1e-3) lets leaves form on a
            # handful of rare samples and training diverges past ~100 trees (macro-F1 0.61 -> 0.16).
            self.model = lgb.LGBMClassifier(
                n_estimators=p["n_estimators"], num_leaves=p["max_leaves"], min_child_weight=1.0,
                class_weight="balanced" if balanced else None,
                random_state=self.seed, n_jobs=-1, verbose=-1, **{k: p[k] for k in LGB_EXTRA if k in p},
            )
            self.model.fit(X, y)
        elif self.kind == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=p["n_estimators"], max_leaves=p["max_leaves"], max_depth=0,
                grow_policy="lossguide", tree_method="hist", random_state=self.seed, n_jobs=-1,
            )
            weights = compute_sample_weight("balanced", y) if balanced else None
            self.model.fit(X, y, sample_weight=weights)
        elif self.kind in ("random_forest", "extra_trees"):
            # Extra Trees: random split thresholds instead of the best split; same inference cost per tree.
            forest = RandomForestClassifier if self.kind == "random_forest" else ExtraTreesClassifier
            self.model = forest(
                n_estimators=p["n_estimators"], max_leaf_nodes=p["max_leaves"],
                class_weight="balanced" if balanced else None, random_state=self.seed, n_jobs=-1,
                **{k: p[k] for k in FOREST_EXTRA if k in p},
            )
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown tree kind {self.kind!r}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._inputs(X))).astype(np.int64).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._inputs(X)), dtype=np.float32)

    def _estimators(self) -> list:
        return [self.model] if self.kind == "decision_tree" else self.model.estimators_

    def n_nodes(self) -> int:
        if self.kind == "lightgbm":
            trees = self.model.booster_.dump_model()["tree_info"]
            return sum(2 * t["num_leaves"] - 1 for t in trees)
        if self.kind == "xgboost":
            return len(self.model.get_booster().trees_to_dataframe())
        return sum(est.tree_.node_count for est in self._estimators())

    def n_comparisons(self) -> int:
        """Worst-case threshold comparisons per prediction: the deepest path of every tree, summed.

        Multiclass boosting grows one tree per class per round, so it counts n_estimators x n_classes trees.
        """
        if self.kind == "lightgbm":
            def depth(node: dict) -> int:
                if "leaf_value" in node:
                    return 0
                return 1 + max(depth(node["left_child"]), depth(node["right_child"]))
            return sum(depth(t["tree_structure"]) for t in self.model.booster_.dump_model()["tree_info"])
        if self.kind == "xgboost":
            return sum(
                max(len(line) - len(line.lstrip("\t")) for line in tree.splitlines() if "leaf=" in line)
                for tree in self.model.get_booster().get_dump()
            )
        return sum(est.tree_.max_depth for est in self._estimators())

    def size_kb(self) -> float:
        return self.n_nodes() * BYTES_PER_NODE / 1024

    @property
    def name(self) -> str:
        n = "" if self.kind == "decision_tree" else f"_n{self.params['n_estimators']}"
        k = self.params.get("n_features")
        tag = self.params.get("tag")
        return f"{self.kind}{n}_l{self.params['max_leaves']}" + (f"_f{k}" if k else "") + (f"_{tag}" if tag else "")


def tree_grid(kinds: list[str], n_estimators: list[int], max_leaves: list[int], seed: int = 0) -> list[TreeModel]:
    return [
        TreeModel(kind, {"n_estimators": n, "max_leaves": leaves}, seed)
        for kind in kinds for n in n_estimators for leaves in max_leaves
    ]

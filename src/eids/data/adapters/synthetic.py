"""Imbalanced synthetic data with planted duplicates and a leaky ID column, for tests and debugging."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from eids.data.adapters.base import BENIGN_CATEGORY, FINE_LABEL, DatasetAdapter


class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"
    benign_label = "Benign"

    def load_raw(self) -> pd.DataFrame:
        n_classes = self.cfg.get("n_classes", 6)
        n_features = self.cfg.get("n_features", 30)
        # Benign dominates; the last classes are rare.
        weights = np.array([0.6] + [0.4 / (2 ** i) for i in range(n_classes - 1)])
        weights /= weights.sum()
        X, y = make_classification(
            n_samples=self.cfg.get("n_samples", 20000),
            n_features=n_features,
            n_informative=max(4, n_features // 2),
            n_redundant=n_features // 5,
            n_classes=n_classes,
            n_clusters_per_class=1,
            weights=weights.tolist(),
            class_sep=1.2,
            random_state=self.cfg.get("seed", 0),
        )
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(n_features)])
        df["proto"] = np.where(X[:, 0] > 0, "tcp", "udp")  # one categorical column
        names = [self.benign_label] + [f"Attack{i}" for i in range(1, n_classes)]
        df[FINE_LABEL] = [names[i] for i in y]
        df = pd.concat([df, df.iloc[:200]], ignore_index=True)  # planted exact duplicates
        df["flow_id"] = np.arange(len(df))  # planted leaky identifier
        return df

    def leaky_columns(self) -> list[str]:
        return ["flow_id"]

    def category_of(self, fine_label: str) -> str:
        if fine_label == self.benign_label:
            return BENIGN_CATEGORY
        return "Odd" if int(fine_label.removeprefix("Attack")) % 2 else "Even"

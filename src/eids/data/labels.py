"""Label levels (binary / category / fine), class encoding and the minority-class list."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eids.data.adapters.base import BENIGN_CATEGORY, FINE_LABEL, DatasetAdapter

LEVELS = ("binary", "category", "fine")
BINARY_BENIGN, BINARY_ATTACK = "Benign", "Attack"


def label_column(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"label level must be one of {LEVELS}")
    return f"label_{level}"


def add_label_levels(df: pd.DataFrame, adapter: DatasetAdapter) -> pd.DataFrame:
    fine = df[FINE_LABEL].astype(str).str.strip()
    mapping = {name: adapter.category_of(name) for name in fine.unique()}
    df = df.assign(label_fine=fine, label_category=fine.map(mapping))
    df["label_binary"] = np.where(df["label_category"] == BENIGN_CATEGORY, BINARY_BENIGN, BINARY_ATTACK)
    return df


def benign_name(level: str, adapter: DatasetAdapter) -> str:
    return {"binary": BINARY_BENIGN, "category": BENIGN_CATEGORY, "fine": adapter.benign_label}[level]


@dataclass
class LabelSpace:
    level: str
    classes: list[str]
    benign_index: int | None
    minority: list[int]  # indices of classes below the threshold share of the training set

    def encode(self, labels: pd.Series | np.ndarray) -> np.ndarray:
        index = {c: i for i, c in enumerate(self.classes)}
        return np.array([index[v] for v in labels], dtype=np.int64)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def build_label_space(train_labels: pd.Series, level: str, benign: str, minority_threshold: float) -> LabelSpace:
    counts = train_labels.value_counts()
    classes = sorted(counts.index.tolist())
    shares = counts / counts.sum()
    minority = [i for i, c in enumerate(classes) if shares[c] < minority_threshold and c != benign]
    return LabelSpace(
        level=level,
        classes=classes,
        benign_index=classes.index(benign) if benign in classes else None,
        minority=minority,
    )

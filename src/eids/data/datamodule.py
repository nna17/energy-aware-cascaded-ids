"""Processed arrays in memory, with fast batch iteration on the training device."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from eids.data.labels import LabelSpace
from eids.data.pipeline import SPLITS, prepare_dataset, processed_path
from eids.utils.config import Config
from eids.utils.io import load_json
from eids.utils.logging import get_logger

log = get_logger()


@dataclass
class DataModule:
    name: str
    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    labels: LabelSpace
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X["train"].shape[1]

    @property
    def n_classes(self) -> int:
        return self.labels.n_classes

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y["train"], minlength=self.n_classes)

    def tensors(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.X[split]).to(device), torch.from_numpy(self.y[split]).to(device))

    def batches(self, split: str, batch_size: int, device: torch.device, shuffle: bool = False,
                generator: torch.Generator | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        X, y = self.tensors(split, device)
        n = len(y)
        order = torch.randperm(n, generator=generator).to(device) if shuffle else torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield X[idx], y[idx]


def load_datamodule(cfg: Config, rebuild: bool = False) -> DataModule:
    path = processed_path(cfg)
    if rebuild or not (path / "arrays.npz").exists():
        log.info(f"[{cfg.dataset}] no processed cache at {path}; preparing")
        prepare_dataset(cfg)
    arrays = np.load(path / "arrays.npz")
    meta = load_json(path / "meta.json")
    labels = LabelSpace(meta["label_level"], meta["classes"], meta["benign_index"], meta["minority"])
    return DataModule(
        name=meta["dataset"],
        X={s: arrays[f"X_{s}"] for s in SPLITS},
        y={s: arrays[f"y_{s}"] for s in SPLITS},
        labels=labels,
        feature_names=meta["feature_names"],
    )

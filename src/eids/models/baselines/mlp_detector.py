"""Compact MLP detector with the same interface as TreeModel, so it can join the detector pool.

Cost is counted as multiply-accumulates per flow (the tree pool counts threshold comparisons; both are
reported as operations), and size assumes INT8 weights (one byte per parameter), as on MCU deployments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from eids.distill.class_weights import class_weights
from eids.models.mlp import MLP
from eids.utils.seed import resolve_device, set_seed


@dataclass
class MLPDetector:
    kind: str
    params: dict[str, Any]            # hidden: list[int], n_features: int | None, epochs, lr, batch_size
    seed: int = 0
    model: Any = field(default=None, repr=False)
    n_inputs: int = 0
    n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, balanced: bool = True) -> "MLPDetector":
        set_seed(self.seed)
        device = resolve_device("auto")
        self.n_inputs = self.params.get("n_features") or X.shape[1]
        self.n_classes = int(y.max()) + 1
        self.model = MLP(self.n_inputs, self.n_classes, list(self.params["hidden"])).to(device)
        counts = np.bincount(y, minlength=self.n_classes)
        weights = class_weights(counts, "inverse_frequency" if balanced else "none").to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.params.get("lr", 2e-3))
        Xt = torch.from_numpy(X[:, : self.n_inputs]).to(device)
        yt = torch.from_numpy(y.astype(np.int64)).to(device)
        bs = self.params.get("batch_size", 4096)
        gen = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.params.get("epochs", 10)):
            for idx in torch.randperm(len(yt), generator=gen).to(device).split(bs):
                opt.zero_grad(set_to_none=True)
                loss_fn(self.model(Xt[idx]), yt[idx]).backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        out = [torch.softmax(self.model(torch.from_numpy(X[i:i + 65536, : self.n_inputs]).to(device)), -1).cpu()
               for i in range(0, len(X), 65536)]
        return torch.cat(out).numpy().astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _layer_sizes(self) -> list[int]:
        return [self.n_inputs, *self.params["hidden"], self.n_classes]

    def n_comparisons(self) -> int:
        """Operations per flow: multiply-accumulates (BatchNorm folded into the linear layers)."""
        s = self._layer_sizes()
        return int(sum(a * b for a, b in zip(s[:-1], s[1:])))

    def size_kb(self) -> float:
        s = self._layer_sizes()
        return sum(a * b + b for a, b in zip(s[:-1], s[1:])) / 1024  # INT8 weights + biases

    @property
    def name(self) -> str:
        k = self.params.get("n_features")
        return "mlp_" + "x".join(map(str, self.params["hidden"])) + (f"_f{k}" if k else "")

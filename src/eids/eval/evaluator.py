"""Run any model on a split and return metrics. Knows nothing about how the model was trained."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eids.data.datamodule import DataModule
from eids.eval.metrics import compute_metrics
from eids.models.elastic import ElasticMLP, SubnetConfig
from eids.training.bn_recalib import recalibrate_bn


@torch.no_grad()
def predict(model: nn.Module, dm: DataModule, split: str, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    preds = [model(xb).argmax(dim=-1) for xb, _ in dm.batches(split, batch_size, device)]
    return torch.cat(preds).cpu().numpy()


def metrics_from_predictions(dm: DataModule, split: str, preds: np.ndarray) -> dict:
    return compute_metrics(dm.y[split], preds, dm.n_classes, dm.labels.minority, dm.labels.benign_index)


def evaluate(model: nn.Module, dm: DataModule, split: str, device: torch.device) -> dict:
    return metrics_from_predictions(dm, split, predict(model, dm, split, device))


def evaluate_subnet(supernet: ElasticMLP, cfg: SubnetConfig, dm: DataModule, split: str,
                    device: torch.device, recalib_batches: int = 50, batch_size: int = 1024) -> dict:
    """Activate a sub-model, recalibrate its BatchNorm statistics on training data, then evaluate."""
    supernet.set_active(cfg)
    recalibrate_bn(supernet, dm.batches("train", batch_size, device, shuffle=True), recalib_batches)
    return evaluate(supernet, dm, split, device)

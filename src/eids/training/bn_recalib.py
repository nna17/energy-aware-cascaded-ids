"""Recompute BatchNorm statistics for the active sub-model (weight sharing mixes them during training)."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


@torch.no_grad()
def recalibrate_bn(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], n_batches: int) -> None:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over the recalibration batches
    model.train()
    for i, (xb, _) in enumerate(batches):
        if i >= n_batches:
            break
        model(xb)
    for bn, m in zip(bns, momenta):
        bn.momentum = m
    model.eval()

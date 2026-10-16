"""Layers whose active size is a prefix of their full size, so sub-models share weights."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ElasticLinear(nn.Linear):
    """Uses weight[:active_out, :in], where `in` is the width of the incoming tensor."""

    def __init__(self, in_max: int, out_max: int):
        super().__init__(in_max, out_max)
        self.active_out = out_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight[: self.active_out, : x.shape[-1]], self.bias[: self.active_out])


class ElasticBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm over the first C channels. Running stats are per-prefix views, recalibrated per sub-model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
        use_batch_stats = self.training or not self.track_running_stats
        return F.batch_norm(
            x, self.running_mean[:c], self.running_var[:c], self.weight[:c], self.bias[:c],
            use_batch_stats, factor, self.eps,
        )

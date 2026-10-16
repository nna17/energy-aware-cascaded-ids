"""Weight-sharing MLP supernet: any SubnetConfig in its space can be activated without retraining."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from eids.models.elastic.layers import ElasticBatchNorm1d, ElasticLinear
from eids.models.elastic.subnet import SubnetConfig


class ElasticBlock(nn.Module):
    def __init__(self, in_max: int, out_max: int, dropout: float):
        super().__init__()
        self.linear = ElasticLinear(in_max, out_max)
        self.bn = ElasticBatchNorm1d(out_max)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.bn(self.linear(x))))


class ElasticMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, max_widths: list[int], dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_widths = list(max_widths)
        ins = [n_features] + self.max_widths[:-1]
        self.blocks = nn.ModuleList(ElasticBlock(i, o, dropout) for i, o in zip(ins, self.max_widths))
        self.head = ElasticLinear(max(self.max_widths), n_classes)
        self.active = SubnetConfig(len(max_widths), tuple(max_widths), n_features)

    def set_active(self, cfg: SubnetConfig) -> None:
        if cfg.depth > len(self.blocks) or cfg.n_features > self.n_features:
            raise ValueError(f"{cfg} exceeds the supernet")
        for block, width, max_width in zip(self.blocks, cfg.widths, self.max_widths):
            if width > max_width:
                raise ValueError(f"width {width} exceeds block max {max_width}")
            block.linear.active_out = width
        self.active = cfg

    @contextmanager
    def activated(self, cfg: SubnetConfig) -> Iterator["ElasticMLP"]:
        previous = self.active
        self.set_active(cfg)
        try:
            yield self
        finally:
            self.set_active(previous)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.active.n_features]
        for block in self.blocks[: self.active.depth]:
            x = block(x)
        return self.head(x)

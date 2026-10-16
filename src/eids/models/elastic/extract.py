"""Copy an active sub-model out of the supernet into a standalone MLP (for export and deployment)."""
from __future__ import annotations

import torch
from torch import nn

from eids.models.elastic.elastic_mlp import ElasticMLP
from eids.models.elastic.subnet import SubnetConfig
from eids.models.mlp import MLP


@torch.no_grad()
def extract_subnet(supernet: ElasticMLP, cfg: SubnetConfig) -> MLP:
    static = MLP(cfg.n_features, supernet.n_classes, hidden=list(cfg.widths), dropout=0.0)
    linears = [m for m in static.net if isinstance(m, nn.Linear)]
    bns = [m for m in static.net if isinstance(m, nn.BatchNorm1d)]

    prev = cfg.n_features
    for block, width, linear, bn in zip(supernet.blocks, cfg.widths, linears, bns):
        linear.weight.copy_(block.linear.weight[:width, :prev])
        linear.bias.copy_(block.linear.bias[:width])
        bn.weight.copy_(block.bn.weight[:width])
        bn.bias.copy_(block.bn.bias[:width])
        bn.running_mean.copy_(block.bn.running_mean[:width])
        bn.running_var.copy_(block.bn.running_var[:width])
        bn.eps = block.bn.eps
        prev = width

    head = linears[-1]
    head.weight.copy_(supernet.head.weight[:, :prev])
    head.bias.copy_(supernet.head.bias)
    return static.to(supernet.head.weight.device).eval()

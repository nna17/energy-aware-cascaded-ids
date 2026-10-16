"""Plain MLP shared by the MLP teacher and the fixed-student baseline."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Dropout blocks, then a linear head.

    `n_inputs` < number of data columns means the model reads only the first (most important) columns.
    """

    def __init__(self, n_inputs: int, n_classes: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        self.n_inputs = n_inputs
        layers: list[nn.Module] = []
        prev = n_inputs
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, : self.n_inputs])

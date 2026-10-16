"""1D-CNN over the feature vector, the model family Diab et al. search with HW-NAS.

Used only as an external baseline and in the permutation test. The exact architecture found by
their search should be set in the config once we reproduce their setup.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: list[int], kernel_size: int = 3, hidden: int = 128):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for c in channels:
            layers += [nn.Conv1d(prev, c, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.MaxPool1d(2, ceil_mode=True)]
            prev = c
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_features)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)))

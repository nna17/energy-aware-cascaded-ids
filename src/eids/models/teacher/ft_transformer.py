"""FT-Transformer (Gorishniy et al., 2021): each feature becomes a token, so no column order is assumed."""
from __future__ import annotations

import math

import torch
from torch import nn

from eids.utils.config import Config


class FeatureTokenizer(nn.Module):
    """token_i = x_i * W_i + b_i, one learned embedding per numeric feature."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        bound = 1 / math.sqrt(d_token)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias  # (batch, n_features, d_token)


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int, d_token: int = 64, n_blocks: int = 3,
                 n_heads: int = 8, ffn_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=int(d_token * ffn_factor),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0])


def build_ft_transformer(cfg: Config, n_features: int, n_classes: int) -> FTTransformer:
    return FTTransformer(
        n_features, n_classes,
        d_token=cfg.get("d_token", 64), n_blocks=cfg.get("n_blocks", 3), n_heads=cfg.get("n_heads", 8),
        ffn_factor=cfg.get("ffn_factor", 2.0), dropout=cfg.get("dropout", 0.1),
    )

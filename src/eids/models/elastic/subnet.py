"""Sub-model description and the space it is drawn from."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from eids.utils.config import Config


@dataclass(frozen=True)
class SubnetConfig:
    depth: int
    widths: tuple[int, ...]  # one width per active block
    n_features: int          # reads the first n_features (ranked) columns
    bits: int = 32

    def __post_init__(self):
        if len(self.widths) != self.depth:
            raise ValueError(f"widths {self.widths} must have length depth={self.depth}")

    def as_dict(self) -> dict:
        return asdict(self)


def param_count(cfg: SubnetConfig, n_classes: int) -> int:
    """Deployed parameters (BatchNorm folds into the preceding linear layer at export)."""
    total, prev = 0, cfg.n_features
    for w in cfg.widths:
        total += prev * w + w
        prev = w
    return total + prev * n_classes + n_classes


def macs(cfg: SubnetConfig, n_classes: int) -> int:
    """Multiply-accumulates per prediction (BatchNorm folded, ReLU and bias adds not counted)."""
    total, prev = 0, cfg.n_features
    for w in cfg.widths:
        total += prev * w
        prev = w
    return total + prev * n_classes


def size_kb(n_params: int, bits: int) -> float:
    return n_params * bits / 8 / 1024


def _choices(maximum: int, ratios: list[float]) -> list[int]:
    return sorted({max(1, round(r * maximum)) for r in ratios})


@dataclass
class SearchSpace:
    n_features: int
    max_widths: list[int]
    width_ratios: list[float]
    depths: list[int]
    feature_ratios: list[float]
    bits: list[int]

    @classmethod
    def from_config(cls, cfg: Config, n_features: int) -> "SearchSpace":
        return cls(n_features, list(cfg.max_widths), list(cfg.width_ratios), list(cfg.depths),
                   list(cfg.feature_ratios), list(cfg.get("bits", [32])))

    def width_choices(self, block: int) -> list[int]:
        return _choices(self.max_widths[block], self.width_ratios)

    def feature_choices(self) -> list[int]:
        return _choices(self.n_features, self.feature_ratios)

    def largest(self) -> SubnetConfig:
        d = max(self.depths)
        return SubnetConfig(d, tuple(self.max_widths[:d]), self.n_features, max(self.bits))

    def smallest(self) -> SubnetConfig:
        d = min(self.depths)
        return SubnetConfig(d, tuple(self.width_choices(i)[0] for i in range(d)),
                            self.feature_choices()[0], min(self.bits))

    def uniform(self, depth: int, width_ratio: float, feature_ratio: float, bits: int = 32) -> SubnetConfig:
        """Same width ratio in every block; used for named sub-models in experiment configs."""
        widths = tuple(max(1, round(width_ratio * self.max_widths[i])) for i in range(depth))
        return SubnetConfig(depth, widths, max(1, round(feature_ratio * self.n_features)), bits)

    def sample(self, rng: np.random.Generator) -> SubnetConfig:
        d = int(rng.choice(self.depths))
        widths = tuple(int(rng.choice(self.width_choices(i))) for i in range(d))
        return SubnetConfig(d, widths, int(rng.choice(self.feature_choices())), int(rng.choice(self.bits)))

"""Baseline B2: one fixed-size student per tier, trained separately with KD from the same teacher."""
from __future__ import annotations

from eids.models.elastic.subnet import SubnetConfig
from eids.models.mlp import MLP


def build_fixed_student(shape: SubnetConfig, n_classes: int) -> MLP:
    """Same shape as an elastic sub-model, so the comparison isolates weight sharing."""
    return MLP(shape.n_features, n_classes, hidden=list(shape.widths), dropout=0.0)

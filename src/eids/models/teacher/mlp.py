from eids.models.mlp import MLP
from eids.utils.config import Config


def build_mlp_teacher(cfg: Config, n_features: int, n_classes: int) -> MLP:
    return MLP(n_features, n_classes, hidden=list(cfg.hidden), dropout=cfg.get("dropout", 0.0))

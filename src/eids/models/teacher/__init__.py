from torch import nn

from eids.models.teacher.ensemble import TeacherEnsemble
from eids.models.teacher.ft_transformer import build_ft_transformer
from eids.models.teacher.mlp import build_mlp_teacher
from eids.utils.config import Config

_BUILDERS = {"mlp": build_mlp_teacher, "ft_transformer": build_ft_transformer}


def build_teacher(cfg: Config, n_features: int, n_classes: int) -> nn.Module:
    try:
        return _BUILDERS[cfg.kind](cfg, n_features, n_classes)
    except KeyError:
        raise ValueError(f"Unknown teacher kind {cfg.kind!r}; known: {sorted(_BUILDERS)}") from None


__all__ = ["build_teacher", "TeacherEnsemble"]
